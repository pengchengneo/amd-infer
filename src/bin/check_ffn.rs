use amd_infer::{gguf::Gguf, hip, ops};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    let g = Gguf::open(std::env::args().nth(1).ok_or("MODEL")?)?;
    let workspace = hip::LinearWorkspace::new(17408, 248320)?;
    let mut maximum = 0f32;
    let layers: Vec<usize> = if let Ok(list) = std::env::var("AMD_INFER_FFN_LAYERS") {
        let values = list
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<usize>, _>>()?;
        if values.is_empty() || values.iter().any(|v| *v >= 64) {
            return Err("FFN layers must be0..63".into());
        }
        values
    } else if std::env::var("AMD_INFER_ALL_FFN_LAYERS").as_deref() == Ok("1") {
        (0..64).collect()
    } else {
        vec![0, 17, 48, 63]
    };
    let layer_count = layers.len();
    let format_ab = std::env::var("AMD_INFER_FORMAT_AB").as_deref() == Ok("1");
    let case_count = if format_ab { 8 } else { 4 };
    for layer in layers {
        let names: [String; 3] = [
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ];
        let tensors: Vec<_> = names
            .iter()
            .map(|n| g.tensor(n))
            .collect::<Result<_, _>>()?;
        let sizes: Vec<_> = tensors
            .iter()
            .map(|t| t.bytes())
            .collect::<Result<_, _>>()?;
        println!("FFN layer={layer} kinds={:?} packed_bytes={} dims=gate/up[17408,5120],down[5120,17408] layout=row-major-GGUF", tensors.iter().map(|t| t.kind).collect::<Vec<_>>(), sizes.iter().sum::<u64>());
        if std::env::var("AMD_INFER_RESOURCE_QUERY").as_deref() == Ok("1") {
            match hip::gate_resources(tensors[0].kind,tensors[1].kind) {
                Ok(v)=>println!("FFN resources layer={layer} regs={} shared_bytes={} local_bytes={} max_active_blocks_per_MP={} max_threads_per_MP={} MPs={} warp_size={} max_block_threads={} scope=runtime-estimate-not-measured-occupancy",v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]),
                Err(e)=>println!("FFN resources layer={layer} unavailable={e}"),
            }
        }
        let storage = hip::PackedStorage::allocate(sizes.iter().sum::<u64>() as usize)?;
        let mut matrices = Vec::new();
        let mut offset = 0;
        for t in &tensors {
            let m = hip::PackedLinear::in_storage(
                &storage,
                offset,
                t.dims[0] as usize,
                t.dims[1] as usize,
                t.kind,
            )?;
            let raw = g.read_rows(t, 0, t.dims[1])?;
            m.upload_rows(0, &raw)?;
            offset += raw.len();
            matrices.push(m);
        }
        let norm = g.tensor(&format!("blk.{layer}.post_attention_norm.weight"))?;
        let raw = g.read_rows(norm, 0, 1)?;
        let weights: Vec<_> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let bank = if std::env::var("AMD_INFER_RESIDENT_FFN_NORM").as_deref() == Ok("1") {
            Some(hip::NormBank::new(&weights)?)
        } else {
            None
        };
        let mut ab_control: Option<Vec<f32>> = None;
        for case in 0..case_count {
            let candidate = format_ab && [false, true, false, true, true, false, false, true][case];
            if format_ab {
                unsafe {
                    if std::env::var("AMD_INFER_FFN_AB_KIND").as_deref() == Ok("fp32-regroup") {
                        std::env::set_var(
                            "AMD_INFER_FP32_REGROUP",
                            if candidate { "1" } else { "0" },
                        );
                    } else if std::env::var("AMD_INFER_FFN_AB_KIND").as_deref() == Ok("gemv-tune") {
                        let mode: usize = std::env::var("AMD_INFER_AB_TUNING")
                            .unwrap_or("1".into())
                            .parse()?;
                        if !(1..=7).contains(&mode) {
                            return Err("tuning candidate must be1..7".into());
                        }
                        std::env::set_var(
                            "AMD_INFER_GEMV_TUNE",
                            if candidate {
                                mode.to_string()
                            } else {
                                "0".into()
                            },
                        );
                    } else if std::env::var("AMD_INFER_FFN_AB_KIND").as_deref() == Ok("coop-matrix")
                    {
                        for flag in ["AMD_INFER_COOP_IQ3S", "AMD_INFER_COOP_QUANT_GEMV"] {
                            std::env::set_var(flag, if candidate { "1" } else { "0" });
                        }
                    } else {
                        std::env::set_var(
                            if matches!(
                                std::env::var("AMD_INFER_FFN_AB_KIND").as_deref(),
                                Ok("coop-gate-up" | "coop-matrix")
                            ) {
                                "AMD_INFER_COOP_GATE_UP"
                            } else {
                                "AMD_INFER_UNIFORM_ROW"
                            },
                            if candidate { "1" } else { "0" },
                        );
                    }
                }
            }
            let mut x: Vec<f32> = (0..5120)
                .map(|i| {
                    ((i * 31 + 7) as f32 * 0.013).sin()
                        * if !format_ab && case == 0 { 0.1 } else { 2. }
                })
                .collect();
            if !format_ab && case == 2 {
                x.fill(0.);
            }
            if !format_ab && case == 3 {
                x.fill(0.);
                x[17] = -20.;
                x[2] = 30.;
            }
            let n = ops::rms_norm(&x, &weights, 1e-6)?;
            let gate = matrices[0].run_workspace(&n, &workspace)?;
            let up = matrices[1].run_workspace(&n, &workspace)?;
            let act = ops::swiglu(&gate, &up)?;
            let expected = matrices[2].run_workspace(&act, &workspace)?;
            let before_stats = hip::kernel_stats();
            let before = before_stats.iter().find(|r| r.0 == 24).map_or(0, |r| r.1);
            let actual = hip::ffn(
                &matrices[0],
                &matrices[1],
                &matrices[2],
                &x,
                &weights,
                &workspace,
                bank.as_ref().map(|b| (b, 0)),
            )?;
            if format_ab
                && matches!(
                    std::env::var("AMD_INFER_FFN_AB_KIND").as_deref(),
                    Ok("coop-gate-up" | "coop-matrix" | "gemv-tune")
                )
            {
                if let Some(control) = &ab_control {
                    if control
                        .iter()
                        .zip(&actual)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        return Err(format!(
                            "FFN cooperative A/B bit mismatch layer={layer} case={case}"
                        )
                        .into());
                    }
                } else {
                    ab_control = Some(actual.clone());
                }
            }
            if format_ab && std::env::var("AMD_INFER_FFN_AB_KIND").as_deref() == Ok("fp32-regroup")
            {
                if let Some(control) = &ab_control {
                    let e = control
                        .iter()
                        .zip(&actual)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    println!(
                        "FFN regrouped numerical layer={layer} case={case} max_abs={e} gate=0.001"
                    );
                    if e > 0.001 {
                        return Err(format!(
                            "regrouped FFN gate layer={layer} case={case} error={e}"
                        )
                        .into());
                    }
                } else {
                    ab_control = Some(actual.clone());
                }
            }
            let after_stats = hip::kernel_stats();
            if format_ab {
                println!(
                    "FFN AB layer={layer} case={case} candidate={candidate} warmup={}",
                    case < 2
                );
            }
            let after = after_stats.iter().find(|r| r.0 == 24).map_or(0, |r| r.1);
            if after > before {
                let seconds = (after - before) as f64 / 1e9;
                println!("FFN event layer={layer} case={case} seconds={seconds} logical_payload_GB_s={}; not measured physical bandwidth, includes scheduling and cache effects", sizes.iter().sum::<u64>() as f64 / seconds / 1e9);
            }
            for stage in [26, 27, 28] {
                let time = |stats: &Vec<(u32, u64, usize)>| {
                    stats.iter().find(|r| r.0 == stage).map_or(0, |r| r.1)
                };
                let elapsed = time(&after_stats) - time(&before_stats);
                if elapsed > 0 {
                    let payload = if stage == 27 {
                        sizes[0] + sizes[1]
                    } else if stage == 28 {
                        sizes[2]
                    } else {
                        0
                    };
                    println!("FFN stage layer={layer} case={case} type={stage} seconds={} packed_payload_bytes={payload}; diagnostic only", elapsed as f64 / 1e9);
                }
            }
            for (a, b) in expected.iter().zip(actual) {
                let e = (a - b).abs();
                maximum = maximum.max(e);
                if !b.is_finite() || e > 1e-4 + 1e-4 * a.abs() {
                    return Err(format!("FFN layer={layer} case={case} {a} {b}").into());
                }
            }
        }
    }
    println!("PASS device FFN pipeline versus CPU RMS/SwiGLU + independently validated GEMV: {layer_count} real layers, {case_count} inputs each, {} values; max_abs={maximum}",layer_count*case_count*5120);
    Ok(())
}
