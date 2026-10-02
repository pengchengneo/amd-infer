use amd_infer::gguf::{type_name, Gguf};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let cmd = a.next().unwrap_or_default();
    #[cfg(feature = "hip")]
    if cmd == "diagnose-read-bandwidth" {
        if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
            return Err("coordinate GPU use".into());
        }
        println!("{{\"payload_bytes\":268435456,\"warmup\":1,\"trials\":8,\"logical_GB_s\":{:?},\"note\":\"read diagnostic only, includes scheduling/cache effects; not engine throughput or physical bandwidth counters\"}}", amd_infer::hip::diagnose_read_bandwidth()?);
        return Ok(());
    }
    #[cfg(feature = "hip")]
    if cmd == "check-delta" {
        if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
            return Err("coordinate GPU use first".into());
        }
        let mut cpu = amd_infer::ops::DeltaState::new(128, 48, 16)?;
        let mut gpu = amd_infer::hip::DeltaGpu::new()?;
        let mut maximum = 0f32;
        let steps: usize = std::env::var("AMD_INFER_DELTA_STEPS")
            .unwrap_or("128".into())
            .parse()?;
        if ![128, 256, 512].contains(&steps) {
            return Err("Delta oracle steps must be128/256/512".into());
        }
        let mut state_maximum = 0f32;
        let mut state_values = 0usize;
        for step in 0..steps {
            if step == steps / 2 && std::env::var("AMD_INFER_DELTA_NO_RESET").as_deref() != Ok("1")
            {
                cpu.reset();
                gpu.reset()?;
            }
            let make = |n: usize, shift: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 17 + shift + step * 13) as f32 * 0.013).sin() * 0.08)
                    .collect()
            };
            let q = make(2048, 3);
            let k = make(2048, 31);
            let v = make(6144, 71);
            let g: Vec<_> = (0..48)
                .map(|h| -0.01 * (1 + (h + step) % 9) as f32)
                .collect();
            let beta: Vec<_> = (0..48)
                .map(|h| 0.1 + 0.08 * ((h + step) % 10) as f32)
                .collect();
            let expected = cpu.step(&q, &k, &v, &g, &beta)?;
            let actual = gpu.step(&q, &k, &v, &g, &beta)?;
            for (a, b) in expected.iter().zip(&actual) {
                let error = (a - b).abs();
                maximum = maximum.max(error);
                if !b.is_finite() || error > 1e-5 + 1e-4 * a.abs() {
                    return Err(format!(
                        "GPU DeltaNet numerical mismatch at step {step}: {a} versus {b}"
                    )
                    .into());
                }
            }
            if std::env::var("AMD_INFER_DELTA_CHECK_STATE").as_deref() == Ok("1")
                && (step + 1) % 128 == 0
            {
                let actual = gpu.snapshot_state()?;
                for (a, b) in cpu.data.iter().zip(actual) {
                    let error = (a - b).abs();
                    state_maximum = state_maximum.max(error);
                    if !b.is_finite() || error > 1e-5 + 1e-4 * a.abs() {
                        return Err(format!("Delta state mismatch step={step} {a} {b}").into());
                    }
                }
                state_values += cpu.data.len();
            }
        }
        println!(
            "PASS fused GPU DeltaNet {steps} steps, reset_at_half={}, {} output values; max_abs={maximum}; state_values={state_values} state_max_abs={state_maximum}",
            std::env::var("AMD_INFER_DELTA_NO_RESET").as_deref() != Ok("1"),steps*6144
        );
        return Ok(());
    }
    #[cfg(feature = "hip")]
    if cmd == "gpu-memory" {
        let (free, total) = amd_infer::hip::memory()?;
        println!("HIP free_bytes={free} total_bytes={total}; instantaneous, not a reservation");
        return Ok(());
    }
    let path = a
        .next()
        .ok_or("usage: amd-infer inspect MODEL.gguf [--tensors]")?;
    let g = Gguf::open(path)?;
    match cmd.as_str() {
        "inspect" => {
            println!(
                "file_bytes={} data_offset={} tensor_count={}",
                g.file_bytes,
                g.data_offset,
                g.tensors.len()
            );
            let mut kinds = std::collections::BTreeMap::<u32, (u64, u64)>::new();
            for t in &g.tensors {
                let e = kinds.entry(t.kind).or_default();
                e.0 += 1;
                e.1 += t.bytes()?;
            }
            for (k, (n, b)) in kinds {
                println!(
                    "type={} id={} count={} bytes={} MiB={:.3}",
                    type_name(k),
                    k,
                    n,
                    b,
                    b as f64 / 1048576.
                );
            }
            for key in [
                "general.architecture",
                "general.file_type",
                "qwen35.block_count",
                "qwen35.embedding_length",
                "qwen35.attention.head_count",
                "qwen35.attention.head_count_kv",
                "qwen35.ssm.state_size",
                "qwen35.ssm.group_count",
                "qwen35.ssm.inner_size",
            ] {
                if let Some(v) = g.metadata.get(key) {
                    println!("{key}={v:?}")
                }
            }
            if a.any(|x| x == "--tensors") {
                for t in &g.tensors {
                    println!(
                        "{} {} {:?} {}",
                        t.name,
                        type_name(t.kind),
                        t.dims,
                        t.bytes()?
                    );
                }
            }
        }
        "budget" => amd_infer::budget::report(&g)?,
        #[cfg(feature = "hip")]
        "forward" => {
            let ids_path = a
                .next()
                .ok_or("forward MODEL IDS_JSON LAYERS OUTPUT_PREFIX")?;
            let layers: usize = a.next().ok_or("layers")?.parse()?;
            let prefix = a.next().ok_or("output prefix")?;
            let raw = std::fs::read_to_string(ids_path)?;
            let ids: Result<Vec<u32>, _> = raw
                .trim()
                .trim_matches(['[', ']'])
                .split(',')
                .filter(|x| !x.trim().is_empty())
                .map(|x| x.trim().parse())
                .collect();
            let ids = ids?;
            if ids.is_empty() || ids.len() > 128 {
                return Err("forward validation allows 1..128 tokens".into());
            }
            let mut model = amd_infer::qwen::Qwen::new(&g)?;
            if std::env::var("AMD_INFER_RESIDENT").as_deref() == Ok("1") {
                model.load_resident()?;
            }
            let mut hidden = Vec::new();
            for id in ids {
                hidden = model.forward(id, layers)?;
            }
            fn save(path: String, x: &[f32]) -> std::io::Result<()> {
                let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
                use std::io::Write;
                for v in x {
                    f.write_all(&v.to_le_bytes())?
                }
                f.flush()
            }
            save(format!("{prefix}.hidden.f32"), &hidden)?;
            if layers == 64 {
                let logits = model.logits(&hidden)?;
                let top = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .ok_or("no logits")?
                    .0;
                let mode = if std::env::var("AMD_INFER_RESIDENT").as_deref() == Ok("1") {
                    "resident packed matrices + CPU state"
                } else {
                    "streaming matrices + CPU state"
                };
                println!(
                    "Rust/HIP correctness path ({mode}) top_token={top}; not a throughput baseline"
                );
                save(format!("{prefix}.logits.f32"), &logits)?;
            }
        }
        #[cfg(feature = "hip")]
        "debug-quant" => {
            let name = a.next().ok_or("tensor name")?;
            let t = g.tensor(&name)?;
            let raw = g.read_rows(t, 0, 1)?;
            let y = amd_infer::hip::debug_quant(&raw, t.kind)?;
            println!("debug {:?}", &y[..20]);
        }
        "check-quant" => {
            let oracle = a.next().ok_or("check-quant MODEL ORACLE_EXE RESULTS_DIR")?;
            let dir = a.next().ok_or("missing results directory")?;
            amd_infer::validate_quant::check(&g, &oracle, std::path::Path::new(&dir))?;
        }
        "check-iq4" => {
            let oracle = a
                .next()
                .ok_or("check-iq4 MODEL ORACLE_EXE RESULTS_DIR [--gpu]")?;
            let dir = a.next().ok_or("missing results directory")?;
            let gpu = a.any(|x| x == "--gpu");
            amd_infer::validate::check_iq4(&g, &oracle, std::path::Path::new(&dir), gpu)?;
        }
        _ => return Err("unknown command".into()),
    };
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1)
    }
}
