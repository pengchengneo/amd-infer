//! Serial B=1 fixture runner. Keeps one packed model loaded; separates phases.
use amd_infer::{gguf::Gguf, qwen::Qwen};
use std::{error::Error, fs, io::Write, path::Path, time::Instant};
fn save(path: impl AsRef<Path>, values: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(fs::File::create(path)?);
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 4 {
        return Err("evaluate MODEL TOKENIZER CASE_DIR RESULTS_DIR".into());
    }
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    fs::create_dir_all(&a[3])?;
    tokenizers::parallelism::set_parallelism(false);
    let tok = tokenizers::Tokenizer::from_file(&a[1]).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let g = Gguf::open(&a[0])?;
    let mut model = Qwen::new(&g)?;
    model.load_resident()?;
    let load = started.elapsed().as_secs_f64();
    let mut files: Vec<_> = fs::read_dir(&a[2])?
        .filter_map(|f| f.ok().map(|f| f.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    let output_tokens: usize = std::env::var("AMD_INFER_OUTPUT_TOKENS")
        .unwrap_or("8".into())
        .parse()?;
    if !(2..=512).contains(&output_tokens) {
        return Err("output tokens must be 2..512".into());
    }
    let context_capacity: usize = std::env::var("AMD_INFER_CONTEXT_CAPACITY")
        .unwrap_or("128".into())
        .parse()?;
    let teacher_root = std::env::var("AMD_INFER_TEACHER_TOKENS").ok();
    let save_trajectory = std::env::var("AMD_INFER_SAVE_TRAJECTORY").as_deref() == Ok("1");
    let case_warmup = std::env::var("AMD_INFER_CASE_WARMUP").as_deref() == Ok("1");
    let sample_seed: Option<u64> = std::env::var("AMD_INFER_SAMPLE_SEED")
        .ok()
        .map(|s| s.parse())
        .transpose()?;
    let repeats: usize = std::env::var("AMD_INFER_EVAL_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    let layouts: Option<Vec<bool>> = std::env::var("AMD_INFER_FFN_LAYOUT_SEQUENCE")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|v| match v {
                    "0" => Ok(false),
                    "1" => Ok(true),
                    _ => Err("layout sequence must contain 0 or 1"),
                })
                .collect()
        })
        .transpose()?;
    if let Some(sequence) = &layouts {
        if sequence.len() != repeats
            || ![6, 12].contains(&repeats)
            || sequence.iter().filter(|v| **v).count() * 2 != repeats
        {
            return Err("layout A/B requires six or twelve balanced trials".into());
        }
        if std::env::var("AMD_INFER_GPU_FFN").as_deref() != Ok("1") {
            return Err("layout A/B requires device FFN".into());
        }
    } else if !(1..=3).contains(&repeats) {
        return Err("repeats must be 1..3, or six for explicit layout A/B".into());
    }
    let mode_list = std::env::var("AMD_INFER_EVAL_MODES").unwrap_or("0,1".into());
    let ab_kind = std::env::var("AMD_INFER_AB_KIND").unwrap_or("layout".into());
    if layouts.is_some()
        && ![
            "layout",
            "down",
            "arithmetic",
            "blocks",
            "iq3s-packed",
            "attention-cache",
            "device-segments",
            "uniform-row",
            "local-graphs",
            "warp-gemv",
            "gdn-head-fuse",
            "compute-pair",
            "coop-gate-up",
            "coop-matrix",
            "gemv-tune",
            "fp32-regroup",
            "gdn-column-fuse",
            "gdn-column",
        ]
        .contains(&ab_kind.as_str())
    {
        return Err("unknown A/B kind".into());
    }
    let block_candidates: Option<Vec<usize>> = if ab_kind == "blocks" {
        let values = std::env::var("AMD_INFER_AB_BLOCKS")?
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != 2 || values.iter().any(|n| !(128..=4096).contains(n)) {
            return Err("block A/B requires two counts in 128..4096".into());
        }
        Some(values)
    } else {
        None
    };
    for mode in mode_list.split(',') {
        if mode != "0" && mode != "1" {
            return Err("GPU ops mode must be 0 or 1".into());
        }
        std::env::set_var("AMD_INFER_GPU_OPS", mode);
        model.reset_state()?;
        let t = Instant::now();
        let h = model.forward(9419, 64)?;
        model.logits(&h)?;
        let precondition: usize = std::env::var("AMD_INFER_PRECONDITION_TOKENS")
            .unwrap_or("0".into())
            .parse()?;
        if precondition > context_capacity {
            return Err("preconditioning exceeds context capacity".into());
        }
        if precondition > 0 {
            model.reset_state()?;
            for _ in 0..precondition {
                model.forward(9419, 64)?;
            }
            eprintln!(
                "sustained_warmup tokens={precondition} seconds={} scope=warmup-only-not-benchmark",
                t.elapsed().as_secs_f64()
            );
        }
        let warmup = t.elapsed().as_secs_f64();
        fs::write(
            Path::new(&a[3]).join("initial-warmup-graph.json"),
            format!("{:?}", amd_infer::hip::graph_stats(false)?),
        )?;
        for path in &files {
            let mut case_warmup_seconds = 0.;
            let mut layout_warmup_seconds = [0.; 2];
            let warmup_count = if case_warmup {
                if layouts.is_some() {
                    2
                } else {
                    1
                }
            } else {
                0
            };
            for iteration in 0..repeats + warmup_count {
                let is_warmup = iteration < warmup_count;
                let trial = iteration.saturating_sub(warmup_count);
                if let Some(sequence) = &layouts {
                    let row = if is_warmup {
                        iteration == 1
                    } else {
                        sequence[trial]
                    };
                    if ["gdn-column", "gdn-column-fuse"].contains(&ab_kind.as_str()) {
                        std::env::set_var("AMD_INFER_GDN_COLUMN", if row { "1" } else { "0" });
                        std::env::set_var(
                            "AMD_INFER_GDN_HEAD_FUSE",
                            if row && ab_kind == "gdn-column-fuse" {
                                "1"
                            } else {
                                "0"
                            },
                        );
                    } else if ab_kind == "fp32-regroup" {
                        std::env::set_var("AMD_INFER_FP32_REGROUP", if row { "1" } else { "0" });
                    } else if ab_kind == "gemv-tune" {
                        let mode: usize = std::env::var("AMD_INFER_AB_TUNING")
                            .unwrap_or("1".into())
                            .parse()?;
                        if !(1..=7).contains(&mode) {
                            return Err("tuning candidate must be1..7".into());
                        }
                        std::env::set_var(
                            "AMD_INFER_GEMV_TUNE",
                            if row { mode.to_string() } else { "0".into() },
                        );
                    } else if ab_kind == "coop-matrix" {
                        for flag in ["AMD_INFER_COOP_IQ3S", "AMD_INFER_COOP_QUANT_GEMV"] {
                            std::env::set_var(flag, if row { "1" } else { "0" });
                        }
                    } else if ab_kind == "coop-gate-up" {
                        std::env::set_var("AMD_INFER_COOP_GATE_UP", if row { "1" } else { "0" });
                    } else if ab_kind == "arithmetic" {
                        for flag in [
                            "AMD_INFER_STABLE_RMS",
                            "AMD_INFER_SCALAR_GDN",
                            "AMD_INFER_SCALAR_ATTENTION",
                            "AMD_INFER_SCALAR_FFN",
                            "AMD_INFER_GPU_ATTN_CACHE",
                        ] {
                            std::env::set_var(flag, if row { "1" } else { "0" });
                        }
                    } else if ["warp-gemv", "gdn-head-fuse", "compute-pair"]
                        .contains(&ab_kind.as_str())
                    {
                        if ab_kind != "gdn-head-fuse" {
                            std::env::set_var("AMD_INFER_WARP_GEMV", if row { "1" } else { "0" });
                        }
                        if ab_kind != "warp-gemv" {
                            std::env::set_var(
                                "AMD_INFER_GDN_HEAD_FUSE",
                                if row { "1" } else { "0" },
                            );
                        }
                    } else if ab_kind == "local-graphs" {
                        std::env::set_var("AMD_INFER_LOCAL_GRAPHS", if row { "1" } else { "0" });
                    } else if ab_kind == "uniform-row" {
                        std::env::set_var("AMD_INFER_UNIFORM_ROW", if row { "1" } else { "0" });
                    } else if ab_kind == "device-segments" {
                        std::env::set_var("AMD_INFER_DEVICE_SEGMENTS", if row { "1" } else { "0" });
                    } else if ab_kind == "attention-cache" {
                        std::env::set_var("AMD_INFER_GPU_ATTN_CACHE", if row { "1" } else { "0" });
                    } else if ab_kind == "iq3s-packed" {
                        std::env::set_var(
                            "AMD_INFER_IQ3S_PACKED_GRID",
                            if row { "1" } else { "0" },
                        );
                    } else if ab_kind == "blocks" {
                        std::env::set_var(
                            "AMD_INFER_PERSISTENT_FFN",
                            block_candidates.as_ref().unwrap()[usize::from(row)].to_string(),
                        );
                    } else if ab_kind == "down" {
                        std::env::set_var("AMD_INFER_DOWN_VARIANT", if row { "1" } else { "0" });
                    } else {
                        std::env::set_var("AMD_INFER_FFN_WARP_ROWS", if row { "1" } else { "0" });
                        std::env::set_var(
                            "AMD_INFER_PERSISTENT_FFN",
                            if row { "1024" } else { "2048" },
                        );
                    }
                }
                let warp_rows = std::env::var("AMD_INFER_FFN_WARP_ROWS").as_deref() == Ok("1");
                let candidate_variant = if let Some(sequence) = &layouts {
                    if is_warmup {
                        iteration == 1
                    } else {
                        sequence[trial]
                    }
                } else {
                    warp_rows
                };
                let name = path
                    .file_stem()
                    .ok_or("case name")?
                    .to_str()
                    .ok_or("UTF8 name")?;
                let text = fs::read_to_string(path)?;
                let teacher = teacher_root
                    .as_ref()
                    .map(|root| -> Result<Vec<u32>, Box<dyn Error>> {
                        let raw = fs::read_to_string(
                            Path::new(root).join(format!("{name}.generated.json")),
                        )?;
                        Ok(raw
                            .trim()
                            .trim_start_matches('[')
                            .trim_end_matches(']')
                            .split(',')
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| s.trim().parse())
                            .collect::<Result<Vec<_>, _>>()?)
                    })
                    .transpose()?;
                let output_tokens = teacher.as_ref().map_or(output_tokens, Vec::len);
                if !(2..=512).contains(&output_tokens)
                    || teacher
                        .as_ref()
                        .is_some_and(|ids| ids.iter().any(|id| *id >= 248320))
                {
                    return Err("invalid teacher token trajectory".into());
                }
                let decode_intervals = output_tokens - 1;
                let ids = tok
                    .encode(text.as_str(), false)
                    .map_err(|e| e.to_string())?
                    .get_ids()
                    .to_vec();
                if ids.is_empty() || ids.len() + output_tokens > context_capacity {
                    return Err("fixture exceeds configured context capacity".into());
                }
                let prefix = Path::new(&a[3]).join(if repeats == 1 {
                    format!("{name}.ops{mode}")
                } else {
                    format!("{name}.ops{mode}.trial{trial}")
                });
                fs::write(
                    format!("{}.prompt.tokens.json", prefix.display()),
                    format!("{ids:?}"),
                )?;
                fs::write(format!("{}.sampling.json", prefix.display()), format!("{{\"seed\":{},\"temperature\":0.7,\"top_k\":40,\"greedy\":{},\"output_tokens\":{output_tokens},\"note\":\"same-Rust-sampler A/B; not Vulkan sampler equivalence\"}}", sample_seed.map_or("null".into(), |s| s.to_string()), sample_seed.is_none()))?;
                fs::write(
                    format!("{}.layout.json", prefix.display()),
                    format!(
                        "{{\"warp_rows\":{warp_rows},\"candidate_variant\":{candidate_variant},\"ab_kind\":\"{ab_kind}\",\"down_variant\":{},\"persistent_blocks\":{},\"interleaved\":{}}}",
                        std::env::var("AMD_INFER_DOWN_VARIANT").unwrap_or("0".into()),
                        std::env::var("AMD_INFER_PERSISTENT_FFN").unwrap_or("0".into()),
                        layouts.is_some()
                    ),
                )?;
                fs::write(
                    format!("{}.preceding-graph-counters.json", prefix.display()),
                    format!("{:?}", amd_infer::hip::graph_stats(false)?),
                )?;
                let e2e = Instant::now();
                model.reset_state()?;
                amd_infer::hip::reset_memory_stats();
                amd_infer::hip::reset_kernel_stats();
                amd_infer::hip::reset_host_stats();
                amd_infer::hip::graph_stats(true)?;
                amd_infer::hip::gdn_profile_stats(true)?;
                amd_infer::hip::attention_profile_stats(true)?;
                let prefill = Instant::now();
                let mut hidden = Vec::new();
                for (position, id) in ids.iter().enumerate() {
                    hidden = model.forward(*id, 64)?;
                    if std::env::var("AMD_INFER_PROGRESS").as_deref() == Ok("1")
                        && (position + 1) % 64 == 0
                    {
                        eprintln!(
                            "prefill case={name} processed={} capacity={context_capacity}",
                            position + 1
                        );
                    }
                }
                let prefill_seconds = prefill.elapsed().as_secs_f64();
                let mut logits = model.logits(&hidden)?;
                let ttft = e2e.elapsed().as_secs_f64();
                let mut prompt_logits = None;
                let mut generated = Vec::new();
                let mut model_selected = Vec::new();
                let mut trajectory = Vec::new();
                let mut random_state = sample_seed.unwrap_or(0);
                let kernel_before = amd_infer::hip::kernel_stats();
                let gdn_before = amd_infer::hip::gdn_profile_stats(false)?;
                let attention_before = amd_infer::hip::attention_profile_stats(false)?;
                let host_before = amd_infer::hip::host_stats();
                let matrix_before = model.matrix_nanoseconds.get();
                let decode = Instant::now();
                let mut decode_compute = 0.;
                for step in 0..output_tokens {
                    if std::env::var("AMD_INFER_PROGRESS").as_deref() == Ok("1") && step % 16 == 0 {
                        eprintln!(
                            "decode case={name} step={step} processed_position={}",
                            ids.len() + step - 1
                        );
                    }
                    if logits.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite logits".into());
                    }
                    if save_trajectory {
                        trajectory.push(logits.clone());
                    }
                    let id = if sample_seed.is_some() {
                        amd_infer::sampling::sample(&logits, &mut random_state, 0.7, 40)?
                    } else {
                        logits
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .ok_or("empty logits")?
                            .0 as u32
                    };
                    if teacher.is_some() {
                        model_selected.push(id);
                    }
                    let id = teacher.as_ref().map_or(id, |ids| ids[step]);
                    generated.push(id);
                    if step == 0 {
                        prompt_logits = Some(std::mem::take(&mut logits));
                    }
                    if step == output_tokens - 1 {
                        break;
                    }
                    let t = Instant::now();
                    hidden = model.forward(id, 64)?;
                    logits = model.logits(&hidden)?;
                    decode_compute += t.elapsed().as_secs_f64();
                }
                let decode_wall = decode.elapsed().as_secs_f64();
                let wall = e2e.elapsed().as_secs_f64();
                if is_warmup {
                    case_warmup_seconds = wall;
                    layout_warmup_seconds[usize::from(candidate_variant)] = wall;
                    eprintln!("warmup case={name} full_request_seconds={wall}");
                    continue;
                }
                if layouts.is_some() {
                    case_warmup_seconds = layout_warmup_seconds[usize::from(candidate_variant)];
                }
                let gd = amd_infer::hip::gdn_profile_stats(false)?;
                let gd_delta: [u64; 18] = std::array::from_fn(|i| gd[i] - gdn_before[i]);
                let ad = amd_infer::hip::attention_profile_stats(false)?;
                let ad_delta: [u64; 14] = std::array::from_fn(|i| ad[i] - attention_before[i]);
                let hd = amd_infer::hip::host_stats();
                let hd_delta: Vec<[usize; 3]> = (0..4)
                    .map(|i| {
                        [
                            hd[i].0 - host_before[i].0,
                            hd[i].1 - host_before[i].1,
                            hd[i].2 - host_before[i].2,
                        ]
                    })
                    .collect();
                let kd: Vec<String> = amd_infer::hip::kernel_stats()
                    .iter()
                    .map(|(kind, ns, n)| {
                        let before = kernel_before.iter().find(|r| r.0 == *kind);
                        let n = n - before.map_or(0, |r| r.2);
                        let ns = ns - before.map_or(0, |r| r.1);
                        format!("type={kind} calls={n} event_seconds={}", ns as f64 / 1e9)
                    })
                    .collect();
                fs::write(format!("{}.decode-budget.json",prefix.display()),format!("{{\"intervals\":{},\"compute_seconds\":{decode_compute},\"wall_seconds\":{decode_wall},\"matrix_host_seconds\":{},\"gdn_parts_ns_calls\":{gd_delta:?},\"attention_parts_ns_calls\":{ad_delta:?},\"kernel_events\":{kd:?},\"host_calls_bytes_ns\":{hd_delta:?},\"scope\":\"Decode only; excludes prefill and initial LM head; events include enqueue/scheduling gaps; nested containers overlap; diagnostic flags required\"}}",output_tokens-1,(model.matrix_nanoseconds.get()-matrix_before) as f64/1e9))?;
                if save_trajectory {
                    for (step, values) in trajectory.iter().enumerate() {
                        save(
                            format!("{}.step{step}.logits.f32", prefix.display()),
                            values,
                        )?;
                    }
                    fs::write(format!("{}.trajectory.json",prefix.display()),format!("{{\"steps\":{},\"processed_tokens\":{},\"total_prompt_plus_output_tokens\":{},\"context_capacity\":{context_capacity},\"teacher_forced\":{},\"timing_includes_logit_copies\":true,\"diagnostic_only\":true}}",trajectory.len(),ids.len()+output_tokens-1,ids.len()+output_tokens,teacher.is_some()))?;
                }
                save(
                    format!("{}.prompt.logits.f32", prefix.display()),
                    prompt_logits.as_ref().ok_or("missing prompt logits")?,
                )?;
                fs::write(
                    format!("{}.gdn-parts.json", prefix.display()),
                    format!("{:?}", amd_infer::hip::gdn_profile_stats(false)?),
                )?;
                let graph = amd_infer::hip::graph_stats(false)?;
                fs::write(format!("{}.graph.json",prefix.display()),format!("{{\"captures\":{},\"graph_launches\":{},\"capture_setup_seconds\":{},\"failures\":{},\"reserve_or_limit_skips\":{},\"invalidations\":{},\"captured_nodes\":{},\"cached_graphs\":{},\"scope\":\"Measured request only; warmup setup excluded; cache survives counter reset\"}}",graph[0],graph[1],graph[2] as f64/1e9,graph[3],graph[4],graph[5],graph[6],graph[7]))?;
                let stats = amd_infer::hip::kernel_stats();
                let host = amd_infer::hip::host_stats();
                let layers = model
                    .layer_host_nanoseconds
                    .each_ref()
                    .map(|value| value.get() as f64 / 1e9);
                fs::write(format!("{}.layer-wall.json",prefix.display()),format!("{{\"attention_including_projection_prepare_cache\":{},\"gdn_including_projection_prepare_state\":{},\"ffn\":{},\"pre_attention_norm\":{},\"scope\":\"Non-overlapping host wall spans inside forwards, diagnostic flag required; not pure CPU time or device time\"}}",layers[0],layers[1],layers[2],layers[3]))?;
                fs::write(format!("{}.host.json",prefix.display()),format!("{{\"h2d\":{{\"calls\":{},\"bytes\":{},\"wall_seconds\":{}}},\"d2h\":{{\"calls\":{},\"bytes\":{},\"wall_seconds\":{}}},\"sync\":{{\"calls\":{},\"wall_seconds\":{}}},\"event_blocking_calls\":{{\"calls\":{},\"wall_seconds\":{}}},\"scope\":\"Explicit Rust-visible copies/device sync; event category includes timed-linear launch plus blocking event end; not pure synchronization overhead; other runtime calls unmeasured; diagnostic only\"}}",host[0].0,host[0].1,host[0].2 as f64/1e9,host[1].0,host[1].1,host[1].2 as f64/1e9,host[2].0,host[2].2 as f64/1e9,host[3].0,host[3].2 as f64/1e9))?;
                fs::write(
                    format!("{}.kernel.json", prefix.display()),
                    format!(
                        "{:?}",
                        stats
                            .iter()
                            .map(|(kind, ns, n)| format!(
                                "type={kind} calls={n} event_seconds={}",
                                *ns as f64 / 1e9
                            ))
                            .collect::<Vec<_>>()
                    ),
                )?;
                fs::write(format!("{}.profile.json",prefix.display()),format!("{{\"matrix_host_wall_seconds\":{},\"request_wall_seconds\":{wall},\"note\":\"matrix time includes read/copy/launch/sync; not HIP event kernel time\"}}",model.matrix_nanoseconds.get() as f64/1e9))?;
                let (live, peak, min_free) = amd_infer::hip::memory_stats();
                let min_free_text = if min_free == usize::MAX {
                    "null".into()
                } else {
                    min_free.to_string()
                };
                fs::write(format!("{}.memory.json",prefix.display()),format!("{{\"packed_resident_bytes\":{},\"requested_live_bytes\":{live},\"requested_peak_bytes\":{peak},\"observed_min_free_bytes\":{min_free_text},\"memory_trace\":{},\"note\":\"requested bytes exclude driver allocation rounding; min-free includes other desktop applications\"}}",model.resident_bytes,std::env::var("AMD_INFER_MEMORY_TRACE").as_deref()==Ok("1")))?;
                save(format!("{}.last.logits.f32", prefix.display()), &logits)?;
                if teacher.is_some() {
                    fs::write(
                        format!("{}.model_selected.json", prefix.display()),
                        format!("{model_selected:?}"),
                    )?;
                }
                fs::write(
                    format!("{}.generated.json", prefix.display()),
                    format!("{generated:?}"),
                )?;
                fs::write(
                    format!("{}.generated.txt", prefix.display()),
                    tok.decode(&generated, false).map_err(|e| e.to_string())?,
                )?;
                fs::write(format!("{}.metrics.json",prefix.display()),format!("{{\"mode_gpu_ops\":{mode},\"prompt_tokens\":{},\"output_tokens\":{output_tokens},\"load_seconds\":{load},\"warmup_seconds\":{warmup},\"case_warmup_seconds\":{case_warmup_seconds},\"prefill_forward_seconds\":{prefill_seconds},\"ttft_compute_seconds\":{ttft},\"decode_forward_lmhead_seconds\":{decode_compute},\"decode_intervals\":{decode_intervals},\"decode_wall_seconds\":{decode_wall},\"e2e_seconds\":{wall},\"decode_tokens_per_second\":{},\"e2e_output_tokens_per_second\":{},\"precision\":\"packed mixed quant weights; FP32 activations/KV/state; sequential prefill\",\"fixed_length_ignores_eos\":true}}",ids.len(),decode_intervals as f64/decode_compute,output_tokens as f64/wall))?;
                fs::write(format!("{}.execution.json",prefix.display()),format!("{{\"device_layer_calls\":{},\"mixed_layer_calls\":{},\"scope\":\"Executed full GDN/attention+FFN layers; counters reset per case\"}}",model.device_layer_calls.get(),model.mixed_layer_calls.get()))?;
                eprintln!(
                    "case={name} GPUops={mode} prompt={} decode_tok_s={:.3}",
                    ids.len(),
                    decode_intervals as f64 / decode_compute
                );
            }
        }
    }
    Ok(())
}
