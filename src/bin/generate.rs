//! Bounded correctness-first greedy generation; raw text, no implicit chat template.
use amd_infer::{gguf::Gguf, qwen::Qwen};
use std::{error::Error, fs};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 5 {
        return Err(
            "usage: generate MODEL TOKENIZER_JSON PROMPT_FILE OUTPUT_FILE MAX_NEW_TOKENS".into(),
        );
    }
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use first; set AMD_INFER_GPU_EXCLUSIVE=1".into());
    }
    let limit: usize = a[4].parse()?;
    if !(1..=512).contains(&limit) {
        return Err("generation allows 1..512 new tokens".into());
    }
    let capacity: usize = std::env::var("AMD_INFER_CONTEXT_CAPACITY")
        .unwrap_or("128".into())
        .parse()?;
    if ![128, 256, 512].contains(&capacity) {
        return Err("context capacity must be 128, 256 or 512".into());
    }
    tokenizers::parallelism::set_parallelism(false);
    let tokenizer = tokenizers::Tokenizer::from_file(&a[1]).map_err(|e| e.to_string())?;
    let text = fs::read_to_string(&a[2])?;
    let ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    if ids.is_empty() || ids.len() + limit > capacity {
        return Err("prompt plus generation exceeds configured context capacity".into());
    }
    let load_started = std::time::Instant::now();
    let g = Gguf::open(&a[0])?;
    let eos = g
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.uint());
    let mut model = Qwen::new(&g)?;
    if std::env::var("AMD_INFER_RESIDENT").as_deref() == Ok("1")
        || std::env::var("AMD_INFER_PARTIAL_RESIDENCY").as_deref() == Ok("1")
    {
        model.load_resident()?;
    }
    let load_seconds = load_started.elapsed().as_secs_f64();
    let prompt_tokens = ids.len();
    let prefill_started = std::time::Instant::now();
    let mut hidden = Vec::new();
    for id in &ids {
        hidden = model.forward(*id, 64)?;
    }
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    let mut last_logits = Vec::new();
    let mut stopped_at_eos = false;
    let mut generated = Vec::new();
    let decode_started = std::time::Instant::now();
    for step in 0..limit {
        let logits = model.logits(&hidden)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("non-finite logits".into());
        }
        let id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty logits")?
            .0 as u32;
        last_logits = logits;
        generated.push(id);
        fs::write(
            &a[3],
            tokenizer
                .decode(&generated, false)
                .map_err(|e| e.to_string())?,
        )?;
        fs::write(format!("{}.tokens.json", &a[3]), format!("{generated:?}\n"))?;
        eprintln!("greedy step={step} token={id}; raw-text greedy generation");
        stopped_at_eos = eos == Some(id as u64);
        if stopped_at_eos || step + 1 == limit {
            break;
        }
        hidden = model.forward(id, 64)?;
    }
    let decode_wall_seconds = decode_started.elapsed().as_secs_f64();
    // Persist the final vocabulary once; streaming text and tokens remain per step.
    {
        use std::io::Write;
        let mut out =
            std::io::BufWriter::new(fs::File::create(format!("{}.last-logits.f32", &a[3]))?);
        for value in &last_logits {
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
    }
    fs::write(format!("{}.metrics.json", &a[3]),format!("{{\"prompt_tokens\":{prompt_tokens},\"output_tokens\":{},\"requested_max_new_tokens\":{limit},\"context_capacity\":{capacity},\"stopped_at_eos\":{stopped_at_eos},\"load_seconds\":{load_seconds},\"prefill_seconds\":{prefill_seconds},\"decode_wall_seconds\":{decode_wall_seconds},\"scope\":\"Raw text greedy CLI; decode wall includes initial LM head, sampling, streaming text/token writes; excludes final vocabulary write; no warmup; not evaluate throughput\"}}",generated.len()))?;
    eprintln!("generated_tokens={} context_capacity={} stopped_at_eos={} decode_wall_seconds={:.6}; raw-text greedy CLI including streaming writes", generated.len(), capacity, stopped_at_eos, decode_wall_seconds);
    Ok(())
}
