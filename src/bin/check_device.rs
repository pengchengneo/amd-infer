//! Fixed-history hidden/logit oracle, single layer then contiguous and full graph.
use amd_infer::{gguf::Gguf, qwen::Qwen};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("check-device MODEL".into());
    }
    let g = Gguf::open(&args[0])?;
    let mut model = Qwen::new(&g)?;
    model.load_resident()?;
    let uniform = std::env::var("AMD_INFER_CHECK_UNIFORM_ROW").as_deref() == Ok("1");
    let column = std::env::var("AMD_INFER_CHECK_COLUMN").as_deref() == Ok("1");
    let graphs = std::env::var("AMD_INFER_CHECK_GRAPHS").as_deref() == Ok("1");
    let compute = std::env::var("AMD_INFER_CHECK_COMPUTE").ok();
    let tokens = [9419, 11, 100, 200, 500, 1000, 2000, 3000];
    for layers in [1, 3, 4, 64] {
        unsafe {
            std::env::set_var(
                "AMD_INFER_DEVICE_SEGMENTS",
                if uniform || graphs || column || compute.is_some() {
                    "1"
                } else {
                    "0"
                },
            );
            if column {
                std::env::set_var("AMD_INFER_GDN_COLUMN", "0");
                std::env::set_var("AMD_INFER_GDN_HEAD_FUSE", "0");
            }
            if graphs {
                std::env::set_var("AMD_INFER_LOCAL_GRAPHS", "0");
                std::env::set_var("AMD_INFER_UNIFORM_ROW", "1");
            }
            if compute.is_some() {
                std::env::set_var("AMD_INFER_UNIFORM_ROW", "1");
                std::env::set_var("AMD_INFER_WARP_GEMV", "0");
                std::env::set_var("AMD_INFER_GDN_HEAD_FUSE", "0");
            }
            if uniform {
                std::env::set_var("AMD_INFER_UNIFORM_ROW", "0");
            }
        }
        model.reset_state()?;
        let mut expected = Vec::new();
        for token in tokens {
            let h = model.forward(token, layers)?;
            let logits = model.logits(&h)?;
            expected.push((h, logits, model.recurrent_state_digests(layers)?));
        }
        for trial in 0..3 {
            unsafe {
                std::env::set_var("AMD_INFER_DEVICE_SEGMENTS", "1");
                if column {
                    std::env::set_var("AMD_INFER_GDN_COLUMN", "1");
                    std::env::set_var("AMD_INFER_GDN_HEAD_FUSE", "1");
                }
                if graphs {
                    std::env::set_var("AMD_INFER_LOCAL_GRAPHS", "1");
                }
                if let Some(mode) = &compute {
                    std::env::set_var(
                        "AMD_INFER_WARP_GEMV",
                        if mode != "head" { "1" } else { "0" },
                    );
                    std::env::set_var(
                        "AMD_INFER_GDN_HEAD_FUSE",
                        if mode != "gemv" { "1" } else { "0" },
                    );
                }
                if uniform {
                    std::env::set_var("AMD_INFER_UNIFORM_ROW", "1");
                }
                if trial == 2 {
                    std::env::set_var("AMD_INFER_DEVICE_SKIP_LAYERS", "2,17,31");
                } else {
                    std::env::remove_var("AMD_INFER_DEVICE_SKIP_LAYERS");
                }
            }
            model.reset_state()?;
            amd_infer::hip::graph_stats(true)?;
            let mut max = 0f32;
            let mut exact = true;
            for (step, token) in tokens.into_iter().enumerate() {
                // Reset captured column layout must survive a mid-request env change.
                if column && trial == 1 && step == 4 {
                    std::env::set_var("AMD_INFER_GDN_COLUMN", "0");
                }
                let h = model.forward(token, layers)?;
                let logits = model.logits(&h)?;
                if model.recurrent_state_digests(layers)? != expected[step].2 {
                    return Err(format!("recurrent/conv state digest mismatch layers={layers} trial={trial} step={step}").into());
                }
                for (label, a, b) in [
                    ("hidden", &h, &expected[step].0),
                    ("logits", &logits, &expected[step].1),
                ] {
                    let error = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    max = max.max(error);
                    exact &= a.len() == b.len()
                        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
                    if error > 0.001 {
                        return Err(format!(
                            "layers={layers} trial={trial} step={step} tensor={label} max={error}"
                        )
                        .into());
                    }
                }
            }
            if graphs {
                let counters = amd_infer::hip::graph_stats(false)?;
                if counters[1] == 0 || counters[3] != 0 {
                    return Err(format!("graph not executed or failed: {counters:?}").into());
                }
                eprintln!("GRAPH layers={layers} trial={trial} counters={counters:?}");
            }
            println!("PASS layers={layers} trial={trial} steps=8 hidden+full_logits max_abs={max} byte_exact={exact} recurrent_conv_digest_exact=true forced_fallback={}",trial==2);
        }
    }
    if graphs && std::env::var("AMD_INFER_CHECK_GRAPH_RELOAD").as_deref() == Ok("1") {
        unsafe {
            std::env::set_var("AMD_INFER_LOCAL_GRAPHS", "0");
        }
        model.reset_state()?;
        let expected = model.forward(9419, 64)?;
        let logits = model.logits(&expected)?;
        let states = model.recurrent_state_digests(64)?;
        drop(model);
        let after_drop = amd_infer::hip::graph_stats(false)?;
        if after_drop[7] != 0 || after_drop[5] == 0 {
            return Err(
                format!("referenced graph cache survived model drop: {after_drop:?}").into(),
            );
        }
        let mut reloaded = Qwen::new(&g)?;
        reloaded.load_resident()?;
        unsafe {
            std::env::set_var("AMD_INFER_LOCAL_GRAPHS", "1");
        }
        amd_infer::hip::graph_stats(true)?;
        reloaded.reset_state()?;
        let h = reloaded.forward(9419, 64)?;
        let actual = reloaded.logits(&h)?;
        for (a, b) in [(&h, &expected), (&actual, &logits)] {
            if a.len() != b.len() || !a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                return Err("model reload graph output differs from ordinary dispatch".into());
            }
        }
        if reloaded.recurrent_state_digests(64)? != states {
            return Err("model reload graph recurrent/conv digest mismatch".into());
        }
        let counters = amd_infer::hip::graph_stats(false)?;
        if counters[0] != 112 || counters[1] != 112 || counters[3] != 0 || counters[4] != 0 {
            return Err(format!(
                "model reload did not freshly capture all local segments: {counters:?}"
            )
            .into());
        }
        println!("PASS model-drop-reload bit_exact=true recurrent_conv_digest_exact=true graphs_invalidated=true counters={counters:?}");
    }
    Ok(())
}
