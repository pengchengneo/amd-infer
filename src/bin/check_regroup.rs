//! Independent arithmetic decision screen;not a model throughput claim.
use amd_infer::{gguf::Gguf, hip};
use std::{error::Error, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    let g = Gguf::open(std::env::args().nth(1).ok_or("MODEL")?)?;
    let ws = hip::LinearWorkspace::new(17408, 248320)?;
    let scrub = hip::CacheScrub::new()?;
    let mut seen = std::collections::BTreeSet::new();
    for t in &g.tensors {
        if t.dims.len() != 2 || ![21, 23].contains(&t.kind) || t.name.starts_with("blk.64.") {
            continue;
        }
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        if !(cols == 5120 && [17408, 10240, 6144].contains(&rows)
            || [6144, 17408].contains(&cols) && rows == 5120)
            || !seen.insert((t.kind, cols, rows))
        {
            continue;
        }
        let bytes = t.bytes()? as usize;
        let storage = hip::PackedStorage::allocate(bytes)?;
        let mat = hip::PackedLinear::in_storage(&storage, 0, cols, rows, t.kind)?;
        mat.upload_rows(0, &g.read_rows(t, 0, t.dims[1])?)?;
        let warm = Instant::now();
        let x: Vec<_> = (0..cols)
            .map(|i| ((i * 31 + 7) as f32 * 0.013).sin() * 2.0)
            .collect();
        while warm.elapsed().as_secs_f64() < 1.0 {
            scrub.run()?;
            let _ = mat.run_workspace(&x, &ws)?;
            let _ = mat.run_regrouped_probe(&x, &ws)?;
        }
        let reference = mat.run_workspace(&x, &ws)?;
        let mut maximum = 0f32;
        let mut changed = 0usize;
        for cold in [false, true] {
            for (trial, candidate) in [false, true, true, false, false, true]
                .into_iter()
                .enumerate()
            {
                for repeat in 0..5 {
                    if cold {
                        scrub.run()?;
                    }
                    let (y, ns) = if candidate {
                        mat.run_regrouped_probe(&x, &ws)?
                    } else {
                        let before = hip::kernel_stats()
                            .iter()
                            .find(|v| v.0 == t.kind)
                            .map_or(0, |v| v.1);
                        let y = mat.run_workspace(&x, &ws)?;
                        let after = hip::kernel_stats()
                            .iter()
                            .find(|v| v.0 == t.kind)
                            .map_or(0, |v| v.1);
                        (y, (after - before) as f64)
                    };
                    if y.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite regroup probe".into());
                    }
                    if candidate && cold && trial == 1 && repeat == 1 {
                        changed = y
                            .iter()
                            .zip(&reference)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                    }
                    let error = y
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    maximum = maximum.max(error);
                    if repeat > 0 {
                        println!("PROBE kind={} cols={cols} rows={rows} cold={cold} trial={trial} candidate={candidate} ns={ns} max_abs={error}",t.kind);
                    }
                }
            }
        }
        println!("QUALITY kind={} cols={cols} rows={rows} changed_bits={changed} max_abs={maximum} input_amplitude=2.0 original_absolute_gate=0.001 gate_changed=false",t.kind);
    }
    Ok(())
}
