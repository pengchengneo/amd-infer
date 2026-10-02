//! GPU quantized linear against independent CPU GGML decoding + f64 dot product.
use crate::gguf::Gguf;
#[cfg(feature = "hip")]
use std::process::Command;
use std::{error::Error, path::Path};
pub fn check(g: &Gguf, oracle: &str, dir: &Path) -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "hip"))]
    {
        let _ = (g, oracle, dir);
        Err("check-quant requires --features hip".into())
    }
    #[cfg(feature = "hip")]
    {
        if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
            return Err("GPU validation requires exclusive access; use tools/check-gpu.sh only after original engine exits".into());
        }
        std::fs::create_dir_all(dir)?;
        let input = dir.join("input.bin");
        let output = dir.join("output.bin");
        let mut counts = std::collections::BTreeMap::<u32, u64>::new();
        let mut calls = 0;
        let mut max_abs = 0f64;
        for t in g.tensors.iter().filter(|t| {
            t.dims.len() == 2 && t.dims[0].is_multiple_of(256) && !t.name.starts_with("blk.64.")
        }) {
            let cols = t.dims[0] as usize;
            let rows = t.dims[1];
            for row in [0, rows / 2, rows - 1] {
                let raw = g.read_rows(t, row, 1)?;
                std::fs::write(&input, &raw)?;
                if !Command::new(oracle)
                    .arg(t.kind.to_string())
                    .arg(cols.to_string())
                    .arg(&input)
                    .arg(&output)
                    .status()?
                    .success()
                {
                    return Err(format!("oracle failed for {}", t.name).into());
                }
                let b = std::fs::read(&output)?;
                if b.len() != cols * 4 {
                    return Err("bad oracle output length".into());
                }
                let w: Vec<f32> = b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                for batch in [1, 2, 4, 8] {
                    let x: Vec<f32> = (0..cols * batch)
                        .map(|i| (((i * 31 + row as usize * 7 + 19) % 509) as f32 - 254.) / 254.)
                        .collect();
                    let y = crate::hip::linear_quant(&raw, &x, cols, 1, batch, t.kind)?;
                    for n in 0..batch {
                        let reference = w
                            .iter()
                            .zip(&x[n * cols..(n + 1) * cols])
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>();
                        let error = (reference - y[n] as f64).abs();
                        let tolerance = 1e-4 + 2e-4 * reference.abs();
                        if !reference.is_finite() || !y[n].is_finite() || error > tolerance {
                            return Err(format!(
                                    "{} row={} type={} batch={} gpu={} reference={} error={} tolerance={}",
                                    t.name, row, t.kind, batch, y[n], reference, error, tolerance
                            )
                            .into());
                        }
                        max_abs = max_abs.max(error);
                    }
                    calls += 1;
                }
                *counts.entry(t.kind).or_default() += 1;
            }
        }
        println!("PASS GPU mixed-quant format_rows={counts:?} calls={calls} max_abs={max_abs}");
        Ok(())
    }
}
