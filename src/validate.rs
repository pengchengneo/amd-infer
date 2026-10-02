use crate::{gguf::Gguf, quant::iq4_xs};
use std::{error::Error, path::Path, process::Command};
pub fn check_iq4(g: &Gguf, oracle: &str, dir: &Path, gpu: bool) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let raw_path = dir.join("oracle-input.bin");
    let out_path = dir.join("oracle-output.bin");
    let mut checked = 0;
    let mut values = 0;
    let mut global_abs = 0f32;
    for t in g.tensors.iter().filter(|t| t.kind == 23) {
        let width = t.dims[0] as usize;
        let rows = t.elements()? / t.dims[0];
        for r in [0, rows / 2, rows - 1] {
            let raw = g.read_rows(t, r, 1)?;
            std::fs::write(&raw_path, &raw)?;
            if !Command::new(oracle)
                .arg("23")
                .arg(width.to_string())
                .arg(&raw_path)
                .arg(&out_path)
                .status()?
                .success()
            {
                return Err("upstream oracle failed".into());
            }
            let ref_bytes = std::fs::read(&out_path)?;
            if ref_bytes.len() != width * 4 {
                return Err("oracle output length".into());
            }
            let reference: Vec<f32> = ref_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let mut decoded = vec![0.; width];
            for (b, y) in raw
                .as_chunks::<136>()
                .0
                .iter()
                .zip(decoded.as_chunks_mut::<256>().0.iter_mut())
            {
                iq4_xs(b, y)?;
            }
            for (a, b) in decoded.iter().zip(&reference) {
                if !a.is_finite() || !b.is_finite() {
                    return Err("nonfinite decoded weight".into());
                }
                let abs = (a - b).abs();
                global_abs = global_abs.max(abs);
                if abs > 1e-7 {
                    return Err(
                        format!("{} row {} dequant mismatch: {} vs {}", t.name, r, a, b).into(),
                    );
                }
            }
            if gpu {
                #[cfg(feature = "hip")]
                for batch in [1, 2, 4, 8] {
                    let x: Vec<f32> = (0..width * batch)
                        .map(|i| (((i * 17 + 13) % 257) as f32 - 128.) / 128.)
                        .collect();
                    let y = crate::hip::linear_iq4(&raw, &x, width, 1, batch)?;
                    for b in 0..batch {
                        let expected = reference
                            .iter()
                            .zip(&x[b * width..(b + 1) * width])
                            .map(|(w, v)| *w as f64 * *v as f64)
                            .sum::<f64>();
                        let err = (y[b] as f64 - expected).abs();
                        let tol = 1e-4 + expected.abs() * 2e-4;
                        if !y[b].is_finite() || err > tol {
                            return Err(format!(
                                "GPU {} row {} batch {} error={} tolerance={}",
                                t.name, r, batch, err, tol
                            )
                            .into());
                        }
                    }
                }
                #[cfg(not(feature = "hip"))]
                return Err("--gpu requires --features hip".into());
            }
            checked += 1;
            values += width;
        }
    }
    if checked == 0 {
        return Err("no IQ4_XS tensors found".into());
    }
    println!(
        "PASS IQ4_XS tensors={} sampled_rows={} values={} cpu_max_abs={} gpu={}",
        checked / 3,
        checked,
        values,
        global_abs,
        gpu
    );
    Ok(())
}
