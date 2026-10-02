use amd_infer::{hip, ops};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use first".into());
    }
    let make = |n: usize, shift: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 17 + shift) as f32 * 0.017).sin())
            .collect()
    };
    let mut max = 0f32;
    for n in [128, 5120, 17408] {
        let x = make(n, 1);
        let w = make(n, 11);
        let cpu = ops::rms_norm(&x, &w, 1e-6)?;
        let gpu = hip::rms(&x, &w)?;
        for (a, b) in cpu.iter().zip(gpu) {
            let e = (a - b).abs();
            max = max.max(e);
            if !b.is_finite() || e > 3e-5 + 1e-4 * a.abs() {
                return Err("RMS mismatch".into());
            }
        }
    }
    for count in [1, 7, 32, 128, 129, 255, 256, 257, 511, 512] {
        let q = make(6144, 3);
        let gate = make(6144, 41);
        let keys: Vec<_> = (0..count).map(|t| make(1024, t * 31 + 5)).collect();
        let values: Vec<_> = (0..count).map(|t| make(1024, t * 79 + 17)).collect();
        let gpu = hip::attention(&q, &keys, &values, &gate)?;
        for h in 0..24 {
            let kh = h / 6;
            let mut scores: Vec<f32> = keys
                .iter()
                .map(|k| {
                    q[h * 256..(h + 1) * 256]
                        .iter()
                        .zip(&k[kh * 256..(kh + 1) * 256])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / 16.
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = ops::activation_exp(*s - m);
            }
            let sum = scores.iter().sum::<f32>();
            for j in 0..256 {
                let cpu = scores
                    .iter()
                    .zip(&values)
                    .map(|(s, v)| s / sum * v[kh * 256 + j])
                    .sum::<f32>()
                    * ops::sigmoid(gate[h * 256 + j]);
                let b = gpu[h * 256 + j];
                let e = (cpu - b).abs();
                max = max.max(e);
                if !b.is_finite() || e > 3e-5 + 1e-4 * cpu.abs() {
                    return Err(format!("attention count={count} mismatch {cpu} {b}").into());
                }
            }
        }
    }
    let capacity: usize = std::env::var("AMD_INFER_CHECK_ATTN_CAPACITY")
        .unwrap_or("128".into())
        .parse()?;
    let mut cache = hip::AttentionGpu::with_capacity(capacity)?;
    for round in 0..2 {
        cache.reset();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for step in 0..capacity {
            let q = make(6144, step * 13 + 3);
            let gate = make(6144, step * 17 + 41);
            keys.push(make(1024, step * 31 + 5));
            values.push(make(1024, step * 79 + 17));
            let actual = cache.step(&q, &keys[step], &values[step], &gate)?;
            let expected = hip::attention(&q, &keys, &values, &gate)?;
            if actual != expected {
                return Err(format!("persistent attention round={round} step={step} not bit-identical to checked attention").into());
            }
        }
        if cache
            .step(
                &make(6144, 3),
                &make(1024, 5),
                &make(1024, 17),
                &make(6144, 41),
            )
            .is_ok()
        {
            return Err("cache capacity guard failed".into());
        }
    }
    println!("PASS GPU RMS and fused attention CPU oracle counts through512; persistent attention 2x{capacity} steps with reset matches existing checked GPU path bit-exact, capacity guard passed; max_abs={max}");
    Ok(())
}
