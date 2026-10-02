use amd_infer::{gguf::Gguf, hip, ops};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    let g = Gguf::open(std::env::args().nth(1).ok_or("MODEL")?)?;
    let vector = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let t = g.tensor(name)?;
        if t.kind != 0 {
            return Err("expected F32 vector".into());
        }
        Ok(g.read_rows(t, 0, t.elements()? / t.dims[0])?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let mut maximum = 0f32;
    for layer in [0, 17, 48, 62] {
        let conv_w = vector(&format!("blk.{layer}.ssm_conv1d.weight"))?;
        let norm_w = vector(&format!("blk.{layer}.ssm_norm.weight"))?;
        let a = vector(&format!("blk.{layer}.ssm_a"))?;
        let bias = vector(&format!("blk.{layer}.ssm_dt.bias"))?;
        let mut gpu = hip::GdnPipelineGpu::new(&conv_w, &norm_w, &a, &bias)?;
        let mut conv = ops::ConvState::new(10240)?;
        let mut delta = ops::DeltaState::new(128, 48, 16)?;
        for step in 0..192 {
            if step == 128 {
                gpu.reset()?;
                conv.reset();
                delta.reset();
            }
            let make = |n: usize, shift: usize, scale: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 17 + shift + step * 31) as f32 * 0.017).sin() * scale)
                    .collect()
            };
            let qkv = make(10240, 3, 0.08);
            let z = make(6144, 31, 0.7);
            let alpha = make(48, 71, 4.);
            let beta_raw = make(48, 107, 7.);
            let mixed = conv.step(&qkv, &conv_w)?;
            let mut q = Vec::new();
            let mut k = Vec::new();
            for h in 0..16 {
                q.extend(ops::l2_norm(&mixed[h * 128..(h + 1) * 128], 1e-6)?);
                k.extend(ops::l2_norm(
                    &mixed[2048 + h * 128..2048 + (h + 1) * 128],
                    1e-6,
                )?);
            }
            let gates: Vec<_> = alpha
                .iter()
                .zip(&bias)
                .zip(&a)
                .map(|((x, b), w)| ops::softplus(x + b) * w)
                .collect();
            let beta: Vec<_> = beta_raw.iter().copied().map(ops::sigmoid).collect();
            let core = delta.step(&q, &k, &mixed[4096..], &gates, &beta)?;
            let mut expected = Vec::with_capacity(6144);
            for h in 0..48 {
                let n = ops::rms_norm(&core[h * 128..(h + 1) * 128], &norm_w, 1e-6)?;
                for j in 0..128 {
                    expected.push(n[j] * z[h * 128 + j] * ops::sigmoid(z[h * 128 + j]));
                }
            }
            let actual = gpu.step(&qkv, &z, &alpha, &beta_raw)?;
            for (index, (x, y)) in expected.iter().zip(actual).enumerate() {
                let error = (x - y).abs();
                maximum = maximum.max(error);
                if !y.is_finite() || error > 2e-5 + 2e-4 * x.abs() {
                    return Err(format!("GDN layer={layer} step={step} value={index}: reference={x} gpu={y} error={error}").into());
                }
            }
        }
        println!(
            "PASS GDN pipeline actual weights layer={layer}, 128 uninterrupted +64 after reset"
        );
    }
    println!("PASS four actual GDN layers, 4718592 values, max_abs={maximum}; CPU conv/L2/Delta/RMS/gate reference");
    Ok(())
}
