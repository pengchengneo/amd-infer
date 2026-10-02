use amd_infer::{gguf::Gguf, hip};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    let g = Gguf::open(std::env::args().nth(1).ok_or("MODEL")?)?;
    let vector = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let t = g.tensor(name)?;
        if t.kind != 0 {
            return Err("expected F32".into());
        }
        Ok(g.read_rows(t, 0, t.elements()? / t.dims[0])?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let workspace = hip::LinearWorkspace::new(17408, 248320)?;
    let mut maximum = 0f32;
    for layer in [0, 17, 48, 62] {
        let names = ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"]
            .map(|n| format!("blk.{layer}.{n}.weight"));
        let ts = names
            .iter()
            .map(|n| g.tensor(n))
            .collect::<Result<Vec<_>, _>>()?;
        let sizes = ts
            .iter()
            .map(|t| t.bytes())
            .collect::<Result<Vec<_>, _>>()?;
        let storage = hip::PackedStorage::allocate(sizes.iter().sum::<u64>() as usize)?;
        let mut offset = 0;
        let mut matrices = Vec::new();
        for t in ts {
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
        let conv = vector(&format!("blk.{layer}.ssm_conv1d.weight"))?;
        let norm = vector(&format!("blk.{layer}.ssm_norm.weight"))?;
        let a = vector(&format!("blk.{layer}.ssm_a"))?;
        let bias = vector(&format!("blk.{layer}.ssm_dt.bias"))?;
        let mut reference = hip::GdnPipelineGpu::new(&conv, &norm, &a, &bias)?;
        let mut candidate = hip::GdnPipelineGpu::new(&conv, &norm, &a, &bias)?;
        for step in 0..64 {
            if step == 32 {
                reference.reset()?;
                candidate.reset()?;
            }
            let x: Vec<_> = (0..5120)
                .map(|i| ((i * 31 + step * 17) as f32 * 0.013).sin() * 0.5)
                .collect();
            let qkv = matrices[0].run_workspace(&x, &workspace)?;
            let z = matrices[1].run_workspace(&x, &workspace)?;
            let alpha = matrices[2].run_workspace(&x, &workspace)?;
            let beta = matrices[3].run_workspace(&x, &workspace)?;
            let gated = reference.step(&qkv, &z, &alpha, &beta)?;
            let expected = matrices[4].run_workspace(&gated, &workspace)?;
            let actual = candidate.step_projected(
                [
                    &matrices[0],
                    &matrices[1],
                    &matrices[2],
                    &matrices[3],
                    &matrices[4],
                ],
                &x,
                &workspace,
            )?;
            for (i, (a, b)) in expected.into_iter().zip(actual).enumerate() {
                let error = (a - b).abs();
                maximum = maximum.max(error);
                if !b.is_finite() || error > 1e-4 + 1e-4 * a.abs() {
                    return Err(format!(
                        "layer={layer} step={step} index={i} expected={a} actual={b}"
                    )
                    .into());
                }
            }
        }
        println!("PASS projected GDN layer={layer}, 32 uninterrupted +32 after reset");
    }
    println!("PASS projected GDN 1310720 values max_abs={maximum}; oracle=independent GEMV + validated GDN pipeline");
    Ok(())
}
