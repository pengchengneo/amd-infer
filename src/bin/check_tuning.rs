//! Development screen: exact row results and interleaved hot/cold kernel windows.
use amd_infer::{gguf::Gguf, hip};
use std::error::Error;
fn main() -> Result<(), Box<dyn Error>> {
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU use".into());
    }
    let g = Gguf::open(std::env::args().nth(1).ok_or("MODEL")?)?;
    let workspace = hip::LinearWorkspace::new(17408, 248320)?;
    let scrub = hip::CacheScrub::new()?;
    let mut seen = std::collections::BTreeSet::new();
    let mut tensors = Vec::new();
    for t in &g.tensors {
        if t.dims.len() == 2 && [21, 23].contains(&t.kind) && !t.name.starts_with("blk.64.") {
            let k = (t.kind, t.dims[0], t.dims[1]);
            if (t.dims[0] == 5120 && [17408, 10240].contains(&t.dims[1])
                || t.dims[0] == 17408 && t.dims[1] == 5120)
                && seen.insert(k)
            {
                tensors.push(t);
            }
        }
    }
    for t in tensors {
        let cols = t.dims[0] as usize;
        let rows = t.dims[1] as usize;
        let bytes = t.bytes()? as usize;
        let storage = hip::PackedStorage::allocate(bytes)?;
        let matrix = hip::PackedLinear::in_storage(&storage, 0, cols, rows, t.kind)?;
        matrix.upload_rows(0, &g.read_rows(t, 0, t.dims[1])?)?;
        let x: Vec<_> = (0..cols)
            .map(|i| ((i * 31 + 7) as f32 * 0.013).sin() * 0.15)
            .collect();
        unsafe {
            std::env::set_var("AMD_INFER_GEMV_TUNE", "0");
        }
        let reference = matrix.run_workspace(&x, &workspace)?;
        for mode in 1..=7 {
            unsafe {
                std::env::set_var("AMD_INFER_GEMV_TUNE", mode.to_string());
            }
            let resource = hip::linear_resources(t.kind, rows)?;
            println!(
                "RESOURCE kind={} cols={cols} rows={rows} mode={mode} values={resource:?}",
                t.kind
            );
            for cold in [false, true] {
                for (trial, variant) in [0, mode, mode, 0, 0, mode].into_iter().enumerate() {
                    unsafe {
                        std::env::set_var("AMD_INFER_GEMV_TUNE", variant.to_string());
                    }
                    // Warm both routes before retaining three paired observations per arm.
                    for repeat in 0..5 {
                        if cold {
                            scrub.run()?;
                        }
                        let before = hip::kernel_stats()
                            .iter()
                            .find(|v| v.0 == t.kind)
                            .map_or(0, |v| v.1);
                        let y = matrix.run_workspace(&x, &workspace)?;
                        let after = hip::kernel_stats()
                            .iter()
                            .find(|v| v.0 == t.kind)
                            .map_or(0, |v| v.1);
                        if y.iter()
                            .zip(&reference)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                        {
                            return Err(
                                format!("tuning bit mismatch {} mode={variant}", t.name).into()
                            );
                        }
                        if repeat > 0 {
                            println!("EVENT kind={} cols={cols} rows={rows} mode={mode} cold={cold} trial={trial} variant={variant} repeat={repeat} ns={} bytes={bytes}",t.kind,after-before);
                        }
                    }
                }
            }
        }
    }
    println!("PASS tuning all full-row outputs bit exact;FP32 order unchanged;hot/cold event windows diagnostic only");
    Ok(())
}
