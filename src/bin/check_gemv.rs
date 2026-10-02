//! Real format/shape representatives; untouched CPU GGML dequant + FP32 FMA tree.
use amd_infer::{gguf::Gguf, hip};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        return Err("check-gemv MODEL CPU_DEQUANT_ORACLE OUTPUT".into());
    }
    if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
        return Err("coordinate GPU".into());
    }
    fs::create_dir_all(&a[2])?;
    let g = Gguf::open(&a[0])?;
    let mut groups = BTreeMap::new();
    for t in &g.tensors {
        if t.dims.len() == 2
            && t.dims[0].is_multiple_of(256)
            && t.name != "token_embd.weight"
            && !t.name.starts_with("blk.64.")
        {
            groups
                .entry((t.kind, t.dims[0] as usize, t.dims[1] as usize))
                .or_insert_with(Vec::new)
                .push(t);
        }
    }
    let workspace = hip::LinearWorkspace::new(17408, 248320)?;
    let mut maximum = 0f32;
    for ((kind, cols, rows), tensors) in groups {
        let t = tensors[0];
        let packed_bytes = t.bytes()? as usize;
        let row_bytes = packed_bytes / rows;
        let storage = hip::PackedStorage::allocate(packed_bytes)?;
        let matrix = hip::PackedLinear::in_storage(&storage, 0, cols, rows, kind)?;
        let chunk_rows = (16 * 1024 * 1024 / row_bytes).max(1);
        for start in (0..rows).step_by(chunk_rows) {
            let count = chunk_rows.min(rows - start);
            let chunk = g.read_rows(t, start as u64, count as u64)?;
            matrix.upload_rows(start, &chunk)?;
        }
        let resource = hip::linear_resources(kind, rows)?;
        println!("GEMV group type={kind} cols={cols} rows={rows} count={} representative={} packed_bytes={packed_bytes} resources={resource:?} scope=runtime-estimated-not-measured-occupancy",tensors.len(),t.name);
        let selected: BTreeSet<_> = [0, 7.min(rows - 1), rows / 2, rows - 1]
            .into_iter()
            .collect();
        let mut references = Vec::new();
        for row in selected {
            let prefix =
                Path::new(&a[2]).join(format!("type{kind}.cols{cols}.rows{rows}.row{row}"));
            let packed = std::path::PathBuf::from(format!("{}.packed", prefix.display()));
            let output = std::path::PathBuf::from(format!("{}.f32", prefix.display()));
            fs::write(&packed, g.read_rows(t, row as u64, 1)?)?;
            let status = std::process::Command::new(&a[1])
                .args([
                    kind.to_string(),
                    cols.to_string(),
                    packed.to_string_lossy().into(),
                    output.to_string_lossy().into(),
                ])
                .status()?;
            if !status.success() {
                return Err("independent CPU dequantizer failed".into());
            }
            let bytes = fs::read(output)?;
            let w: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if w.len() != cols {
                return Err("oracle row shape".into());
            }
            references.push((row, w));
        }
        for trial in 0..5 {
            let mut x: Vec<_> = (0..cols)
                .map(|i| ((i * 31 + 7 + trial * 19) as f32 * 0.013).sin() * 0.15)
                .collect();
            if trial == 3 {
                x.fill(0.);
            }
            if trial == 4 {
                x.fill(0.);
                x[17] = -2.;
                x[2] = 3.;
            }
            let before = hip::kernel_stats()
                .iter()
                .find(|v| v.0 == kind)
                .map_or(0, |v| v.1);
            let y = matrix.run_workspace(&x, &workspace)?;
            let after = hip::kernel_stats()
                .iter()
                .find(|v| v.0 == kind)
                .map_or(0, |v| v.1);
            let mut error = 0f32;
            for (row, w) in &references {
                let mut partial = [0f32; 128];
                for tile in 0..cols / 256 {
                    for (lane, sum) in partial.iter_mut().enumerate() {
                        let j = tile * 256 + lane;
                        *sum = w[j].mul_add(x[j], *sum);
                        *sum = w[j + 128].mul_add(x[j + 128], *sum);
                    }
                }
                for n in [64, 32, 16, 8, 4, 2, 1] {
                    for lane in 0..n {
                        partial[lane] += partial[lane + n];
                    }
                }
                error = error.max((partial[0] - y[*row]).abs());
            }
            if y.iter().any(|v| !v.is_finite()) || error > 0.001 {
                return Err(format!("independent GEMV type={kind} cols={cols} rows={rows} trial={trial} max_abs={error}").into());
            }
            maximum = maximum.max(error);
            println!("GEMV event type={kind} cols={cols} rows={rows} trial={trial} seconds={} packed_bytes={packed_bytes} cpu_rows={} max_abs={error}",(after-before) as f64/1e9,references.len());
        }
    }
    println!("PASS GEMV format/shape representatives with untouched CPU GGML dequant + exact FP32 FMA tree; max_abs={maximum}");
    Ok(())
}
