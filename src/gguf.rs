//! Read-only GGUF v3 directory reader. Tensor payloads stay on disk until requested.
use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Array(Vec<Value>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn fixture(kind: u32, width: u64, overlap: bool, truncated: bool) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "amd-infer-gguf-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((if overlap { 2u64 } else { 1u64 }).to_le_bytes());
        b.extend(0u64.to_le_bytes());
        for name in if overlap { vec!["x", "y"] } else { vec!["x"] } {
            b.extend(1u64.to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(1u32.to_le_bytes());
            b.extend(width.to_le_bytes());
            b.extend(kind.to_le_bytes());
            b.extend(0u64.to_le_bytes());
        }
        b.resize((b.len() + 31) & !31, 0);
        if !truncated {
            b.resize(b.len() + 4096, 0)
        }
        std::fs::write(&path, b).unwrap();
        Fixture(path)
    }
    #[test]
    fn directory_and_row_bounds() {
        let f = fixture(23, 256, false, false);
        let g = Gguf::open(&f.0).unwrap();
        let t = g.tensor("x").unwrap();
        assert_eq!(t.bytes().unwrap(), 136);
        assert_eq!(g.read_rows(t, 0, 1).unwrap().len(), 136);
        assert!(g.read_rows(t, 1, 1).is_err());
        assert!(g.read_rows(t, u64::MAX, 1).is_err());
    }
    #[test]
    fn rejects_payload_truncation() {
        let f = fixture(23, 256, false, true);
        assert!(Gguf::open(&f.0).is_err());
    }
    #[test]
    fn rejects_overlap_and_bad_quant_alignment() {
        let f = fixture(23, 256, true, false);
        assert!(Gguf::open(&f.0).is_err());
        let f = fixture(23, 255, false, false);
        assert!(Gguf::open(&f.0).is_err());
    }
    #[test]
    fn rejects_unknown_quant_type() {
        let f = fixture(999, 256, false, false);
        assert!(Gguf::open(&f.0).is_err());
    }
}
impl Value {
    pub fn uint(&self) -> Option<u64> {
        if let Self::UInt(n) = self {
            Some(*n)
        } else {
            None
        }
    }
}
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
}
impl Tensor {
    pub fn elements(&self) -> io::Result<u64> {
        self.dims.iter().try_fold(1u64, |a, b| {
            a.checked_mul(*b)
                .ok_or_else(|| invalid("tensor size overflow"))
        })
    }
    pub fn bytes(&self) -> io::Result<u64> {
        let (block, bytes) =
            type_layout(self.kind).ok_or_else(|| invalid("unsupported GGML type"))?;
        let n = self.elements()?;
        if self.dims.is_empty() || !self.dims[0].is_multiple_of(block) {
            return Err(invalid("invalid quantized row alignment"));
        }
        (n / block)
            .checked_mul(bytes)
            .ok_or_else(|| invalid("tensor bytes overflow"))
    }
}
pub struct Gguf {
    pub path: PathBuf,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<Tensor>,
    pub data_offset: u64,
    pub file_bytes: u64,
}
fn invalid(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
fn bytes<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut x = [0; N];
    r.read_exact(&mut x)?;
    Ok(x)
}
fn u32r(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(bytes(r)?))
}
fn u64r(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(bytes(r)?))
}
fn text(r: &mut impl Read) -> io::Result<String> {
    let n = u64r(r)?;
    if n > 64 * 1024 * 1024 {
        return Err(invalid("GGUF string too large"));
    }
    let mut b = vec![0; n as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|_| invalid("invalid UTF-8"))
}
fn value(r: &mut impl Read, t: u32, depth: u8) -> io::Result<Value> {
    if depth > 8 {
        return Err(invalid("nested GGUF arrays too deep"));
    }
    Ok(match t {
        0 => Value::UInt(u8::from_le_bytes(bytes(r)?) as u64),
        1 => Value::Int(i8::from_le_bytes(bytes(r)?) as i64),
        2 => Value::UInt(u16::from_le_bytes(bytes(r)?) as u64),
        3 => Value::Int(i16::from_le_bytes(bytes(r)?) as i64),
        4 => Value::UInt(u32r(r)? as u64),
        5 => Value::Int(i32::from_le_bytes(bytes(r)?) as i64),
        6 => Value::Float(f32::from_le_bytes(bytes(r)?) as f64),
        7 => {
            let b = bytes::<1>(r)?[0];
            if b > 1 {
                return Err(invalid("invalid boolean"));
            }
            Value::Bool(b == 1)
        }
        8 => Value::Text(text(r)?),
        9 => {
            let element = u32r(r)?;
            let n = u64r(r)?;
            if n > 2_000_000 {
                return Err(invalid("GGUF array too large"));
            }
            let mut a = Vec::with_capacity(n as usize);
            for _ in 0..n {
                a.push(value(r, element, depth + 1)?)
            }
            Value::Array(a)
        }
        10 => Value::UInt(u64r(r)?),
        11 => Value::Int(i64::from_le_bytes(bytes(r)?)),
        12 => Value::Float(f64::from_le_bytes(bytes(r)?)),
        _ => return Err(invalid("unknown GGUF metadata type")),
    })
}
pub fn type_layout(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 | 28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        _ => return None,
    })
}
pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        30 => "BF16",
        _ => "other",
    }
}
impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_bytes = file.metadata()?.len();
        let mut f = BufReader::with_capacity(256 * 1024, file);
        if &bytes::<4>(&mut f)? != b"GGUF" || u32r(&mut f)? != 3 {
            return Err(invalid("requires GGUF v3"));
        }
        let nt = u64r(&mut f)?;
        let nm = u64r(&mut f)?;
        if nt > 100_000 || nm > 100_000 {
            return Err(invalid("directory too large"));
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..nm {
            let k = text(&mut f)?;
            let t = u32r(&mut f)?;
            let v = value(&mut f, t, 0)?;
            if metadata.insert(k, v).is_some() {
                return Err(invalid("duplicate metadata"));
            }
        }
        let mut tensors = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for _ in 0..nt {
            let name = text(&mut f)?;
            if !names.insert(name.clone()) {
                return Err(invalid("duplicate tensor"));
            }
            let nd = u32r(&mut f)?;
            if !(1..=4).contains(&nd) {
                return Err(invalid("invalid tensor rank"));
            }
            let mut dims = Vec::new();
            for _ in 0..nd {
                let n = u64r(&mut f)?;
                if n == 0 {
                    return Err(invalid("zero tensor extent"));
                }
                dims.push(n)
            }
            let kind = u32r(&mut f)?;
            let offset = u64r(&mut f)?;
            tensors.push(Tensor {
                name,
                dims,
                kind,
                offset,
            });
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::uint)
            .unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 4096 {
            return Err(invalid("invalid data alignment"));
        }
        let pos = f.stream_position()?;
        let data_offset = pos
            .checked_add(alignment - 1)
            .ok_or_else(|| invalid("offset overflow"))?
            & !(alignment - 1);
        let mut ranges = Vec::new();
        for t in &tensors {
            let end = t
                .offset
                .checked_add(t.bytes()?)
                .and_then(|n| data_offset.checked_add(n))
                .ok_or_else(|| invalid("offset overflow"))?;
            if t.offset % alignment != 0 || end > file_bytes {
                return Err(invalid("tensor outside file or misaligned"));
            }
            ranges.push((t.offset, end - data_offset));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|x| x[0].1 > x[1].0) {
            return Err(invalid("overlapping tensors"));
        }
        Ok(Self {
            path,
            metadata,
            tensors,
            data_offset,
            file_bytes,
        })
    }
    pub fn tensor(&self, name: &str) -> io::Result<&Tensor> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_string()))
    }
    pub fn read_rows(&self, t: &Tensor, start: u64, count: u64) -> io::Result<Vec<u8>> {
        let rows = t.elements()? / t.dims[0];
        if start.checked_add(count).is_none_or(|n| n > rows) {
            return Err(invalid("row range outside tensor"));
        }
        let row_bytes = t.bytes()? / rows;
        let len = row_bytes
            .checked_mul(count)
            .ok_or_else(|| invalid("allocation overflow"))?;
        if len > 512 * 1024 * 1024 {
            return Err(invalid("read exceeds 512MiB safety limit"));
        }
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(
            self.data_offset + t.offset + start * row_bytes,
        ))?;
        let mut b = vec![0; len as usize];
        f.read_exact(&mut b)?;
        Ok(b)
    }
}
