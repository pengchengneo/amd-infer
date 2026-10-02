//! B=1 hybrid text execution, packed weight residency with budgeted streaming.
//! Optional HIP DeltaNet, FFN pipeline, norm bank and append-only attention cache.
//! Host convolution and graph scheduling remain; this is not a whole-model megakernel.
use crate::{
    gguf::{Gguf, Value},
    hip,
    ops::{self, ConvState, DeltaState},
};
use std::{collections::BTreeMap, error::Error};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const EPS: f32 = 1e-6;
struct LinearState {
    conv: ConvState,
    delta: DeltaState,
    gpu_delta: Option<hip::DeltaGpu>,
    gpu_pipeline: Option<hip::GdnPipelineGpu>,
}
struct AttentionState {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    gpu: Option<hip::AttentionGpu>,
}
pub struct Qwen<'a> {
    g: &'a Gguf,
    linear: Vec<Option<LinearState>>,
    attention: Vec<Option<AttentionState>>,
    small: BTreeMap<String, Vec<f32>>,
    position: usize,
    context_capacity: usize,
    sequence_gpu_attention_cache: Option<bool>,
    resident: BTreeMap<String, hip::PackedLinear>,
    use_gpu_delta: bool,
    workspace: Option<hip::LinearWorkspace>,
    ffn_norm_bank: Option<hip::NormBank>,
    device_hidden: Option<hip::DeviceHidden>,
    device_attention_norms: Option<hip::NormBank>,
    pub resident_bytes: u64,
    pub matrix_nanoseconds: std::cell::Cell<u64>,
    pub layer_host_nanoseconds: [std::cell::Cell<u64>; 4],
    pub device_layer_calls: std::cell::Cell<usize>,
    pub mixed_layer_calls: std::cell::Cell<usize>,
}
impl<'a> Qwen<'a> {
    fn diagnostic_position(&self) -> bool {
        std::env::var("AMD_INFER_TENSOR_TRACE_DIR").is_ok()
            && std::env::var("AMD_INFER_TENSOR_TRACE_POSITIONS")
                .unwrap_or("0,110,111".into())
                .split(',')
                .any(|v| v.parse::<usize>().ok() == Some(self.position))
    }
    fn diagnostic(&self, token: u32, layer: usize, tensor: &str, data: &[f32]) -> Result<()> {
        let Ok(root) = std::env::var("AMD_INFER_TENSOR_TRACE_DIR") else {
            return Ok(());
        };
        if !self.diagnostic_position() {
            return Ok(());
        }
        std::fs::create_dir_all(&root)?;
        let path = std::path::Path::new(&root).join(format!(
            "pos{}.token{}.layer{}.{}.f32",
            self.position, token, layer, tensor
        ));
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        use std::io::Write;
        for value in data {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }
    pub fn reset_state(&mut self) -> Result<()> {
        self.position = 0;
        self.sequence_gpu_attention_cache = None;
        self.matrix_nanoseconds.set(0);
        self.device_layer_calls.set(0);
        self.mixed_layer_calls.set(0);
        for value in &self.layer_host_nanoseconds {
            value.set(0);
        }
        for state in self.linear.iter_mut().flatten() {
            state.conv.reset();
            state.delta.reset();
            if let Some(delta) = &mut state.gpu_delta {
                delta.reset()?;
            }
            if let Some(pipeline) = &mut state.gpu_pipeline {
                pipeline.reset()?;
            }
        }
        for state in self.attention.iter_mut().flatten() {
            state.keys.clear();
            state.values.clear();
            if let Some(gpu) = &mut state.gpu {
                gpu.reset();
            }
        }
        Ok(())
    }
    pub fn new(g: &'a Gguf) -> Result<Self> {
        let context_capacity = std::env::var("AMD_INFER_CONTEXT_CAPACITY")
            .unwrap_or("128".into())
            .parse::<usize>()?;
        if ![128, 256, 512].contains(&context_capacity) {
            return Err("diagnostic context capacity must be 128, 256 or 512".into());
        }
        if g.metadata.get("general.architecture") != Some(&Value::Text("qwen35".into()))
            || g.metadata
                .get("qwen35.embedding_length")
                .and_then(Value::uint)
                != Some(5120)
            || g.metadata.get("qwen35.block_count").and_then(Value::uint) != Some(65)
        {
            return Err("requires confirmed Qwen3.8-27B layout".into());
        }
        let mut linear = Vec::new();
        let mut attention = Vec::new();
        for l in 0..64 {
            if l % 4 == 3 {
                linear.push(None);
                attention.push(Some(AttentionState {
                    keys: Vec::new(),
                    values: Vec::new(),
                    gpu: None,
                }));
            } else {
                linear.push(Some(LinearState {
                    conv: ConvState::new(10240)?,
                    delta: DeltaState::new(128, 48, 16)?,
                    gpu_delta: None,
                    gpu_pipeline: None,
                }));
                attention.push(None);
            }
        }
        Ok(Self {
            g,
            linear,
            attention,
            small: BTreeMap::new(),
            position: 0,
            context_capacity,
            sequence_gpu_attention_cache: None,
            resident: BTreeMap::new(),
            use_gpu_delta: std::env::var("AMD_INFER_GPU_DELTA").as_deref() == Ok("1"),
            workspace: None,
            ffn_norm_bank: None,
            device_hidden: None,
            device_attention_norms: None,
            resident_bytes: 0,
            matrix_nanoseconds: std::cell::Cell::new(0),
            layer_host_nanoseconds: std::array::from_fn(|_| std::cell::Cell::new(0)),
            device_layer_calls: std::cell::Cell::new(0),
            mixed_layer_calls: std::cell::Cell::new(0),
        })
    }
    /// Optional B=1 packed weight residency. Keeps a 2GiB measured-free reserve;
    /// no allocator guarantee or system-wide reservation is implied.
    pub fn load_resident(&mut self) -> Result<()> {
        if std::env::var("AMD_INFER_GPU_EXCLUSIVE").as_deref() != Ok("1") {
            return Err("resident loading requires coordinated exclusive GPU use".into());
        }
        let mut matrices: Vec<_> = self
            .g
            .tensors
            .iter()
            .filter(|t| {
                t.dims.len() == 2
                    && t.dims[0].is_multiple_of(256)
                    && t.name != "token_embd.weight"
                    && !t.name.starts_with("blk.64.")
            })
            .collect();
        // Correctness-only fallback for shared desktops: stream selected FFNs.
        // Opt-in and explicitly reported; never call it full residency.
        if std::env::var("AMD_INFER_PARTIAL_RESIDENCY").as_deref() == Ok("1") {
            let (free, _) = hip::memory()?;
            let allowance = free.saturating_sub(2 * 1024 * 1024 * 1024 + 256 * 1024 * 1024);
            for layer in (0..64).rev() {
                let sum = matrices
                    .iter()
                    .try_fold(0u64, |n, t| Ok::<_, Box<dyn Error>>(n + t.bytes()?))?;
                if sum as usize <= allowance {
                    break;
                }
                let p = format!("blk.{layer}.ffn_");
                matrices.retain(|t| !t.name.starts_with(&p));
            }
            eprintln!("partial residency enabled; omitted FFNs stream packed tiles to GPU; allowance_bytes={allowance}");
        }
        let needed = matrices
            .iter()
            .try_fold(0u64, |n, t| Ok::<_, Box<dyn Error>>(n + t.bytes()?))?;
        let reserve = 2usize * 1024 * 1024 * 1024;
        let (free, _) = hip::memory()?;
        if needed as usize + reserve > free {
            return Err(format!(
                "resident admission rejected: need {needed} + reserve {reserve}, free {free}"
            )
            .into());
        }
        eprintln!(
            "resident admission packed_bytes={needed} free_bytes={free} reserve_bytes={reserve}"
        );
        self.resident_bytes = needed;
        let storage = hip::PackedStorage::allocate(needed as usize)?;
        if std::env::var("AMD_INFER_REUSE_WORKSPACE").as_deref() == Ok("1") {
            self.workspace = Some(hip::LinearWorkspace::new(17408, 248320)?);
        }
        let (allocated_free, _) = hip::memory()?;
        if allocated_free < reserve {
            return Err(format!(
                "resident storage overhead exceeds reserve: free={allocated_free}"
            )
            .into());
        }
        let mut offset = 0usize;
        for t in matrices {
            if self.resident.contains_key(&t.name) {
                return Err("weights already resident".into());
            }
            let (free, _) = hip::memory()?;
            if reserve > free {
                return Err("free VRAM changed during resident load".into());
            }
            let cols = t.dims[0] as usize;
            let rows = t.dims[1] as usize;
            let matrix = hip::PackedLinear::in_storage(&storage, offset, cols, rows, t.kind)?;
            for start in (0..rows).step_by(4096) {
                let raw = self
                    .g
                    .read_rows(t, start as u64, (rows - start).min(4096) as u64)?;
                matrix.upload_rows(start, &raw)?;
            }
            self.resident.insert(t.name.clone(), matrix);
            offset += t.bytes()? as usize;
            eprintln!("resident loaded {}", t.name);
        }
        if std::env::var("AMD_INFER_RESIDENT_FFN_NORM").as_deref() == Ok("1") {
            let mut norms = Vec::with_capacity(64 * 5120);
            for layer in 0..64 {
                norms.extend(self.vector(&format!("blk.{layer}.post_attention_norm.weight"))?);
            }
            self.ffn_norm_bank = Some(hip::NormBank::new(&norms)?);
            eprintln!("resident FFN norm bank requested_bytes={}", norms.len() * 4);
        }
        Ok(())
    }
    fn vector(&mut self, name: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.small.get(name) {
            return Ok(v.clone());
        }
        let t = self.g.tensor(name)?;
        if t.kind != 0 {
            return Err(format!("{name}: expected F32 small tensor").into());
        }
        let raw = self.g.read_rows(t, 0, t.elements()? / t.dims[0])?;
        let v: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        self.small.insert(name.into(), v.clone());
        Ok(v)
    }
    fn matrix(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();
        let result = self.matrix_inner(name, x);
        self.matrix_nanoseconds
            .set(self.matrix_nanoseconds.get() + start.elapsed().as_nanos() as u64);
        result
    }
    fn matrix_inner(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        if let Some(matrix) = self.resident.get(name) {
            if let Some(workspace) = &self.workspace {
                return Ok(matrix.run_workspace(x, workspace)?);
            }
            return Ok(matrix.run(x)?);
        }
        let t = self.g.tensor(name)?;
        if t.dims.len() != 2 || t.dims[0] as usize != x.len() {
            return Err(format!("{name}: linear shape mismatch").into());
        }
        let rows = t.dims[1] as usize;
        let mut y = Vec::with_capacity(rows);
        // <= ~2.4MiB per tile for the widest IQ4 matrix; bounded allocation.
        for start in (0..rows).step_by(4096) {
            let count = (rows - start).min(4096);
            let raw = self.g.read_rows(t, start as u64, count as u64)?;
            if let Some(workspace) = &self.workspace {
                let storage = hip::PackedStorage::allocate(raw.len())?;
                let matrix = hip::PackedLinear::in_storage(&storage, 0, x.len(), count, t.kind)?;
                matrix.upload_rows(0, &raw)?;
                y.extend(matrix.run_workspace(x, workspace)?);
            } else {
                y.extend(hip::linear_quant(&raw, x, x.len(), count, 1, t.kind)?);
            }
        }
        Ok(y)
    }
    fn norm(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let w = self.vector(name)?;
        if std::env::var("AMD_INFER_GPU_OPS").as_deref() == Ok("1") {
            return Ok(hip::rms(x, &w)?);
        }
        Ok(ops::rms_norm(x, &w, EPS)?)
    }
    fn feed_forward(&mut self, layer: usize, p: &str, hidden: &[f32]) -> Result<Vec<f32>> {
        let weight = self.vector(&format!("{p}.post_attention_norm.weight"))?;
        if std::env::var("AMD_INFER_GPU_FFN").as_deref() == Ok("1") {
            if let (Some(gate), Some(up), Some(down), Some(workspace)) = (
                self.resident.get(&format!("{p}.ffn_gate.weight")),
                self.resident.get(&format!("{p}.ffn_up.weight")),
                self.resident.get(&format!("{p}.ffn_down.weight")),
                self.workspace.as_ref(),
            ) {
                let start = std::time::Instant::now();
                let out = hip::ffn(
                    gate,
                    up,
                    down,
                    hidden,
                    &weight,
                    workspace,
                    self.ffn_norm_bank.as_ref().map(|bank| (bank, layer)),
                )?;
                self.matrix_nanoseconds
                    .set(self.matrix_nanoseconds.get() + start.elapsed().as_nanos() as u64);
                return Ok(out);
            }
        }
        let norm = if std::env::var("AMD_INFER_GPU_OPS").as_deref() == Ok("1") {
            hip::rms(hidden, &weight)?
        } else {
            ops::rms_norm(hidden, &weight, EPS)?
        };
        let gate = self.matrix(&format!("{p}.ffn_gate.weight"), &norm)?;
        let up = self.matrix(&format!("{p}.ffn_up.weight"), &norm)?;
        let act = ops::swiglu(&gate, &up)?;
        self.matrix(&format!("{p}.ffn_down.weight"), &act)
    }
    fn delta_layer(&mut self, l: usize, x: &[f32], token: u32) -> Result<Vec<f32>> {
        let p = format!("blk.{l}");
        let names = ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"]
            .map(|n| format!("{p}.{n}.weight"));
        if std::env::var("AMD_INFER_GPU_GDN_PROJECTED").as_deref() == Ok("1")
            && self.workspace.is_some()
            && names.iter().all(|n| self.resident.contains_key(n))
        {
            if self.linear[l]
                .as_ref()
                .ok_or("missing DeltaNet state")?
                .gpu_pipeline
                .is_none()
            {
                let a = self.vector(&format!("{p}.ssm_a"))?;
                let bias = self.vector(&format!("{p}.ssm_dt.bias"))?;
                let conv = self.vector(&format!("{p}.ssm_conv1d.weight"))?;
                let norm = self.vector(&format!("{p}.ssm_norm.weight"))?;
                self.linear[l].as_mut().unwrap().gpu_pipeline =
                    Some(hip::GdnPipelineGpu::new(&conv, &norm, &a, &bias)?);
            }
            let start = std::time::Instant::now();
            let matrices = names.each_ref().map(|n| self.resident.get(n).unwrap());
            let result = self.linear[l]
                .as_mut()
                .unwrap()
                .gpu_pipeline
                .as_mut()
                .unwrap()
                .step_projected(matrices, x, self.workspace.as_ref().unwrap())?;
            self.matrix_nanoseconds
                .set(self.matrix_nanoseconds.get() + start.elapsed().as_nanos() as u64);
            if l == 0 && self.diagnostic_position() {
                let parts = self.linear[l]
                    .as_ref()
                    .unwrap()
                    .gpu_pipeline
                    .as_ref()
                    .unwrap()
                    .snapshot_stages()?;
                for (name, part) in ["gdn_raw", "gdn_mixed", "gdn_core", "gdn_gated"]
                    .iter()
                    .zip(parts)
                {
                    self.diagnostic(token, l, name, &part)?;
                }
            }
            return Ok(result);
        }
        let qkv = self.matrix(&format!("{p}.attn_qkv.weight"), x)?;
        let z = self.matrix(&format!("{p}.attn_gate.weight"), x)?;
        let beta_raw = self.matrix(&format!("{p}.ssm_beta.weight"), x)?;
        let alpha = self.matrix(&format!("{p}.ssm_alpha.weight"), x)?;
        if l == 0 && self.diagnostic_position() {
            let mut raw = qkv.clone();
            raw.extend_from_slice(&z);
            raw.extend_from_slice(&alpha);
            raw.extend_from_slice(&beta_raw);
            self.diagnostic(token, l, "gdn_raw", &raw)?;
        }
        if std::env::var("AMD_INFER_GPU_GDN_PIPELINE").as_deref() == Ok("1") {
            if self.linear[l]
                .as_ref()
                .ok_or("missing DeltaNet state")?
                .gpu_pipeline
                .is_none()
            {
                let a = self.vector(&format!("{p}.ssm_a"))?;
                let bias = self.vector(&format!("{p}.ssm_dt.bias"))?;
                let conv_w = self.vector(&format!("{p}.ssm_conv1d.weight"))?;
                let norm_w = self.vector(&format!("{p}.ssm_norm.weight"))?;
                self.linear[l].as_mut().unwrap().gpu_pipeline =
                    Some(hip::GdnPipelineGpu::new(&conv_w, &norm_w, &a, &bias)?);
            }
            let state = self.linear[l].as_mut().unwrap();
            let gated = state
                .gpu_pipeline
                .as_mut()
                .unwrap()
                .step(&qkv, &z, &alpha, &beta_raw)?;
            return self.matrix(&format!("{p}.ssm_out.weight"), &gated);
        }
        let a = self.vector(&format!("{p}.ssm_a"))?;
        let bias = self.vector(&format!("{p}.ssm_dt.bias"))?;
        let conv_w = self.vector(&format!("{p}.ssm_conv1d.weight"))?;
        let norm_w = self.vector(&format!("{p}.ssm_norm.weight"))?;
        let beta = beta_raw.into_iter().map(ops::sigmoid).collect::<Vec<_>>();
        let g: Vec<f32> = alpha
            .iter()
            .zip(bias)
            .zip(a)
            .map(|((a, b), w)| ops::softplus(a + b) * w)
            .collect();
        let state = self.linear[l].as_mut().ok_or("missing DeltaNet state")?;
        let mixed = state.conv.step(&qkv, &conv_w)?;
        let mut q = Vec::new();
        let mut k = Vec::new();
        for h in 0..16 {
            q.extend(ops::l2_norm(&mixed[h * 128..(h + 1) * 128], EPS)?);
            k.extend(ops::l2_norm(
                &mixed[2048 + h * 128..2048 + (h + 1) * 128],
                EPS,
            )?);
        }
        let core = if self.use_gpu_delta {
            if state.gpu_delta.is_none() {
                let (free, _) = hip::memory()?;
                if free < 2 * 1024 * 1024 * 1024 + 8 * 1024 * 1024 {
                    return Err("GPU DeltaNet state would consume the VRAM reserve".into());
                }
                state.gpu_delta = Some(hip::DeltaGpu::new()?);
            }
            state.gpu_delta.as_mut().ok_or("GPU state missing")?.step(
                &q,
                &k,
                &mixed[4096..],
                &g,
                &beta,
            )?
        } else {
            state.delta.step(&q, &k, &mixed[4096..], &g, &beta)?
        };
        let mut gated = Vec::with_capacity(6144);
        for h in 0..48 {
            let n = ops::rms_norm(&core[h * 128..(h + 1) * 128], &norm_w, EPS)?;
            for j in 0..128 {
                gated.push(n[j] * z[h * 128 + j] * ops::sigmoid(z[h * 128 + j]));
            }
        }
        if l == 0 && self.diagnostic_position() {
            let mut prepared = q;
            prepared.extend_from_slice(&k);
            prepared.extend_from_slice(&mixed[4096..]);
            prepared.extend_from_slice(&g);
            prepared.extend_from_slice(&beta);
            self.diagnostic(token, l, "gdn_mixed", &prepared)?;
            self.diagnostic(token, l, "gdn_core", &core)?;
            self.diagnostic(token, l, "gdn_gated", &gated)?;
        }
        self.matrix(&format!("{p}.ssm_out.weight"), &gated)
    }
    fn attention_layer(&mut self, l: usize, x: &[f32]) -> Result<Vec<f32>> {
        let p = format!("blk.{l}");
        let qg = self.matrix(&format!("{p}.attn_q.weight"), x)?;
        let mut k = self.matrix(&format!("{p}.attn_k.weight"), x)?;
        let v = self.matrix(&format!("{p}.attn_v.weight"), x)?;
        let qw = self.vector(&format!("{p}.attn_q_norm.weight"))?;
        let kw = self.vector(&format!("{p}.attn_k_norm.weight"))?;
        let mut q = Vec::with_capacity(6144);
        let mut gate = Vec::with_capacity(6144);
        for h in 0..24 {
            let mut n = ops::rms_norm(&qg[h * 512..h * 512 + 256], &qw, EPS)?;
            rope(&mut n, self.position);
            q.extend(n);
            gate.extend_from_slice(&qg[h * 512 + 256..(h + 1) * 512]);
        }
        for h in 0..4 {
            let mut n = ops::rms_norm(&k[h * 256..(h + 1) * 256], &kw, EPS)?;
            rope(&mut n, self.position);
            k[h * 256..(h + 1) * 256].copy_from_slice(&n);
        }
        let state = self.attention[l]
            .as_mut()
            .ok_or("missing attention state")?;
        if std::env::var("AMD_INFER_GPU_ATTN_CACHE").as_deref() == Ok("1") {
            if state.gpu.is_none() {
                state.gpu = Some(hip::AttentionGpu::with_capacity(self.context_capacity)?);
            }
            let output = state.gpu.as_mut().unwrap().step(&q, &k, &v, &gate)?;
            return self.matrix(&format!("{p}.attn_output.weight"), &output);
        }
        state.keys.push(k);
        state.values.push(v);
        if std::env::var("AMD_INFER_GPU_OPS").as_deref() == Ok("1") {
            let output = hip::attention(&q, &state.keys, &state.values, &gate)?;
            return self.matrix(&format!("{p}.attn_output.weight"), &output);
        }
        let mut output = vec![0f32; 6144];
        for h in 0..24 {
            let kh = h / 6;
            let qq = &q[h * 256..(h + 1) * 256];
            let mut scores = state
                .keys
                .iter()
                .map(|k| {
                    qq.iter()
                        .zip(&k[kh * 256..(kh + 1) * 256])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / 16.
                })
                .collect::<Vec<_>>();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for s in &mut scores {
                *s = ops::activation_exp(*s - max)
            }
            let sum = scores.iter().sum::<f32>();
            for (t, s) in scores.iter().enumerate() {
                for j in 0..256 {
                    output[h * 256 + j] += s / sum * state.values[t][kh * 256 + j]
                }
            }
            for j in 0..256 {
                output[h * 256 + j] *= ops::sigmoid(gate[h * 256 + j]);
            }
        }
        self.matrix(&format!("{p}.attn_output.weight"), &output)
    }
    fn device_layer_ready(&self, l: usize) -> bool {
        let names: &[&str] = if l % 4 == 3 {
            &[
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        } else {
            &[
                "attn_qkv",
                "attn_gate",
                "ssm_alpha",
                "ssm_beta",
                "ssm_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        };
        self.workspace.is_some()
            && names
                .iter()
                .all(|n| self.resident.contains_key(&format!("blk.{l}.{n}.weight")))
    }
    fn prepare_device_layer(&mut self, l: usize) -> Result<()> {
        if self.device_hidden.is_none() {
            let mut attention_norms = Vec::with_capacity(64 * 5120);
            let mut ffn_norms = Vec::with_capacity(64 * 5120);
            for layer in 0..64 {
                attention_norms.extend(self.vector(&format!("blk.{layer}.attn_norm.weight"))?);
                ffn_norms.extend(self.vector(&format!("blk.{layer}.post_attention_norm.weight"))?);
            }
            self.device_attention_norms = Some(hip::NormBank::new(&attention_norms)?);
            if self.ffn_norm_bank.is_none() {
                self.ffn_norm_bank = Some(hip::NormBank::new(&ffn_norms)?);
            }
            self.device_hidden = Some(hip::DeviceHidden::new()?);
        }
        if l % 4 == 3 {
            let qw = self.vector(&format!("blk.{l}.attn_q_norm.weight"))?;
            let kw = self.vector(&format!("blk.{l}.attn_k_norm.weight"))?;
            let cache = &mut self.attention[l].as_mut().unwrap().gpu;
            if cache.is_none() {
                *cache = Some(hip::AttentionGpu::with_capacity(self.context_capacity)?);
            }
            cache.as_mut().unwrap().prepare_weights(&qw, &kw)?;
            return Ok(());
        }
        if self.linear[l].as_ref().unwrap().gpu_pipeline.is_none() {
            let p = format!("blk.{l}");
            let a = self.vector(&format!("{p}.ssm_a"))?;
            let bias = self.vector(&format!("{p}.ssm_dt.bias"))?;
            let conv = self.vector(&format!("{p}.ssm_conv1d.weight"))?;
            let norm = self.vector(&format!("{p}.ssm_norm.weight"))?;
            self.linear[l].as_mut().unwrap().gpu_pipeline =
                Some(hip::GdnPipelineGpu::new(&conv, &norm, &a, &bias)?);
        }
        Ok(())
    }
    pub fn forward(&mut self, token: u32, layers: usize) -> Result<Vec<f32>> {
        if layers == 0 || layers > 64 || self.position >= self.context_capacity {
            return Err("correctness path exceeds configured context or 1..64 layers".into());
        }
        let cache = std::env::var("AMD_INFER_GPU_ATTN_CACHE").as_deref() == Ok("1");
        if self
            .sequence_gpu_attention_cache
            .is_some_and(|previous| previous != cache)
        {
            return Err("attention cache backend cannot change without resetting state".into());
        }
        self.sequence_gpu_attention_cache = Some(cache);
        let embedding = self.g.tensor("token_embd.weight")?;
        if token as u64 >= embedding.dims[1] {
            return Err("token outside vocabulary".into());
        }
        let raw = self.g.read_rows(embedding, token as u64, 1)?;
        let mut hidden = hip::dequant(&raw, 5120, embedding.kind)?;
        let device = std::env::var("AMD_INFER_DEVICE_SEGMENTS").as_deref() == Ok("1");
        if device
            && (!cache
                || std::env::var("AMD_INFER_STABLE_RMS").as_deref() != Ok("1")
                || std::env::var("AMD_INFER_GPU_GDN_PROJECTED").as_deref() != Ok("1")
                || std::env::var("AMD_INFER_GPU_FFN").as_deref() != Ok("1")
                || (std::env::var("AMD_INFER_TENSOR_TRACE_DIR").is_ok()
                    && std::env::var("AMD_INFER_DEVICE_TRACE").as_deref() != Ok("1"))
                || std::env::var("AMD_INFER_TRACE").as_deref() == Ok("1"))
        {
            return Err(
                "device segments require stable RMS/projected GDN/GPU FFN; tensor tracing needs AMD_INFER_DEVICE_TRACE=1"
                    .into(),
            );
        }
        let mut on_device = false;
        let skip_devices = std::env::var("AMD_INFER_DEVICE_SKIP_LAYERS")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(str::parse::<usize>)
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        if skip_devices.iter().any(|l| *l >= 64) {
            return Err("device skip layer outside 0..64".into());
        }
        let mut rotation_uploaded = false;
        for l in 0..layers {
            let p = format!("blk.{l}");
            if device && !skip_devices.contains(&l) && self.device_layer_ready(l) {
                self.prepare_device_layer(l)?;
                let vector = self.device_hidden.as_ref().unwrap();
                if !on_device {
                    vector.upload(&hidden)?;
                }
                if self.diagnostic_position()
                    && std::env::var("AMD_INFER_DEVICE_TRACE").as_deref() == Ok("1")
                {
                    let input = if on_device {
                        vector.download()?
                    } else {
                        hidden.clone()
                    };
                    self.diagnostic(token, l, "input_hidden", &input)?;
                }
                if !rotation_uploaded {
                    let mut rotation = [0f32; 64];
                    for i in 0..32 {
                        let theta = self.position as f32 * 10_000_000f32.powf(-2. * i as f32 / 64.);
                        rotation[i] = theta.cos();
                        rotation[i + 32] = theta.sin();
                    }
                    vector.rotation(&rotation)?;
                    rotation_uploaded = true;
                }
                let ffn = ["ffn_gate", "ffn_up", "ffn_down"]
                    .map(|n| self.resident.get(&format!("{p}.{n}.weight")).unwrap());
                let norms = (
                    self.device_attention_norms.as_ref().unwrap(),
                    self.ffn_norm_bank.as_ref().unwrap(),
                    l,
                );
                if l % 4 == 3 {
                    let matrices = ["attn_q", "attn_k", "attn_v", "attn_output"]
                        .map(|n| self.resident.get(&format!("{p}.{n}.weight")).unwrap());
                    vector.attention_layer(
                        self.attention[l].as_mut().unwrap().gpu.as_mut().unwrap(),
                        matrices,
                        ffn,
                        norms,
                        self.workspace.as_ref().unwrap(),
                    )?;
                } else {
                    let gdn = ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"]
                        .map(|n| self.resident.get(&format!("{p}.{n}.weight")).unwrap());
                    vector.gdn_layer(
                        self.linear[l]
                            .as_mut()
                            .unwrap()
                            .gpu_pipeline
                            .as_mut()
                            .unwrap(),
                        gdn,
                        ffn,
                        norms,
                        self.workspace.as_ref().unwrap(),
                    )?;
                }
                if self.diagnostic_position()
                    && std::env::var("AMD_INFER_DEVICE_TRACE").as_deref() == Ok("1")
                {
                    self.diagnostic(token, l, "hidden", &vector.download()?)?;
                    let parts = self.workspace.as_ref().unwrap().snapshot_ffn()?;
                    for (name, part) in ["ffn_norm", "ffn_act", "ffn_out"].iter().zip(parts) {
                        self.diagnostic(token, l, name, &part)?;
                    }
                    if l % 4 == 3 {
                        let cache = self.attention[l].as_ref().unwrap().gpu.as_ref().unwrap();
                        for (name, part) in [
                            "attention_raw",
                            "attention_query_gate_core",
                            "attention_current_kv",
                        ]
                        .iter()
                        .zip(cache.snapshot_prepared()?)
                        {
                            self.diagnostic(token, l, name, &part)?;
                        }
                        if l == 3 {
                            self.diagnostic(token, l, "attention_cache", &cache.snapshot_cache()?)?;
                        }
                    } else {
                        let pipeline = self.linear[l]
                            .as_ref()
                            .unwrap()
                            .gpu_pipeline
                            .as_ref()
                            .unwrap();
                        for (name, part) in ["gdn_raw", "gdn_mixed", "gdn_core", "gdn_gated"]
                            .iter()
                            .zip(pipeline.snapshot_stages()?)
                        {
                            self.diagnostic(token, l, name, &part)?;
                        }
                        if [0, 16, 32, 48].contains(&l) {
                            self.diagnostic(
                                token,
                                l,
                                "recurrent_and_conv_state",
                                &pipeline.snapshot_state()?,
                            )?;
                        }
                    }
                }
                on_device = true;
                self.device_layer_calls
                    .set(self.device_layer_calls.get() + 1);
                continue;
            }
            if on_device {
                hidden = self.device_hidden.as_ref().unwrap().download()?;
                if !hidden.iter().all(|v| v.is_finite()) {
                    return Err(format!("nonfinite device segment before layer {l}").into());
                }
                on_device = false;
            }
            let clock = hip::host_profile().then(std::time::Instant::now);
            let norm = self.norm(&format!("{p}.attn_norm.weight"), &hidden)?;
            self.mixed_layer_calls.set(self.mixed_layer_calls.get() + 1);
            if let Some(clock) = clock {
                let value = &self.layer_host_nanoseconds[3];
                value.set(value.get() + clock.elapsed().as_nanos() as u64);
            }
            self.diagnostic(token, l, "attn_norm", &norm)?;
            let clock = hip::host_profile().then(std::time::Instant::now);
            let out = if l % 4 == 3 {
                self.attention_layer(l, &norm)?
            } else {
                self.delta_layer(l, &norm, token)?
            };
            if let Some(clock) = clock {
                let value = &self.layer_host_nanoseconds[usize::from(l % 4 != 3)];
                value.set(value.get() + clock.elapsed().as_nanos() as u64);
            }
            self.diagnostic(token, l, "attn_out", &out)?;
            if self.diagnostic_position() && l == 0 {
                let state = self.linear[l].as_ref().unwrap();
                let mut values = if let Some(gpu) = &state.gpu_pipeline {
                    gpu.snapshot_state()?
                } else {
                    let mut v = if let Some(gpu) = &state.gpu_delta {
                        gpu.snapshot_state()?
                    } else {
                        state.delta.data.clone()
                    };
                    v.extend_from_slice(state.conv.snapshot_state());
                    v
                };
                self.diagnostic(token, l, "recurrent_and_conv_state", &values)?;
                values.clear();
            }
            for (h, a) in hidden.iter_mut().zip(out) {
                *h += a
            }
            self.diagnostic(token, l, "attn_residual", &hidden)?;
            let clock = hip::host_profile().then(std::time::Instant::now);
            let out = self.feed_forward(l, &p, &hidden)?;
            if let Some(clock) = clock {
                let value = &self.layer_host_nanoseconds[2];
                value.set(value.get() + clock.elapsed().as_nanos() as u64);
            }
            self.diagnostic(token, l, "ffn_out", &out)?;
            for (h, a) in hidden.iter_mut().zip(out) {
                *h += a
            }
            if !hidden.iter().all(|v| v.is_finite()) {
                return Err(format!("nonfinite hidden state at layer {l}").into());
            }
            self.diagnostic(token, l, "hidden", &hidden)?;
            if std::env::var("AMD_INFER_TRACE").as_deref() == Ok("1") {
                eprintln!(
                    "token={} layer={} hidden_rms={:.6}",
                    self.position,
                    l,
                    (hidden.iter().map(|v| v * v).sum::<f32>() / 5120.).sqrt()
                );
            }
        }
        if on_device {
            hidden = self.device_hidden.as_ref().unwrap().download()?;
        }
        if !hidden.iter().all(|v| v.is_finite()) {
            return Err("nonfinite final device hidden".into());
        }
        self.position += 1;
        Ok(hidden)
    }
    pub fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>> {
        let h = self.norm("output_norm.weight", hidden)?;
        self.matrix("output.weight", &h)
    }
    /// Diagnostic bit-pattern digest; this is mixed-path state evidence,
    /// not an independent upstream-state oracle.
    pub fn recurrent_state_digests(&self, layers: usize) -> Result<Vec<(usize, u64)>> {
        let mut hashes = Vec::new();
        for (layer, state) in self.linear.iter().take(layers).enumerate() {
            if let Some(pipeline) = state.as_ref().and_then(|s| s.gpu_pipeline.as_ref()) {
                let mut hash = 0xcbf29ce484222325u64;
                for value in pipeline.snapshot_state()? {
                    for byte in value.to_bits().to_le_bytes() {
                        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                    }
                }
                hashes.push((layer, hash));
            }
        }
        Ok(hashes)
    }
}
fn rope(x: &mut [f32], pos: usize) {
    for i in 0..32 {
        let theta = pos as f32 * 10_000_000f32.powf(-2. * i as f32 / 64.);
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = (x[i], x[i + 32]);
        x[i] = a * c - b * s;
        x[i + 32] = a * s + b * c;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_rope_preserves_tail_and_norm() {
        let mut x = vec![1.; 256];
        rope(&mut x, 7);
        assert!(x[64..].iter().all(|v| *v == 1.));
        assert!((x[..64].iter().map(|v| v * v).sum::<f32>() - 64.).abs() < 1e-4);
    }
}
