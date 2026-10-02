//! Narrow C ABI; Rust owns lifetimes and bounds, HIP owns device compilation.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    ffi::{c_char, c_void, CStr},
    ptr::NonNull,
};
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static MIN_FREE: AtomicUsize = AtomicUsize::new(usize::MAX);
static HOST_COUNTERS: [AtomicUsize; 12] = [const { AtomicUsize::new(0) }; 12];
static PROFILE_HOST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
pub fn host_profile() -> bool {
    *PROFILE_HOST.get_or_init(|| std::env::var("AMD_INFER_PROFILE_HOST").as_deref() == Ok("1"))
}
fn host_record(kind: usize, bytes: usize, started: std::time::Instant) {
    HOST_COUNTERS[kind * 3].fetch_add(1, Ordering::Relaxed);
    HOST_COUNTERS[kind * 3 + 1].fetch_add(bytes, Ordering::Relaxed);
    HOST_COUNTERS[kind * 3 + 2].fetch_add(started.elapsed().as_nanos() as usize, Ordering::Relaxed);
}
unsafe fn host_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32 {
    if !host_profile() {
        return unsafe { ai_h2d(dst, src, bytes) };
    }
    let started = std::time::Instant::now();
    let result = unsafe { ai_h2d(dst, src, bytes) };
    host_record(0, bytes, started);
    result
}
unsafe fn host_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32 {
    if !host_profile() {
        return unsafe { ai_d2h(dst, src, bytes) };
    }
    let started = std::time::Instant::now();
    let result = unsafe { ai_d2h(dst, src, bytes) };
    host_record(1, bytes, started);
    result
}
unsafe fn host_sync() -> i32 {
    if !host_profile() {
        return unsafe { ai_sync() };
    }
    let started = std::time::Instant::now();
    let result = unsafe { ai_sync() };
    host_record(2, 0, started);
    result
}
pub fn reset_host_stats() {
    for counter in &HOST_COUNTERS {
        counter.store(0, Ordering::Relaxed);
    }
}
pub fn host_stats() -> [(usize, usize, usize); 4] {
    std::array::from_fn(|kind| {
        (
            HOST_COUNTERS[kind * 3].load(Ordering::Relaxed),
            HOST_COUNTERS[kind * 3 + 1].load(Ordering::Relaxed),
            HOST_COUNTERS[kind * 3 + 2].load(Ordering::Relaxed),
        )
    })
}
unsafe fn host_timing_end(ptr: *mut c_void, ms: *mut f32) -> i32 {
    if !host_profile() {
        return unsafe { ai_timing_end(ptr, ms) };
    }
    let started = std::time::Instant::now();
    let result = unsafe { ai_timing_end(ptr, ms) };
    host_record(3, 0, started);
    result
}
unsafe fn host_timed_linear(
    ptr: *mut c_void,
    w: *const c_void,
    x: *const f32,
    y: *mut f32,
    cols: i32,
    rows: i32,
    kind: i32,
    ms: *mut f32,
) -> i32 {
    if !host_profile() {
        return unsafe { ai_timed_linear(ptr, w, x, y, cols, rows, kind, ms) };
    }
    let started = std::time::Instant::now();
    let result = unsafe { ai_timed_linear(ptr, w, x, y, cols, rows, kind, ms) };
    host_record(3, 0, started);
    result
}
pub fn memory_stats() -> (usize, usize, usize) {
    // Runtime allocation tracking, distinct from occupancy/resource queries.
    (
        LIVE.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed),
        MIN_FREE.load(Ordering::Relaxed),
    )
}
pub fn reset_memory_stats() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    MIN_FREE.store(usize::MAX, Ordering::Relaxed);
}
pub fn gate_resources(gt: u32, ut: u32) -> Result<[i32; 8], String> {
    let mut values = [0; 8];
    check(unsafe { ai_gate_resources(gt as i32, ut as i32, values.as_mut_ptr()) })?;
    Ok(values)
}
pub fn linear_resources(kind: u32, rows: usize) -> Result<[i32; 8], String> {
    let mut values = [0; 8];
    if rows >= 512 && std::env::var("AMD_INFER_WARP_GEMV").as_deref() == Ok("1") {
        check(unsafe { ai_warp_linear_resources(kind as i32, values.as_mut_ptr()) })?;
    } else {
        check(unsafe { ai_linear_resources(kind as i32, values.as_mut_ptr()) })?;
    }
    Ok(values)
}
unsafe extern "C" {
    fn ai_linear_resources(kind: i32, values: *mut i32) -> i32;
    fn ai_warp_linear_resources(kind: i32, values: *mut i32) -> i32;
    fn ai_gate_resources(gt: i32, ut: i32, values: *mut i32) -> i32;
    fn ai_attention_prepare(
        qg: *const f32,
        k: *const f32,
        w: *const f32,
        rotation: *const f32,
        query: *mut f32,
        key: *mut f32,
    ) -> i32;
    fn ai_gdn_projected_layout(
        matrices: *const *const c_void,
        types: *const i32,
        x: *const f32,
        raw: *mut f32,
        state: *mut f32,
        history: *mut f32,
        weights: *const f32,
        mixed: *mut f32,
        core: *mut f32,
        gated: *mut f32,
        output: *mut f32,
        column: i32,
    ) -> i32;
    fn ai_gdn_pipeline_layout(
        state: *mut f32,
        history: *mut f32,
        weights: *const f32,
        raw: *const f32,
        mixed: *mut f32,
        core: *mut f32,
        gated: *mut f32,
        column: i32,
    ) -> i32;
    fn ai_read_bandwidth(input: *const f32, output: *mut f32, count: i32) -> i32;
    fn ai_ffn(
        gate: *const c_void,
        up: *const c_void,
        down: *const c_void,
        gate_type: i32,
        up_type: i32,
        down_type: i32,
        norm: *const f32,
        input: *mut f32,
        output: *mut f32,
        persistent_blocks: i32,
        specialize: i32,
        warp_reduce: i32,
        warp_rows: i32,
        timing: *mut c_void,
    ) -> i32;
    fn ai_timing_new(out: *mut *mut c_void) -> i32;
    fn ai_timing_free(ptr: *mut c_void) -> i32;
    fn ai_timing_begin(ptr: *mut c_void) -> i32;
    fn ai_timing_end(ptr: *mut c_void, ms: *mut f32) -> i32;
    fn ai_timing_ffn_parts(ptr: *mut c_void, norm: *mut f32, gate: *mut f32, down: *mut f32)
        -> i32;
    fn ai_timed_linear(
        ptr: *mut c_void,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        cols: i32,
        rows: i32,
        kind: i32,
        ms: *mut f32,
    ) -> i32;
    // Narrow device ABI; all shapes validated by Rust callers.
    fn ai_d2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    fn ai_residual(hidden: *mut f32, output: *const f32) -> i32;
    fn ai_rms(x: *const f32, w: *const f32, out: *mut f32, cols: i32) -> i32;
    fn ai_attention(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        out: *mut f32,
        count: i32,
    ) -> i32;
    fn ai_delta(state: *mut f32, input: *const f32, out: *mut f32) -> i32;
    fn ai_dequant(w: *const c_void, y: *mut f32, cols: i32, kind: i32) -> i32;
    fn ai_debug_quant(w: *const c_void, y: *mut f32, kind: i32) -> i32;
    fn ai_alloc(p: *mut *mut c_void, n: usize) -> i32;
    fn ai_zero(p: *mut c_void, n: usize) -> i32;
    fn ai_free(p: *mut c_void) -> i32;
    fn ai_h2d(dst: *mut c_void, src: *const c_void, n: usize) -> i32;
    fn ai_d2h(dst: *mut c_void, src: *const c_void, n: usize) -> i32;
    fn ai_error(e: i32) -> *const c_char;
    fn ai_sync() -> i32;
    fn ai_graph_stats(values: *mut u64, reset: i32) -> i32;
    fn ai_gdn_profile_stats(values: *mut u64, reset: i32) -> i32;
    fn ai_attention_profile_mark(slot: i32) -> i32;
    fn ai_attention_profile_finish() -> i32;
    fn ai_attention_profile_stats(values: *mut u64, reset: i32) -> i32;
    fn ai_quant_linear(
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        cols: i32,
        rows: i32,
        batch: i32,
        kind: i32,
    ) -> i32;
    fn ai_memory(free: *mut usize, total: *mut usize) -> i32;
}
// Slot 24 measures the entire device FFN chain, not a quantization type.
static KERNEL_NS: [std::sync::atomic::AtomicU64; 29] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 29];
static KERNEL_CALLS: [AtomicUsize; 29] = [const { AtomicUsize::new(0) }; 29];
pub fn kernel_stats() -> Vec<(u32, u64, usize)> {
    (0..29)
        .filter_map(|i| {
            let n = KERNEL_CALLS[i].load(Ordering::Relaxed);
            (n > 0).then(|| (i as u32, KERNEL_NS[i].load(Ordering::Relaxed), n))
        })
        .collect()
}
pub fn reset_kernel_stats() {
    for i in 0..29 {
        KERNEL_NS[i].store(0, Ordering::Relaxed);
        KERNEL_CALLS[i].store(0, Ordering::Relaxed);
    }
}
struct Timing(NonNull<c_void>);
impl Drop for Timing {
    fn drop(&mut self) {
        unsafe {
            ai_timing_free(self.0.as_ptr());
        }
    }
}
fn float_buffer(values: &[f32]) -> Result<Buffer, String> {
    let b = Buffer::new(std::mem::size_of_val(values))?;
    b.upload(unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values))
    })?;
    Ok(b)
}
fn read_floats(b: &Buffer, n: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0f32; n];
    check(unsafe { host_sync() })?;
    check(unsafe { host_d2h(out.as_mut_ptr().cast(), b.ptr.as_ptr(), n * 4) })?;
    Ok(out)
}
pub fn rms(x: &[f32], w: &[f32]) -> Result<Vec<f32>, String> {
    if x.is_empty() || x.len() != w.len() || x.len() > i32::MAX as usize {
        return Err("GPU RMS shape mismatch".into());
    }
    let bx = float_buffer(x)?;
    let bw = float_buffer(w)?;
    let by = Buffer::new(x.len() * 4)?;
    check(unsafe {
        ai_rms(
            bx.ptr.as_ptr().cast(),
            bw.ptr.as_ptr().cast(),
            by.ptr.as_ptr().cast(),
            x.len() as i32,
        )
    })?;
    read_floats(&by, x.len())
}
pub fn attention(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    gate: &[f32],
) -> Result<Vec<f32>, String> {
    if q.len() != 6144
        || gate.len() != 6144
        || keys.is_empty()
        || keys.len() > 512
        || keys.len() != values.len()
        || keys.iter().chain(values).any(|v| v.len() != 1024)
    {
        return Err("GPU attention shape mismatch".into());
    }
    let k: Vec<f32> = keys.iter().flatten().copied().collect();
    let v: Vec<f32> = values.iter().flatten().copied().collect();
    let bq = float_buffer(q)?;
    let bk = float_buffer(&k)?;
    let bv = float_buffer(&v)?;
    let bg = float_buffer(gate)?;
    let out = Buffer::new(6144 * 4)?;
    check(unsafe {
        ai_attention(
            bq.ptr.as_ptr().cast(),
            bk.ptr.as_ptr().cast(),
            bv.ptr.as_ptr().cast(),
            bg.ptr.as_ptr().cast(),
            out.ptr.as_ptr().cast(),
            keys.len() as i32,
        )
    })?;
    read_floats(&out, 6144)
}
/// Appends only the new KV row; reusable device cache and query/output scratch.
pub struct AttentionGpu {
    buffer: Buffer,
    count: usize,
    capacity: usize,
    prepared: Option<(Buffer, Buffer)>,
}
impl AttentionGpu {
    pub fn snapshot_prepared(&self) -> Result<[Vec<f32>; 3], String> {
        check(unsafe { host_sync() })?;
        let raw = read_floats(
            &self
                .prepared
                .as_ref()
                .ok_or("attention preparation missing")?
                .0,
            14336,
        )?;
        let mut qgo = vec![0.; 3 * 6144];
        let mut kv = vec![0.; 2048];
        let position = self.count.checked_sub(1).ok_or("attention cache empty")?;
        check(unsafe {
            host_d2h(
                qgo.as_mut_ptr().cast(),
                self.at(2 * self.capacity * 1024).cast(),
                qgo.len() * 4,
            )
        })?;
        check(unsafe {
            host_d2h(
                kv.as_mut_ptr().cast(),
                self.at(position * 1024).cast(),
                1024 * 4,
            )
        })?;
        check(unsafe {
            host_d2h(
                kv.as_mut_ptr().add(1024).cast(),
                self.at(self.capacity * 1024 + position * 1024).cast(),
                1024 * 4,
            )
        })?;
        Ok([raw, qgo, kv])
    }
    pub fn snapshot_cache(&self) -> Result<Vec<f32>, String> {
        read_floats(&self.buffer, 2 * self.capacity * 1024)
    }

    pub fn new() -> Result<Self, String> {
        Self::with_capacity(128)
    }
    pub fn with_capacity(capacity: usize) -> Result<Self, String> {
        if ![128, 256, 512].contains(&capacity) {
            return Err("GPU attention capacity must be 128, 256 or 512".into());
        }
        Ok(Self {
            buffer: Buffer::new((2 * capacity * 1024 + 3 * 6144) * 4)?,
            count: 0,
            capacity,
            prepared: None,
        })
    }
    pub fn reset(&mut self) {
        self.count = 0;
    } // Old rows outside count are never read.
    pub fn prepare_weights(&mut self, q: &[f32], k: &[f32]) -> Result<(), String> {
        if q.len() != 256 || k.len() != 256 {
            return Err("attention norm weight shape".into());
        }
        if self.prepared.is_none() {
            let mut weights = q.to_vec();
            weights.extend_from_slice(k);
            self.prepared = Some((Buffer::new((12288 + 2048) * 4)?, float_buffer(&weights)?));
        }
        Ok(())
    }
    fn at(&self, index: usize) -> *mut f32 {
        unsafe { self.buffer.ptr.as_ptr().cast::<f32>().add(index) }
    }
    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.count >= self.capacity
            || q.len() != 6144
            || gate.len() != 6144
            || k.len() != 1024
            || v.len() != 1024
        {
            return Err("persistent attention cache shape/capacity mismatch".into());
        }
        let keys = self.at(0);
        let values = self.at(self.capacity * 1024);
        let query = self.at(2 * self.capacity * 1024);
        let out = self.at(2 * self.capacity * 1024 + 2 * 6144);
        let mut input = Vec::with_capacity(2 * 6144);
        input.extend_from_slice(q);
        input.extend_from_slice(gate);
        check(unsafe { host_h2d(query.cast(), input.as_ptr().cast(), input.len() * 4) })?;
        check(unsafe {
            host_h2d(
                keys.add(self.count * 1024).cast(),
                k.as_ptr().cast(),
                1024 * 4,
            )
        })?;
        check(unsafe {
            host_h2d(
                values.add(self.count * 1024).cast(),
                v.as_ptr().cast(),
                1024 * 4,
            )
        })?;
        check(unsafe {
            ai_attention(
                query,
                keys,
                values,
                query.add(6144),
                out,
                (self.count + 1) as i32,
            )
        })?;
        check(unsafe { host_sync() })?;
        let mut output = vec![0.; 6144];
        check(unsafe { host_d2h(output.as_mut_ptr().cast(), out.cast(), 6144 * 4) })?;
        self.count += 1;
        Ok(output)
    }
}
pub struct DeltaGpu {
    state: Buffer,
}
pub struct GdnPipelineGpu {
    buffer: Buffer,
    // Captured at reset: environment changes cannot reinterpret live state.
    column: bool,
}
impl GdnPipelineGpu {
    /// Diagnostic only: recurrent state followed by per-channel convolution history.
    pub fn snapshot_state(&self) -> Result<Vec<f32>, String> {
        let mut state = read_floats(&self.buffer, Self::WEIGHTS)?;
        if self.column {
            let physical = state[..Self::STATE].to_vec();
            for h in 0..48 {
                for value in 0..128 {
                    for key in 0..128 {
                        state[h * 16384 + value * 128 + key] =
                            physical[h * 16384 + key * 128 + value];
                    }
                }
            }
        }
        Ok(state)
    }
    pub fn snapshot_stages(&self) -> Result<[Vec<f32>; 4], String> {
        let mut parts = [
            vec![0.; 16480],
            vec![0.; 10336],
            vec![0.; 6144],
            vec![0.; 6144],
        ];
        check(unsafe { host_sync() })?;
        for (part, offset) in
            parts
                .iter_mut()
                .zip([Self::RAW, Self::MIXED, Self::CORE, Self::GATED])
        {
            check(unsafe {
                host_d2h(
                    part.as_mut_ptr().cast(),
                    self.at(offset).cast(),
                    part.len() * 4,
                )
            })?;
        }
        Ok(parts)
    }
    pub fn step_projected(
        &mut self,
        matrices: [&PackedLinear; 5],
        x: &[f32],
        workspace: &LinearWorkspace,
    ) -> Result<Vec<f32>, String> {
        self.projected_dispatch(matrices, Some(x), workspace)?;
        read_floats(&workspace.output, 5120)
    }
    fn projected_dispatch(
        &mut self,
        matrices: [&PackedLinear; 5],
        x: Option<&[f32]>,
        workspace: &LinearWorkspace,
    ) -> Result<(), String> {
        let shapes = [
            (5120, 10240),
            (5120, 6144),
            (5120, 48),
            (5120, 48),
            (6144, 5120),
        ];
        if x.is_some_and(|v| v.len() != 5120)
            || workspace.input.len < 5120 * 4
            || workspace.output.len < 5120 * 4
            || matrices
                .iter()
                .zip(shapes)
                .any(|(m, (c, r))| m.cols != c || m.rows != r)
        {
            return Err("projected GDN shape mismatch".into());
        }
        let pointers = matrices.map(|m| unsafe {
            m.weights
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(m.offset)
                .cast::<c_void>() as *const c_void
        });
        let types = matrices.map(|m| m.kind as i32);
        if let Some(x) = x {
            check(unsafe {
                host_h2d(workspace.input.ptr.as_ptr(), x.as_ptr().cast(), x.len() * 4)
            })?;
        }
        if let Some(t) = &workspace.timing {
            check(unsafe { ai_timing_begin(t.0.as_ptr()) })?;
        }
        check(unsafe {
            ai_gdn_projected_layout(
                pointers.as_ptr(),
                types.as_ptr(),
                workspace.input.ptr.as_ptr().cast(),
                self.at(Self::RAW),
                self.at(0),
                self.at(Self::HISTORY),
                self.at(Self::WEIGHTS),
                self.at(Self::MIXED),
                self.at(Self::CORE),
                self.at(Self::GATED),
                workspace.output.ptr.as_ptr().cast(),
                i32::from(self.column),
            )
        })?;
        if let Some(t) = &workspace.timing {
            let mut ms = 0.;
            check(unsafe { host_timing_end(t.0.as_ptr(), &mut ms) })?;
            KERNEL_NS[25].fetch_add((ms as f64 * 1e6) as u64, Ordering::Relaxed);
            KERNEL_CALLS[25].fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
    const STATE: usize = 48 * 128 * 128;
    const HISTORY: usize = Self::STATE;
    const WEIGHTS: usize = Self::HISTORY + 3 * 10240;
    const RAW: usize = Self::WEIGHTS + 40960 + 128 + 48 + 48;
    const MIXED: usize = Self::RAW + 10240 + 6144 + 48 + 48;
    const CORE: usize = Self::MIXED + 10336;
    const GATED: usize = Self::CORE + 6144;
    pub fn new(conv: &[f32], norm: &[f32], a: &[f32], bias: &[f32]) -> Result<Self, String> {
        if conv.len() != 40960 || norm.len() != 128 || a.len() != 48 || bias.len() != 48 {
            return Err("GDN pipeline weight shape mismatch".into());
        }
        let mut weights = Vec::with_capacity(41184);
        for slice in [conv, norm, a, bias] {
            weights.extend_from_slice(slice);
        }
        let mut result = Self {
            buffer: Buffer::new((Self::GATED + 6144) * 4)?,
            column: false,
        };
        check(unsafe {
            host_h2d(
                result.at(Self::WEIGHTS).cast(),
                weights.as_ptr().cast(),
                weights.len() * 4,
            )
        })?;
        result.reset()?;
        Ok(result)
    }
    fn at(&self, index: usize) -> *mut f32 {
        unsafe { self.buffer.ptr.as_ptr().cast::<f32>().add(index) }
    }
    pub fn reset(&mut self) -> Result<(), String> {
        let column = match std::env::var("AMD_INFER_GDN_COLUMN").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            _ => return Err("AMD_INFER_GDN_COLUMN must be 0 or 1".into()),
        };
        check(unsafe { ai_zero(self.buffer.ptr.as_ptr(), (Self::WEIGHTS) * 4) })?;
        self.column = column;
        check(unsafe { host_sync() })
    }
    pub fn step(
        &mut self,
        qkv: &[f32],
        z: &[f32],
        alpha: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, String> {
        if qkv.len() != 10240 || z.len() != 6144 || alpha.len() != 48 || beta.len() != 48 {
            return Err("GDN pipeline input shape mismatch".into());
        }
        let mut raw = Vec::with_capacity(16480);
        for slice in [qkv, z, alpha, beta] {
            raw.extend_from_slice(slice);
        }
        check(unsafe {
            host_h2d(
                self.at(Self::RAW).cast(),
                raw.as_ptr().cast(),
                raw.len() * 4,
            )
        })?;
        check(unsafe {
            ai_gdn_pipeline_layout(
                self.at(0),
                self.at(Self::HISTORY),
                self.at(Self::WEIGHTS),
                self.at(Self::RAW),
                self.at(Self::MIXED),
                self.at(Self::CORE),
                self.at(Self::GATED),
                i32::from(self.column),
            )
        })?;
        check(unsafe { host_sync() })?;
        let mut out = vec![0.; 6144];
        check(unsafe {
            host_d2h(
                out.as_mut_ptr().cast(),
                self.at(Self::GATED).cast(),
                6144 * 4,
            )
        })?;
        Ok(out)
    }
}
impl DeltaGpu {
    pub fn snapshot_state(&self) -> Result<Vec<f32>, String> {
        read_floats(&self.state, 48 * 128 * 128)
    }
    pub fn new() -> Result<Self, String> {
        let state = Buffer::new(48 * 128 * 128 * 4 + 10336 * 4 + 6144 * 4)?;
        check(unsafe { ai_zero(state.ptr.as_ptr(), state.len) })?;
        Ok(Self { state })
    }
    pub fn reset(&mut self) -> Result<(), String> {
        check(unsafe { ai_zero(self.state.ptr.as_ptr(), 48 * 128 * 128 * 4) })?;
        check(unsafe { host_sync() })
    }
    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, String> {
        if q.len() != 2048
            || k.len() != 2048
            || v.len() != 6144
            || g.len() != 48
            || beta.len() != 48
        {
            return Err("GPU DeltaNet shape mismatch".into());
        }
        let mut input = Vec::with_capacity(10336);
        for values in [q, k, v, g, beta] {
            input.extend_from_slice(values);
        }
        let bx = unsafe {
            self.state
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(48 * 128 * 128 * 4)
                .cast::<c_void>()
        };
        let by = unsafe { bx.cast::<u8>().add(10336 * 4).cast::<c_void>() };
        check(unsafe { host_h2d(bx, input.as_ptr().cast(), input.len() * 4) })?;
        let mut out = vec![0f32; 6144];
        check(unsafe { ai_delta(self.state.ptr.as_ptr().cast(), bx.cast(), by.cast()) })?;
        check(unsafe { host_sync() })?;
        check(unsafe { host_d2h(out.as_mut_ptr().cast(), by, 6144 * 4) })?;
        Ok(out)
    }
}
fn check(e: i32) -> Result<(), String> {
    if e == 0 {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(ai_error(e)).to_string_lossy().into_owned() })
    }
}
// Development-only probe ABI.
unsafe extern "C" {
    fn ai_timed_regrouped_probe(
        ptr: *mut c_void,
        w: *const c_void,
        x: *const f32,
        y: *mut f32,
        cols: i32,
        rows: i32,
        kind: i32,
        ms: *mut f32,
    ) -> i32;
}
pub struct CacheScrub {
    input: Buffer,
    output: Buffer,
}
impl CacheScrub {
    pub fn new() -> Result<Self, String> {
        let input = Buffer::new(256 * 1024 * 1024)?;
        let output = Buffer::new(2048 * 4)?;
        check(unsafe { ai_zero(input.ptr.as_ptr(), input.len) })?;
        Ok(Self { input, output })
    }
    pub fn run(&self) -> Result<(), String> {
        check(unsafe {
            ai_read_bandwidth(
                self.input.ptr.as_ptr().cast(),
                self.output.ptr.as_ptr().cast(),
                (self.input.len / 4) as i32,
            )
        })?;
        check(unsafe { host_sync() })
    }
}
struct Buffer {
    ptr: NonNull<c_void>,
    len: usize,
}
impl Buffer {
    fn new(len: usize) -> Result<Self, String> {
        if len == 0 {
            return Err("empty allocation".into());
        }
        let mut p = std::ptr::null_mut();
        if std::env::var("AMD_INFER_ENFORCE_RESERVE").as_deref() == Ok("1") {
            let (free, _) = memory()?;
            if len
                .checked_add(2 * 1024 * 1024 * 1024 + 16 * 1024 * 1024)
                .is_none_or(|required| required > free)
            {
                return Err(format!(
                    "allocation would consume reserve: requested={len} free={free}"
                ));
            }
        }
        check(unsafe { ai_alloc(&mut p, len) })?;
        let live = LIVE.fetch_add(len, Ordering::Relaxed) + len;
        PEAK.fetch_max(live, Ordering::Relaxed);
        if std::env::var("AMD_INFER_MEMORY_TRACE").as_deref() == Ok("1") {
            let (mut free, mut total) = (0, 0);
            if unsafe { ai_memory(&mut free, &mut total) } == 0 {
                MIN_FREE.fetch_min(free, Ordering::Relaxed);
            }
        }
        Ok(Self {
            ptr: NonNull::new(p).ok_or("null HIP allocation")?,
            len,
        })
    }
    fn upload(&self, b: &[u8]) -> Result<(), String> {
        if b.len() != self.len {
            return Err("upload length mismatch".into());
        }
        check(unsafe { host_h2d(self.ptr.as_ptr(), b.as_ptr().cast(), b.len()) })
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.len, Ordering::Relaxed);
        unsafe {
            ai_free(self.ptr.as_ptr());
        }
    }
}
pub fn memory() -> Result<(usize, usize), String> {
    let (mut f, mut t) = (0, 0);
    check(unsafe { ai_memory(&mut f, &mut t) })?;
    Ok((f, t))
}
/// Packed resident matrix; no expansion of quantized weights.
pub struct PackedLinear {
    weights: std::rc::Rc<Buffer>,
    offset: usize,
    cols: usize,
    rows: usize,
    kind: u32,
    row_bytes: usize,
}
pub struct PackedStorage(std::rc::Rc<Buffer>);
pub struct NormBank(Buffer);
pub fn diagnose_read_bandwidth() -> Result<Vec<f64>, String> {
    let bytes = 256 * 1024 * 1024;
    let input = Buffer::new(bytes)?;
    let output = Buffer::new(2048 * 4)?;
    check(unsafe { ai_zero(input.ptr.as_ptr(), bytes) })?;
    let mut ptr = std::ptr::null_mut();
    check(unsafe { ai_timing_new(&mut ptr) })?;
    let timing = Timing(NonNull::new(ptr).ok_or("null timing context")?);
    check(unsafe {
        ai_read_bandwidth(
            input.ptr.as_ptr().cast(),
            output.ptr.as_ptr().cast(),
            (bytes / 4) as i32,
        )
    })?;
    check(unsafe { host_sync() })?;
    let mut samples = Vec::new();
    for _ in 0..8 {
        check(unsafe { ai_timing_begin(timing.0.as_ptr()) })?;
        check(unsafe {
            ai_read_bandwidth(
                input.ptr.as_ptr().cast(),
                output.ptr.as_ptr().cast(),
                (bytes / 4) as i32,
            )
        })?;
        let mut ms = 0.;
        check(unsafe { host_timing_end(timing.0.as_ptr(), &mut ms) })?;
        if ms <= 0. {
            return Err("invalid bandwidth event duration".into());
        }
        samples.push(bytes as f64 / (ms as f64 * 1e6));
    }
    if read_floats(&output, 2048)?.iter().any(|v| *v != 0.) {
        return Err("bandwidth checksum mismatch".into());
    }
    Ok(samples)
}
impl NormBank {
    pub fn new(values: &[f32]) -> Result<Self, String> {
        if values.is_empty()
            || !values.len().is_multiple_of(5120)
            || values.len() > 64 * 5120
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("invalid resident norm bank".into());
        }
        Ok(Self(float_buffer(values)?))
    }
    fn row(&self, index: usize) -> Result<*const f32, String> {
        if index >= self.0.len / (5120 * 4) {
            return Err("norm bank layer out of bounds".into());
        }
        Ok(unsafe { self.0.ptr.as_ptr().cast::<f32>().add(index * 5120) })
    }
}
pub fn ffn(
    gate: &PackedLinear,
    up: &PackedLinear,
    down: &PackedLinear,
    x: &[f32],
    norm: &[f32],
    workspace: &LinearWorkspace,
    resident_norm: Option<(&NormBank, usize)>,
) -> Result<Vec<f32>, String> {
    ffn_dispatch(gate, up, down, Some(x), norm, workspace, resident_norm)?;
    read_floats(&workspace.output, 5120)
}
fn ffn_dispatch(
    gate: &PackedLinear,
    up: &PackedLinear,
    down: &PackedLinear,
    x: Option<&[f32]>,
    norm: &[f32],
    workspace: &LinearWorkspace,
    resident_norm: Option<(&NormBank, usize)>,
) -> Result<(), String> {
    if gate.cols != 5120
        || up.cols != 5120
        || down.cols != 17408
        || gate.rows != 17408
        || up.rows != 17408
        || down.rows != 5120
        || x.is_some_and(|v| v.len() != 5120)
        || norm.len() != 5120
        || workspace.input.len < 17408 * 4
        || workspace.output.len < 34816 * 4
    {
        return Err("device FFN shape mismatch".into());
    }
    let owned_weight = if resident_norm.is_some()
        || std::env::var("AMD_INFER_REUSE_FFN_NORM").as_deref() == Ok("1")
    {
        None
    } else {
        Some(float_buffer(norm)?)
    };
    let norm_ptr = if let Some((bank, index)) = resident_norm {
        bank.row(index)?
    } else if let Some(weight) = &owned_weight {
        weight.ptr.as_ptr().cast()
    } else {
        check(unsafe { host_h2d(workspace.norm.ptr.as_ptr(), norm.as_ptr().cast(), 5120 * 4) })?;
        workspace.norm.ptr.as_ptr().cast()
    };
    if let Some(x) = x {
        check(unsafe { host_h2d(workspace.input.ptr.as_ptr(), x.as_ptr().cast(), 5120 * 4) })?;
    }
    if let Some(timing) = &workspace.timing {
        check(unsafe { ai_timing_begin(timing.0.as_ptr()) })?;
    }
    check(unsafe {
        ai_ffn(
            gate.weights
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(gate.offset)
                .cast(),
            up.weights.ptr.as_ptr().cast::<u8>().add(up.offset).cast(),
            down.weights
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(down.offset)
                .cast(),
            gate.kind as i32,
            up.kind as i32,
            down.kind as i32,
            norm_ptr,
            workspace.input.ptr.as_ptr().cast(),
            workspace.output.ptr.as_ptr().cast(),
            std::env::var("AMD_INFER_PERSISTENT_FFN")
                .unwrap_or("0".into())
                .parse::<i32>()
                .map_err(|_| "invalid persistent FFN block count")?,
            i32::from(std::env::var("AMD_INFER_FFN_SPECIALIZE").as_deref() == Ok("1")),
            i32::from(std::env::var("AMD_INFER_FFN_WARP_REDUCE").as_deref() == Ok("1")),
            i32::from(std::env::var("AMD_INFER_FFN_WARP_ROWS").as_deref() == Ok("1")),
            workspace
                .timing
                .as_ref()
                .map_or(std::ptr::null_mut(), |t| t.0.as_ptr()),
        )
    })?;
    if let Some(timing) = &workspace.timing {
        let mut ms = 0.;
        check(unsafe { host_timing_end(timing.0.as_ptr(), &mut ms) })?;
        KERNEL_NS[24].fetch_add((ms as f64 * 1e6) as u64, Ordering::Relaxed);
        KERNEL_CALLS[24].fetch_add(1, Ordering::Relaxed);
        let (mut norm, mut gate, mut down) = (0., 0., 0.);
        check(unsafe { ai_timing_ffn_parts(timing.0.as_ptr(), &mut norm, &mut gate, &mut down) })?;
        for (slot, value) in [(26, norm), (27, gate), (28, down)] {
            KERNEL_NS[slot].fetch_add((value as f64 * 1e6) as u64, Ordering::Relaxed);
            KERNEL_CALLS[slot].fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(())
}
/// Hidden vector survives across contiguous device layers. Scratch buffers remain
/// owned by the existing workspace; default-stream kernel boundaries order all uses.
pub struct DeviceHidden(Buffer, Buffer);
impl DeviceHidden {
    pub fn new() -> Result<Self, String> {
        Ok(Self(Buffer::new(5120 * 4)?, Buffer::new(64 * 4)?))
    }
    pub fn upload(&self, x: &[f32]) -> Result<(), String> {
        if x.len() != 5120 {
            return Err("device hidden shape mismatch".into());
        }
        check(unsafe { host_h2d(self.0.ptr.as_ptr(), x.as_ptr().cast(), 5120 * 4) })
    }
    pub fn download(&self) -> Result<Vec<f32>, String> {
        read_floats(&self.0, 5120)
    }
    pub fn rotation(&self, coefficients: &[f32]) -> Result<(), String> {
        if coefficients.len() != 64 {
            return Err("RoPE coefficients shape".into());
        }
        check(unsafe { host_h2d(self.1.ptr.as_ptr(), coefficients.as_ptr().cast(), 64 * 4) })
    }
    pub fn attention_layer(
        &self,
        cache: &mut AttentionGpu,
        matrices: [&PackedLinear; 4],
        ffn: [&PackedLinear; 3],
        norms: (&NormBank, &NormBank, usize),
        workspace: &LinearWorkspace,
    ) -> Result<(), String> {
        let shapes = [(5120, 12288), (5120, 1024), (5120, 1024), (6144, 5120)];
        if cache.count >= cache.capacity
            || matrices
                .iter()
                .zip(shapes)
                .any(|(m, (c, r))| m.cols != c || m.rows != r)
        {
            return Err("device attention shapes/capacity".into());
        }
        let (scratch, weights) = cache
            .prepared
            .as_ref()
            .ok_or("attention preparation weights missing")?;
        let raw = scratch.ptr.as_ptr().cast::<f32>();
        attention_mark(0)?;
        check(unsafe {
            ai_rms(
                self.0.ptr.as_ptr().cast(),
                norms.0.row(norms.2)?,
                workspace.input.ptr.as_ptr().cast(),
                5120,
            )
        })?;
        attention_mark(1)?;
        for (index, (m, offset)) in matrices[..3].iter().zip([0, 12288, 13312]).enumerate() {
            unsafe {
                m.launch_device(
                    workspace.input.ptr.as_ptr().cast(),
                    raw.add(offset),
                    workspace,
                )?;
            }
            attention_mark(2 + index as i32)?;
        }
        let keys = cache.at(0);
        let values = cache.at(cache.capacity * 1024);
        let query = cache.at(2 * cache.capacity * 1024);
        let out = cache.at(2 * cache.capacity * 1024 + 12288);
        check(unsafe {
            ai_attention_prepare(
                raw,
                raw.add(12288),
                weights.ptr.as_ptr().cast(),
                self.1.ptr.as_ptr().cast(),
                query,
                keys.add(cache.count * 1024),
            )
        })?;
        check(unsafe {
            ai_d2d(
                values.add(cache.count * 1024).cast(),
                raw.add(13312).cast(),
                1024 * 4,
            )
        })?;
        attention_mark(5)?;
        check(unsafe {
            ai_attention(
                query,
                keys,
                values,
                query.add(6144),
                out,
                (cache.count + 1) as i32,
            )
        })?;
        attention_mark(6)?;
        let m = matrices[3];
        unsafe {
            m.launch_device(out, workspace.output.ptr.as_ptr().cast(), workspace)?;
        }
        attention_mark(7)?;
        attention_finish()?;
        cache.count += 1;
        self.finish_layer(ffn, (norms.1, norms.2), workspace)
    }
    pub fn gdn_layer(
        &self,
        pipeline: &mut GdnPipelineGpu,
        matrices: [&PackedLinear; 5],
        ffn: [&PackedLinear; 3],
        norms: (&NormBank, &NormBank, usize),
        workspace: &LinearWorkspace,
    ) -> Result<(), String> {
        let (attention_norm, ffn_norm, layer) = norms;
        check(unsafe {
            ai_rms(
                self.0.ptr.as_ptr().cast(),
                attention_norm.row(layer)?,
                workspace.input.ptr.as_ptr().cast(),
                5120,
            )
        })?;
        pipeline.projected_dispatch(matrices, None, workspace)?;
        self.finish_layer(ffn, (ffn_norm, layer), workspace)
    }
    fn finish_layer(
        &self,
        ffn: [&PackedLinear; 3],
        norms: (&NormBank, usize),
        workspace: &LinearWorkspace,
    ) -> Result<(), String> {
        check(unsafe {
            ai_residual(
                self.0.ptr.as_ptr().cast(),
                workspace.output.ptr.as_ptr().cast(),
            )
        })?;
        // Copy hidden to FFN scratch on device: ai_rms with unit weights would
        // change arithmetic, so use the HIP device-to-device copy ABI instead.
        check(unsafe { ai_d2d(workspace.input.ptr.as_ptr(), self.0.ptr.as_ptr(), 5120 * 4) })?;
        ffn_dispatch(
            ffn[0],
            ffn[1],
            ffn[2],
            None,
            &[0.; 5120],
            workspace,
            Some(norms),
        )?;
        check(unsafe {
            ai_residual(
                self.0.ptr.as_ptr().cast(),
                workspace.output.ptr.as_ptr().cast(),
            )
        })
    }
}
pub struct LinearWorkspace {
    input: Buffer,
    output: Buffer,
    norm: Buffer,
    timing: Option<Timing>,
}
impl LinearWorkspace {
    /// Development-only readback;does not change any model buffers or dispatch.
    pub fn snapshot_ffn(&self) -> Result<[Vec<f32>; 3], String> {
        check(unsafe { host_sync() })?;
        let mut act = vec![0.; 17408];
        check(unsafe {
            host_d2h(
                act.as_mut_ptr().cast(),
                self.output.ptr.as_ptr().add(17408 * 4),
                17408 * 4,
            )
        })?;
        Ok([
            read_floats(&self.input, 5120)?,
            act,
            read_floats(&self.output, 5120)?,
        ])
    }

    pub fn new(cols: usize, rows: usize) -> Result<Self, String> {
        let timing = if std::env::var("AMD_INFER_PROFILE_KERNEL").as_deref() == Ok("1") {
            let mut p = std::ptr::null_mut();
            check(unsafe { ai_timing_new(&mut p) })?;
            Some(Timing(NonNull::new(p).ok_or("null timing context")?))
        } else {
            None
        };
        Ok(Self {
            norm: Buffer::new(5120 * 4)?,
            input: Buffer::new(cols.checked_mul(4).ok_or("workspace size overflow")?)?,
            output: Buffer::new(rows.checked_mul(4).ok_or("workspace size overflow")?)?,
            timing,
        })
    }
}
impl PackedStorage {
    pub fn allocate(bytes: usize) -> Result<Self, String> {
        Ok(Self(std::rc::Rc::new(Buffer::new(bytes)?)))
    }
}
impl PackedLinear {
    /// Private: callers validate input/output shapes and retain all buffers.
    unsafe fn launch_device(
        &self,
        x: *const f32,
        y: *mut f32,
        workspace: &LinearWorkspace,
    ) -> Result<(), String> {
        let weight = unsafe {
            self.weights
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(self.offset)
                .cast()
        };
        if let Some(timing) = &workspace.timing {
            let mut ms = 0.;
            check(unsafe {
                host_timed_linear(
                    timing.0.as_ptr(),
                    weight,
                    x,
                    y,
                    self.cols as i32,
                    self.rows as i32,
                    self.kind as i32,
                    &mut ms,
                )
            })?;
            KERNEL_NS[self.kind as usize].fetch_add((ms as f64 * 1e6) as u64, Ordering::Relaxed);
            KERNEL_CALLS[self.kind as usize].fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            check(unsafe {
                ai_quant_linear(
                    weight,
                    x,
                    y,
                    self.cols as i32,
                    self.rows as i32,
                    1,
                    self.kind as i32,
                )
            })
        }
    }
    pub fn in_storage(
        storage: &PackedStorage,
        offset: usize,
        cols: usize,
        rows: usize,
        kind: u32,
    ) -> Result<Self, String> {
        let (block, size) = crate::gguf::type_layout(kind).ok_or("unsupported matrix type")?;
        if cols == 0
            || rows == 0
            || !cols.is_multiple_of(256)
            || cols > i32::MAX as usize
            || rows > i32::MAX as usize
        {
            return Err("invalid resident matrix shape".into());
        }
        let row_bytes = cols / block as usize * size as usize;
        let len = row_bytes.checked_mul(rows).ok_or("matrix size overflow")?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > storage.0.len)
            || !offset.is_multiple_of(16)
        {
            return Err("resident storage bounds or alignment".into());
        }
        Ok(Self {
            weights: storage.0.clone(),
            offset,
            cols,
            rows,
            kind,
            row_bytes,
        })
    }
    pub fn upload_rows(&self, start: usize, raw: &[u8]) -> Result<(), String> {
        let offset = start
            .checked_mul(self.row_bytes)
            .ok_or("row offset overflow")?;
        if !raw.len().is_multiple_of(self.row_bytes)
            || offset
                .checked_add(raw.len())
                .is_none_or(|end| end > self.row_bytes * self.rows)
        {
            return Err("resident upload bounds".into());
        }
        check(unsafe {
            host_h2d(
                self.weights
                    .ptr
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.offset + offset)
                    .cast(),
                raw.as_ptr().cast(),
                raw.len(),
            )
        })
    }
    pub fn run(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let workspace = LinearWorkspace::new(self.cols, self.rows)?;
        self.run_workspace(x, &workspace)
    }
    /// Development-only arithmetic decision probe. Not used by model dispatch.
    pub fn run_regrouped_probe(
        &self,
        x: &[f32],
        workspace: &LinearWorkspace,
    ) -> Result<(Vec<f32>, f64), String> {
        if x.len() != self.cols || ![21, 23].contains(&self.kind) {
            return Err("regroup probe shape/type".into());
        }
        let timing = workspace
            .timing
            .as_ref()
            .ok_or("probe requires kernel timing")?;
        let bx = &workspace.input;
        let by = &workspace.output;
        if std::mem::size_of_val(x) > bx.len || self.rows * 4 > by.len {
            return Err("probe workspace bounds".into());
        }
        check(unsafe { host_h2d(bx.ptr.as_ptr(), x.as_ptr().cast(), std::mem::size_of_val(x)) })?;
        let mut y = vec![0f32; self.rows];
        let mut ms = 0f32;
        check(unsafe {
            ai_timed_regrouped_probe(
                timing.0.as_ptr(),
                self.weights
                    .ptr
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.offset)
                    .cast(),
                bx.ptr.as_ptr().cast(),
                by.ptr.as_ptr().cast(),
                self.cols as i32,
                self.rows as i32,
                self.kind as i32,
                &mut ms,
            )
        })?;
        check(unsafe { host_sync() })?;
        check(unsafe {
            host_d2h(
                y.as_mut_ptr().cast(),
                by.ptr.as_ptr(),
                std::mem::size_of_val(y.as_slice()),
            )
        })?;
        Ok((y, ms as f64 * 1e6))
    }
    pub fn run_workspace(
        &self,
        x: &[f32],
        workspace: &LinearWorkspace,
    ) -> Result<Vec<f32>, String> {
        if x.len() != self.cols {
            return Err("resident input shape mismatch".into());
        }
        let bx = &workspace.input;
        let by = &workspace.output;
        if std::mem::size_of_val(x) > bx.len || self.rows * 4 > by.len {
            return Err("workspace bounds".into());
        }
        check(unsafe { host_h2d(bx.ptr.as_ptr(), x.as_ptr().cast(), std::mem::size_of_val(x)) })?;
        let mut y = vec![0f32; self.rows];
        if let Some(timing) = &workspace.timing {
            let mut ms = 0f32;
            check(unsafe {
                host_timed_linear(
                    timing.0.as_ptr(),
                    self.weights
                        .ptr
                        .as_ptr()
                        .cast::<u8>()
                        .add(self.offset)
                        .cast(),
                    bx.ptr.as_ptr().cast(),
                    by.ptr.as_ptr().cast(),
                    self.cols as i32,
                    self.rows as i32,
                    self.kind as i32,
                    &mut ms,
                )
            })?;
            KERNEL_NS[self.kind as usize].fetch_add((ms as f64 * 1e6) as u64, Ordering::Relaxed);
            KERNEL_CALLS[self.kind as usize].fetch_add(1, Ordering::Relaxed);
        } else {
            check(unsafe {
                ai_quant_linear(
                    self.weights
                        .ptr
                        .as_ptr()
                        .cast::<u8>()
                        .add(self.offset)
                        .cast(),
                    bx.ptr.as_ptr().cast(),
                    by.ptr.as_ptr().cast(),
                    self.cols as i32,
                    self.rows as i32,
                    1,
                    self.kind as i32,
                )
            })?;
        }
        check(unsafe { host_sync() })?;
        check(unsafe {
            host_d2h(
                y.as_mut_ptr().cast(),
                by.ptr.as_ptr(),
                std::mem::size_of_val(y.as_slice()),
            )
        })?;
        Ok(y)
    }
}
pub fn dequant(raw: &[u8], cols: usize, kind: u32) -> Result<Vec<f32>, String> {
    let (block, size) = crate::gguf::type_layout(kind).ok_or("unsupported type")?;
    if cols == 0
        || !cols.is_multiple_of(256)
        || cols > i32::MAX as usize
        || raw.len() != cols / block as usize * size as usize
    {
        return Err("invalid dequant shape".into());
    }
    let w = Buffer::new(raw.len())?;
    w.upload(raw)?;
    let out = Buffer::new(cols * 4)?;
    let mut y = vec![0f32; cols];
    check(unsafe {
        ai_dequant(
            w.ptr.as_ptr(),
            out.ptr.as_ptr().cast(),
            cols as i32,
            kind as i32,
        )
    })?;
    check(unsafe { host_sync() })?;
    check(unsafe { host_d2h(y.as_mut_ptr().cast(), out.ptr.as_ptr(), cols * 4) })?;
    Ok(y)
}
pub fn debug_quant(raw: &[u8], kind: u32) -> Result<Vec<f32>, String> {
    let w = Buffer::new(raw.len())?;
    w.upload(raw)?;
    let out = Buffer::new(260 * 4)?;
    let mut y = vec![0f32; 260];
    check(unsafe { ai_debug_quant(w.ptr.as_ptr(), out.ptr.as_ptr().cast(), kind as i32) })?;
    check(unsafe { host_sync() })?;
    check(unsafe { host_d2h(y.as_mut_ptr().cast(), out.ptr.as_ptr(), 260 * 4) })?;
    Ok(y)
}
pub fn linear_iq4(
    weights: &[u8],
    x: &[f32],
    cols: usize,
    rows: usize,
    batch: usize,
) -> Result<Vec<f32>, String> {
    linear_quant(weights, x, cols, rows, batch, 23)
}
pub fn linear_quant(
    weights: &[u8],
    x: &[f32],
    cols: usize,
    rows: usize,
    batch: usize,
    kind: u32,
) -> Result<Vec<f32>, String> {
    if cols == 0
        || !cols.is_multiple_of(256)
        || rows == 0
        || !(1..=8).contains(&batch)
        || cols > i32::MAX as usize
        || rows > i32::MAX as usize
    {
        return Err("invalid matrix shape".into());
    }
    let (block, size) = crate::gguf::type_layout(kind).ok_or("unknown quantized format")?;
    let wn = rows
        .checked_mul(cols / block as usize)
        .and_then(|n| n.checked_mul(size as usize))
        .ok_or("weight shape overflow")?;
    if weights.len() != wn || x.len() != cols.checked_mul(batch).ok_or("input shape overflow")? {
        return Err("matrix buffer length mismatch".into());
    }
    let w = Buffer::new(wn)?;
    w.upload(weights)?;
    let bx = Buffer::new(std::mem::size_of_val(x))?;
    let raw = unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) };
    bx.upload(raw)?;
    let mut y = vec![0f32; rows.checked_mul(batch).ok_or("output shape overflow")?];
    let by = Buffer::new(std::mem::size_of_val(y.as_slice()))?;
    check(unsafe {
        ai_quant_linear(
            w.ptr.as_ptr(),
            bx.ptr.as_ptr().cast(),
            by.ptr.as_ptr().cast(),
            cols as i32,
            rows as i32,
            batch as i32,
            kind as i32,
        )
    })?;
    check(unsafe { host_sync() })?;
    check(unsafe {
        host_d2h(
            y.as_mut_ptr().cast(),
            by.ptr.as_ptr(),
            std::mem::size_of_val(y.as_slice()),
        )
    })?;
    Ok(y)
}

/// Statistics only; resetting counters preserves live graph cache and model state.
pub fn graph_stats(reset: bool) -> Result<[u64; 8], String> {
    let mut values = [0; 8];
    check(unsafe { ai_graph_stats(values.as_mut_ptr(), i32::from(reset)) })?;
    Ok(values)
}

pub fn attention_profile_stats(reset: bool) -> Result<[u64; 14], String> {
    let mut values = [0; 14];
    check(unsafe { ai_attention_profile_stats(values.as_mut_ptr(), i32::from(reset)) })?;
    Ok(values)
}
fn attention_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("AMD_INFER_PROFILE_ATTN_PARTS").as_deref() == Ok("1"))
}
fn attention_mark(slot: i32) -> Result<(), String> {
    if attention_profile_enabled() {
        check(unsafe { ai_attention_profile_mark(slot) })
    } else {
        Ok(())
    }
}
fn attention_finish() -> Result<(), String> {
    if attention_profile_enabled() {
        check(unsafe { ai_attention_profile_finish() })
    } else {
        Ok(())
    }
}
pub fn gdn_profile_stats(reset: bool) -> Result<[u64; 18], String> {
    let mut values = [0; 18];
    check(unsafe { ai_gdn_profile_stats(values.as_mut_ptr(), i32::from(reset)) })?;
    Ok(values)
}
