//! Scalar correctness specifications. Not a CPU inference or performance path.
/// Process-level diagnostic contract; historical f32-libm oracle stays available.
fn canonical_math() -> bool {
    static MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("AMD_INFER_CANONICAL_MATH").as_deref() == Ok("1"))
}
pub fn activation_exp(x: f32) -> f32 {
    if canonical_math() {
        (x as f64).exp() as f32
    } else {
        x.exp()
    }
}
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + activation_exp(-x))
    } else {
        let e = activation_exp(x);
        e / (1. + e)
    }
}
pub fn softplus(x: f32) -> f32 {
    let e = activation_exp(-x.abs());
    x.max(0.)
        + if canonical_math() {
            (e as f64).ln_1p() as f32
        } else {
            e.ln_1p()
        }
}
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, &'static str> {
    if x.is_empty() || x.len() != w.len() || eps <= 0. {
        return Err("invalid RMSNorm shape/epsilon");
    }
    let mean = x.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / x.len() as f64;
    let inv = (mean + eps as f64).sqrt().recip() as f32;
    Ok(x.iter().zip(w).map(|(a, b)| a * inv * b).collect())
}
/// L2 norm used by GDN, distinct from RMSNorm by sqrt(head_dim).
pub fn l2_norm(x: &[f32], eps: f32) -> Result<Vec<f32>, &'static str> {
    if x.is_empty() || eps <= 0. {
        return Err("invalid L2 normalization");
    }
    let norm = (x.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() + eps as f64).sqrt() as f32;
    Ok(x.iter().map(|v| v / norm).collect())
}
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, &'static str> {
    if gate.len() != up.len() {
        return Err("SwiGLU shape mismatch");
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(g, u)| g * sigmoid(*g) * u)
        .collect())
}

pub struct ConvState {
    channels: usize,
    history: Vec<f32>,
}
impl ConvState {
    pub fn snapshot_state(&self) -> &[f32] {
        &self.history
    }
    pub fn new(channels: usize) -> Result<Self, &'static str> {
        if channels == 0 {
            return Err("empty convolution");
        }
        Ok(Self {
            channels,
            history: vec![0.; channels * 3],
        })
    }
    /// GGUF kernel [4, channels], oldest-to-newest taps. State is per-channel.
    pub fn step(&mut self, x: &[f32], weights: &[f32]) -> Result<Vec<f32>, &'static str> {
        if x.len() != self.channels || weights.len() != self.channels * 4 {
            return Err("conv shape mismatch");
        }
        let mut y = vec![0.; self.channels];
        for c in 0..self.channels {
            let h = &mut self.history[c * 3..c * 3 + 3];
            let w = &weights[c * 4..c * 4 + 4];
            let raw = h[0] * w[0] + h[1] * w[1] + h[2] * w[2] + x[c] * w[3];
            y[c] = raw * sigmoid(raw);
            h[0] = h[1];
            h[1] = h[2];
            h[2] = x[c];
        }
        Ok(y)
    }
    pub fn reset(&mut self) {
        self.history.fill(0.)
    }
}

pub struct DeltaState {
    pub dim: usize,
    pub heads: usize,
    pub key_heads: usize,
    pub data: Vec<f32>,
}
impl DeltaState {
    pub fn new(dim: usize, heads: usize, key_heads: usize) -> Result<Self, &'static str> {
        if dim == 0 || heads == 0 || key_heads == 0 || !heads.is_multiple_of(key_heads) {
            return Err("invalid DeltaNet dimensions");
        }
        let n = dim
            .checked_mul(dim)
            .and_then(|x| x.checked_mul(heads))
            .ok_or("state overflow")?;
        Ok(Self {
            dim,
            heads,
            key_heads,
            data: vec![0.; n],
        })
    }
    /// Inputs q/k are L2-normalized; g is log-decay, beta is sigmoid output.
    /// Storage follows GGML: [head][value_coordinate][key_coordinate].
    pub fn step(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, &'static str> {
        let d = self.dim;
        if q.len() != d * self.key_heads
            || k.len() != q.len()
            || v.len() != d * self.heads
            || g.len() != self.heads
            || beta.len() != self.heads
        {
            return Err("DeltaNet shape mismatch");
        }
        let mut out = vec![0.; v.len()];
        let scale = (d as f32).sqrt().recip();
        for h in 0..self.heads {
            let kh = h % self.key_heads;
            let qh = &q[kh * d..(kh + 1) * d];
            let kk = &k[kh * d..(kh + 1) * d];
            let decay = g[h].exp();
            for j in 0..d {
                let row = &mut self.data[(h * d + j) * d..(h * d + j + 1) * d];
                for s in row.iter_mut() {
                    *s *= decay
                }
                let predicted = row.iter().zip(kk).map(|(s, k)| s * k).sum::<f32>();
                let delta = (v[h * d + j] - predicted) * beta[h];
                for (i, s) in row.iter_mut().enumerate() {
                    *s += kk[i] * delta
                }
                out[h * d + j] = row.iter().zip(qh).map(|(s, q)| s * q).sum::<f32>() * scale;
            }
        }
        Ok(out)
    }
    pub fn reset(&mut self) {
        self.data.fill(0.)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_gates() {
        assert_eq!(sigmoid(1000.), 1.);
        assert_eq!(sigmoid(-1000.), 0.);
        assert_eq!(softplus(1000.), 1000.);
        assert!(softplus(-1000.).is_finite());
    }
    #[test]
    fn norm_semantics() {
        let x = [3., 4.];
        let y = l2_norm(&x, 1e-12).unwrap();
        assert!((y[0] - 0.6).abs() < 1e-6);
        let z = rms_norm(&x, &[1., 1.], 1e-12).unwrap();
        assert!((z.iter().map(|a| a * a).sum::<f32>() / 2. - 1.).abs() < 1e-6);
    }
    #[test]
    fn convolution_tap_order_and_reset() {
        let mut s = ConvState::new(1).unwrap();
        let w = [1., 2., 3., 4.];
        for (x, raw) in [(1., 4.), (2., 11.), (3., 20.), (4., 30.)] {
            let y = s.step(&[x], &w).unwrap();
            assert!((y[0] - raw * sigmoid(raw)).abs() < 1e-6)
        }
        s.reset();
        assert_eq!(s.step(&[1.], &w).unwrap()[0], 4. * sigmoid(4.));
    }
    #[test]
    fn delta_known_rank_one_and_decay() {
        let mut s = DeltaState::new(2, 1, 1).unwrap();
        let a = s
            .step(&[1., 0.], &[1., 0.], &[2., 4.], &[0.], &[0.5])
            .unwrap();
        assert_eq!(s.data, vec![1., 0., 2., 0.]);
        assert!((a[0] - 1. / 2f32.sqrt()).abs() < 1e-6);
        s.step(&[1., 0.], &[1., 0.], &[0., 0.], &[0.5f32.ln()], &[0.])
            .unwrap();
        assert_eq!(s.data, vec![0.5, 0., 1., 0.]);
        s.reset();
        assert!(s.data.iter().all(|x| *x == 0.));
    }
    #[test]
    fn delta_head_mapping_and_isolation() {
        let mut s = DeltaState::new(2, 4, 2).unwrap();
        let a = s
            .step(
                &[1., 0., 0., 1.],
                &[1., 0., 0., 1.],
                &[1.; 8],
                &[0.; 4],
                &[1.; 4],
            )
            .unwrap();
        assert!(a.iter().all(|v| (*v - 1. / 2f32.sqrt()).abs() < 1e-6));
        assert_eq!(&s.data[0..4], &s.data[8..12]);
        assert_ne!(&s.data[0..4], &s.data[4..8]);
        let other = DeltaState::new(2, 4, 2).unwrap();
        assert!(other.data.iter().all(|x| *x == 0.));
    }
}
