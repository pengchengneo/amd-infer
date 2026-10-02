//! Deterministic top-k sampler for same-runtime A/B correctness tests.
pub fn sample(
    logits: &[f32],
    state: &mut u64,
    temperature: f64,
    top_k: usize,
) -> Result<u32, String> {
    if logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || !temperature.is_finite()
        || temperature <= 0.
        || top_k == 0
    {
        return Err("invalid sampler input".into());
    }
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_unstable_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(b.cmp(a)));
    ids.truncate(top_k.min(ids.len()));
    let max = logits[ids[0]] as f64;
    let weights: Vec<f64> = ids
        .iter()
        .map(|i| ((logits[*i] as f64 - max) / temperature).exp())
        .collect();
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    let uniform = (z >> 11) as f64 / 9007199254740992.;
    let mut threshold = uniform * weights.iter().sum::<f64>();
    for (id, weight) in ids.iter().zip(weights) {
        if threshold < weight {
            return Ok(*id as u32);
        }
        threshold -= weight;
    }
    Ok(*ids.last().unwrap() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_seed_softmax_distribution_and_top_k() {
        let mut state = 42;
        let mut count = 0;
        for _ in 0..16000 {
            count += usize::from(sample(&[0., 3f32.ln()], &mut state, 1., 2).unwrap() == 1);
        }
        assert!((count as f64 / 16000. - 0.75).abs() < 0.02);
        assert_eq!(sample(&[0., 100., -100.], &mut state, 0.7, 1).unwrap(), 1);
        assert!(sample(&[f32::NAN], &mut state, 1., 1).is_err());
    }
}
