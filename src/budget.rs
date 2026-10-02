//! First-order state budget for the confirmed Qwen3.8-27B text configuration.
//! Does not predict prefill activation peaks or allocator fragmentation.
use crate::gguf::{Gguf, Value};
pub fn state_bytes(batch: u64, tokens: u64, q8: bool) -> (u64, u64) {
    // 16 full-attention layers, K+V, 4 KV heads, dimension 256.
    let kv_values = batch * tokens * 16 * 2 * 4 * 256;
    let kv = if q8 {
        kv_values / 32 * 34
    } else {
        kv_values * 2
    };
    // 48 DeltaNet layers, 48 heads, 128x128 FP32 state + 3 convolution taps.
    let recurrent = batch * 48 * (48 * 128 * 128 * 4 + 3 * 10240 * 4);
    (kv, recurrent)
}
pub fn report(g: &Gguf) -> Result<(), Box<dyn std::error::Error>> {
    if g.metadata.get("general.architecture") != Some(&Value::Text("qwen35".into()))
        || g.metadata.get("qwen35.block_count").and_then(Value::uint) != Some(65)
    {
        return Err("budget requires verified 64-layer Qwen3.8 plus one MTP block".into());
    }
    let mut all = 0;
    let mut mtp = 0;
    let mut embedding = 0;
    for t in &g.tensors {
        let n = t.bytes()?;
        all += n;
        if t.name.starts_with("blk.64.") {
            mtp += n
        }
        if t.name == "token_embd.weight" {
            embedding = n
        }
    }
    let weights = all - mtp - embedding;
    println!(
        "tensor_bytes={} mtp_bytes={} cpu_embedding_bytes={} gpu_core_bytes={}",
        all, mtp, embedding, weights
    );
    println!("Budget assumption: embedding on CPU; MTP disabled; Q8_0 KV; FP32 recurrent state.");
    println!("batch context weights_MiB KV_MiB recurrent_MiB subtotal_MiB");
    for tokens in [4096, 8192] {
        for batch in [1, 2, 4, 8] {
            let (kv, state) = state_bytes(batch, tokens, true);
            println!(
                "{} {} {:.2} {:.2} {:.2} {:.2}",
                batch,
                tokens,
                weights as f64 / 1048576.,
                kv as f64 / 1048576.,
                state as f64 / 1048576.,
                (weights + kv + state) as f64 / 1048576.
            );
        }
    }
    println!("Add measured prefill workspace, allocator overhead and current display/driver use before admission; these are not included.");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_state_matches_log() {
        let (kv, rs) = state_bytes(1, 4096, true);
        assert_eq!(kv, 136 * 1048576);
        assert_eq!(rs, 149 * 1048576 + 640 * 1024);
    }
    #[test]
    fn bf16_and_q8_capacity() {
        let (kv, _) = state_bytes(1, 4096, false);
        assert_eq!(kv, 256 * 1048576);
        let (kv, rs) = state_bytes(8, 8192, true);
        assert_eq!(kv, 2176 * 1048576);
        assert_eq!(rs, 1197 * 1048576);
    }
}
