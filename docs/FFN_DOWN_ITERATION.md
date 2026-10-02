# Format-specialized FFN down iteration

This stage preserves the previous projected-GDN / warp-row FFN path and adds
`AMD_INFER_DOWN_VARIANT=1`: fixed-shape, format-specialized HIP down GEMV
(17408 input columns, 5120 output rows). Variant 0 preserves the generic path.
There is no cross-block barrier or full-model persistent megakernel.

## End-to-end evidence

Same IQ4_XS GGUF, B=1, raw prompts, 8 fixed greedy outputs, context cap 128.
Both complete requests warmed; one loaded model, balanced ABBAAB sequence,
three trials per variant. Decode rate counts seven intervals after the first
output. Prefill excludes LM head; E2E includes prefill, LM heads and decoding,
excluding model load and warmup. Main matrices were fully resident in this run;
these numbers do not describe streaming fallback performance.

| Prompt tokens | Control tok/s | Candidate tok/s | Gain | Candidate prefill s | Candidate E2E s |
|---|---:|---:|---:|---:|---:|
| English 5 | 5.113745 | 5.632564 | 10.15% | 0.832074 | 2.097172 |
| Chinese 10 | 5.182996 | 5.710997 | 10.19% | 1.629938 | 2.879586 |
| Long fixture 74 | 5.019828 | 5.521826 | 10.00% | 12.295490 | 13.585332 |

Evidence: `results/phase2/interleaved-specialized-down/{launch.json,run.log,summary.json}`.
Prefix logits are byte-identical across variants; generated IDs match the saved
same-weight baseline. Vulkan remains approximately 25.6 tok/s on these fixtures;
its scale/reduction ordering and precision differ from this path. Earlier
activation-quantization attribution was not established for the Vulkan baseline;
see the phase4 shader inspection in `GPU_CONTEXT_EXECUTION.md`.
The remaining performance gap is substantial. No CPU-offloaded vLLM victory claim.

Reproduce after coordinating GPU use, in the configured WSL HIP/Rust environment:

```sh
cargo build --release --features hip,tokenizer --bins
python3 tools/interleave-ffn-layouts.py "$MODEL_GGUF" "$TOKENIZER_JSON" --down
python3 tools/profile-projected-gdn.py "$MODEL_GGUF" "$TOKENIZER_JSON" --warp-rows --down
python3 tools/check-context-boundary.py "$MODEL_GGUF" "$TOKENIZER_JSON" --down
```

Last command intentionally reports a strict numeric-gate failure below; retain
all trajectories for diagnosis. Logs and launch JSON specify actual arguments
and environment. Performance runs disable per-step logit capture and profiling.

## Correctness and limitation

All 64 actual FFN layers, four inputs each: 1,310,720 output values compared with
the independently checked matrix / CPU activation oracle, max abs 3.8146973e-6.
An initial IQ4_XS dispatch error produced NaN, was caught by this gate and fixed
before performance testing; its failure log is preserved.

Expanded padded English, code and Unicode fixtures exercise 96+32, 64+64 and
64+64 prompt/output tokens respectively: 160 sampled outputs, 127 processed
positions per fixture, two seeds. All 39,731,200 step logits are byte-identical
between the new down kernel and the previous best path; sampled IDs match.
This is bounded-context coverage, not long-context qualification.

Against the older hybrid scalar reference, IDs still match but the unchanged
1e-3 absolute gate fails at code generation step 47:

| Case | Maximum absolute logit error | Maximum relative L2 error | Gate |
|---|---:|---:|---|
| English | 0.000642776489 | 2.1672e-5 | pass |
| Code | 0.001778602600 | 5.5650e-5 | fail at step 47 |
| Unicode | 0.000498175621 | 1.1337e-5 | pass |

Evidence: `results/phase2/context-boundary-down/candidate-error-analysis.json`,
`previous-best-summary.json`, `previous-best-summary32.json` and retained
per-step logits. The failure also exists with down variant 0. Turning off the
GPU attention cache does not remove it. Reference-order and scalar-libm
experiments failed to satisfy the gate and were rolled back; their source ZIPs
and logs remain under `results/phase2`. No threshold was relaxed.

## Profile and memory

HIP events cover 12 forward calls (warmup plus measured request), not isolated
decode. Generic down took 0.446261 s; specialized down 0.229247 s. Final FFN
parts: norm 0.011763 s, gate/up/activation 0.491835 s, down 0.229247 s.
Whole FFN 0.732846 s = 49.56% of measured GPU events; projected GDN
0.502124 s = 33.96%. Parts overlap whole-FFN events and are excluded from totals.
Instrumented wall rates are diagnostic and are not the performance table.

Requested allocation peak 13,526,636,696 bytes; resident main matrix storage
13,333,954,560 bytes; observed minimum free 2,405,871,616 bytes in final profile.
Requested bytes exclude driver rounding; observed free includes desktop users.
Embedding rows are read on CPU, dequantized by HIP and returned to host hidden
vectors. Norm/residual and attention preparation retain hybrid CPU operations.
Safety reserve remains enforced; no other applications were stopped.

Compiled GPU `.text` before precision experiments and after rollback has the same
SHA256: `38194d6eb95f807295c94e9d1d5e50144734c64b79086a417026eb9395d7b7a3`.
Compiler metadata shows no spills in specialized down kernels; this is not a
measured occupancy or physical memory-bandwidth claim.

Next priority: capture per-layer hidden/state deltas around code step 47 to find
the first divergent operation against the scalar reference. For performance,
gate/up/activation is now the largest FFN component. GPU retention of residual,
normalization and attention preparation should follow correct state diagnostics,
with the same interleaved E2E and trajectory checks.
