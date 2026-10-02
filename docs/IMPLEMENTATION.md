# Rust/HIP implementation checkpoints

Target: RX 9070 XT (gfx1201), Qwen3.8-27B text-only, Rust host runtime,
HIP kernels, incremental decode fusion. The historical C++/Python host proposal
does not define this implementation.

## Isolation and provenance

Independent source checkout. Initial upstream AMDInfer HEAD: `4113a64`.
Existing reference deployments are external to this repository. Coordinate GPU
tests with their owners and other workloads; do not modify those deployments or
terminate unrelated processes. The experiment runtime and model paths are local
configuration, not repository defaults.

Model file: `Qwen3.8-27B-UD-IQ4_XS.gguf`, 14,252,845,984 bytes.
SHA256 from verified original deployment:
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
GGUF tensor inventory is read directly from that file, not inferred from its name.
Reference GGML is pinned to llama.cpp b11284:
`25747b08e7a0f9a59a2089ce6b98d2229b76042a`.

## Capacity: measured versus derived

Measured original llama.cpp log: GPU model 12,726.34 MiB; KV 136 MiB;
recurrent 149.62 MiB; compute 34.27 MiB; device 16,304 MiB.
Startup free memory was 15,416 MiB in that run, not a permanent guarantee.
Original vLLM later reported about 13.5 GiB free, demonstrating changing
display/driver occupancy and differences between runtimes.

Directly parsed payload: 13,582.09 MiB; MTP block 64: 334.75 MiB;
CPU embedding: 521.00 MiB; remaining GPU weights: 12,726.34 MiB.
The exact match explains the reference log's weight residency.
First version excludes MTP/vision and fetches only the current embedding row.

Derived from actual text configuration: 64 layers, 48 DeltaNet + 16 attention;
KV heads 4 x 256; recurrent heads 48 x 128 x 128 FP32, convolution 3 x 10240
FP32 values per DeltaNet layer. Q8_0 KV includes 34 bytes per 32 values.

| Active requests | Per-request capacity | Q8 KV MiB | State MiB | Core + caches MiB |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 136 | 149.62 | 13011.97 |
| 2 | 4096 | 272 | 299.25 | 13297.59 |
| 4 | 4096 | 544 | 598.50 | 13868.84 |
| 8 | 4096 | 1088 | 1197 | 15011.34 |
| 1 | 8192 | 272 | 149.62 | 13147.97 |
| 2 | 8192 | 544 | 299.25 | 13569.59 |
| 4 | 8192 | 1088 | 598.50 | 14412.84 |
| 8 | 8192 | 2176 | 1197 | 16099.34 |

Estimates exclude workspace, peak prefill activations, fragmentation, HIP state,
display use, and admission reserve. 8 x 8K is not a viable guaranteed full-GPU
configuration with this weight format. Admission must use current free memory
and a measured high-watermark. BF16 KV is 256 MiB/request at 4K (512 at 8K),
so the initial BF16 correctness path must use short sequences and batch one.

## Quantization compatibility

866 tensors use 13 types. IQ4_XS: 211 tensors / 6882.34 MiB. Other types:
F32, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, IQ4_NL,
IQ3_S, IQ2_S. A uniform INT4 kernel cannot consume this artifact.
Preserve packed bytes and GGUF alignment/row ordering; reject unknown types.
Do not requantize for a speed comparison against the existing baselines.
IQ4 uses a non-linear codebook and split signed subblock scales, not uniform
INT4 multiply. The compatibility GPU decoder retains GGML MIT attribution;
its serial per-tile decoder is a correctness fallback, not a speed optimization.

## Runtime/kernel boundary

Rust owns validated GGUF descriptors, device-buffer lifetimes, tensor/state
indices, admission, scheduling, execution plans and output handling.
Narrow C ABI owns HIP device compilation, launches, copies, synchronization and
error translation. Build gfx1201 kernels with hipcc; no Python runtime dependency.
CPU-only GGUF inspection builds without ROCm. GPU buffers hold compressed
weights; tiny per-tile dequantization uses LDS, never a second full BF16 model.

Current implementation: GGUF v3 parser, format inventory, budget calculator,
Rust IQ4_XS CPU decoder, independent upstream oracle, Rust/HIP RAII boundary,
optimized-layout IQ4 linear prototype and mixed-format compatibility linear.
No tokenizer, complete Qwen execution, scheduler or megakernel is complete yet.

## First correct model path, then fusion

1. Validate quant bytes against pinned GGML; test actual matrix rows at batches
   1/2/4/8 plus adversarial nibble/scales and random inputs.
2. Validate RMSNorm, Q/K norm and partial RoPE, attention output gate, convolution
   state, DeltaNet state transition, SwiGLU and residual in isolation.
3. Teacher-forced short token sequences: compare intermediate tensors and logits
   to the same quantized GGML model. Compare recurrence across multi-token
   prefill and one-token decode, reset/cancel and slot reuse.
4. B=1 short-text end-to-end generation with matched tokenizer, template,
   thinking controls and greedy settings; record logit error and top-k agreement.
5. Add fixed batches, then continuous admission and chunked prefill.
6. Fuse adjacent decode operations and evaluate persistent/megakernel launch
   only after state correctness and device synchronization/occupancy are measured.
   The ordinary multi-kernel path remains an oracle and a fallback.

## Fair baseline gaps

Existing Windows single-request short-input medians: llama.cpp Vulkan 21.22
tok/s, Ollama HIP 23.70 tok/s, three 256-token outputs. Both GGUF stacks use
GGML; microbatch differs (128 vs 512), and server timing accounting differs.
They establish deployment functionality, not a Rust-engine or architecture win.

Before speed claims: pin same SHA/model revision, tokenizer/template/thinking,
KV/state precision, context and actual prompt token counts, output length,
sampling/seed, cache hit policy, warmup, offload and backend versions.
Measure B=1/2/4/8; inputs 1K/4K/7K; outputs 256/1024; 8K counts input+output.
Capture TTFT, per-token intervals, aggregate output tok/s, wall time, admission
queue time, peak VRAM and quality. Run engines serially with GPU exclusive use.
Separate fixed-batch decode from online service and prefill-inclusive throughput.
An offloaded vLLM run is a distinct configuration, not the full-GPU reference.

## Verified so far

Rust CPU tests: fourteen passed (GGUF truncation/overlap/type/row bounds,
FP16 edges, nibble/scale order, state budget,
normalization semantics, stable gates, convolution tap order/reset and DeltaNet
rank-one update/decay, head mapping and state isolation).
Actual IQ4_XS CPU comparison: 211 tensors, 633 sampled rows, 4,789,248 values;
maximum absolute error zero against untouched b11284 dequantization.
Mixed-format HIP fallback compiled for gfx1201: twelve packed matrix formats,
1,494 sampled rows and 5,976 B=1/2/4/8 calls passed against independent CPU
dequantization plus f64 dot products; maximum absolute error 5.69e-7.
Native Rust tokenizer IDs and UTF-8 round trips match GGML in six cases.
The 64-layer Rust host/HIP matrix streaming graph completed a one-token forward:
248,320 logits, top-1 agreement, same top-ten set, cosine 0.999740 and relative
RMSE 2.32% against CPU GGML. This comparison has different activation arithmetic;
it is evidence, not a strict identical-arithmetic correctness pass. Multi-token
state verification also completed for the two-token `Hello there` fixture:
top-1 and top-ten set agree, cosine 0.999767, relative RMSE 2.27%. These two short
fixtures do not establish general generation quality. Attention and recurrent state currently
execute on the CPU; matrices stream in 256-row tiles. There is no resident-weight
throughput or megakernel claim.

Packed matrix residency now uses one shared device allocation: 13,333,954,560
bytes. The earlier per-matrix allocation strategy was rejected by the 2GiB
reserve guard near the end of loading and cleanly released all allocations.
With shared storage the two-token forward completed; all 248,320 logits are
bit-identical to the streaming path (maximum absolute difference zero). During
the shared-storage test HIP reported 2,585,382,912 free bytes. This is a snapshot,
not a reservation. Generation and state-kernel optimization remain separate gates.

Native Rust greedy generation for raw `Hello` produced `[11,353]` (`, I`), matching
the CPU reference's first two greedy choices. Last-step logits have cosine
0.999795 and relative RMSE 2.16%. This is a short integration fixture, not a
conversation-quality test. A fused HIP DeltaNet kernel (state decay, prediction,
rank-one update, query projection) passed eight stateful steps / 49,152 output
values against the Rust scalar formula, max error 1.86e-9. Full two-token model
integration completed across all 48 DeltaNet layers: logits max error 4.29e-6,
relative RMSE 2.26e-7, top-ten order identical to resident-matrix/CPU-state mode.
`AMD_INFER_GPU_DELTA=1` selects it;
convolution, normalization and full attention still execute on the host.

Original vLLM subsequently completed arithmetic and Chinese generation:
0.59 token/s, B=1/4K, substantial CPU weight residency. Its API remains running
at port 8000 until the original task explicitly completed and idle metrics were
verified. It was then stopped with SIGTERM under the user's test coordination
authorization; restoration scripts, environment, command and metrics are saved
in `results/reference-service-snapshot`. Source patches and exact result remain
in the original deployment's `outputs/vllm-setup-status.json`.

## Phase 2: budget fallback and reproducible comparison

The entries above describe earlier milestones. The current runtime has optional
HIP DeltaNet, normalization and attention, reusable GEMV workspace, parallel
mixed-format decoders, and a budgeted partial-residency fallback. Fallback removes
whole late-layer FFN triples from residency and streams those matrices in
4096-row GPU tiles. It does not stop other desktop applications. Allocation checks
preserve a 2 GiB reserve, with an additional 16 MiB rounding allowance per check;
the budget planner also leaves 256 MiB for runtime allocations. These checks do
not reserve the GPU against concurrent applications.

The first actual partial-residency validation completed the Chinese 10-token
prompt and eight generated tokens with 12,863,242,240 resident weight bytes,
13,018,434,560 requested peak bytes and minimum observed HIP free memory
2,344,108,032 bytes. Its original 256-row streaming tiles were slow (0.212 token/s);
the runtime now uses larger tiles. Later tests fitted all 13,333,954,560 packed
weight bytes, so their results are **full resident**, even though fallback was
enabled. Requested allocation counters exclude driver rounding; observed free
memory includes driver allocations and other desktop applications.

`results/phase2/benchmark-summary.json` records three trials per raw prompt:
5 English, 10 Chinese and 74 long-prompt tokens, followed by exactly eight greedy
tokens. All output IDs match the pinned llama.cpp b11284 Vulkan reference in all
trials; repeated reset/prefill logits are bit-identical. Median Rust decode rates
are 2.335, 2.330 and 2.473 token/s; Vulkan rates are 25.567, 25.550 and 25.749.
This is a remaining performance gap, not a speedup claim. Both use identical
weight bytes, input IDs, batch one, context 128, FP32 KV, microbatch one and no
prefix cache. GGML internally quantizes matmul activations, so arithmetic differs.
Rust uses HIP GEMV/DeltaNet with host attention, convolution and scalar operations
in that benchmark. Native E2E in this older snapshot includes validation-logit
export and is not directly equivalent to HTTP E2E. Subsequent evaluate runs move
artifact export outside timing. Warmup also differs: Rust one full forward per
mode; Vulkan startup warmup plus one complete request per case. Shared desktop
and only eight output tokens limit statistical conclusions.

CPU-reference prefix-logit relative RMSE is 2.35%, 3.34%, 4.69% for these prompts,
with matching top-1. English and Chinese eight-token CPU-reference greedy outputs
also match. Optional GPU attention/norm matches the host path within 1.383e-5
on the long fixture but did not improve speed in the measured configuration.

ROCm profiler tracing is unavailable in this WSL topology (no KFD sysfs nodes);
the runtime instead supports optional HIP-event GEMV timing. The recorded profile
identified serial IQ3_S dequantization as the largest kernel contributor, followed
by IQ3_XXS and Q3_K. Parallel IQ decoders passed the complete 5,976-call matrix
oracle before the benchmark above. Q2_K and Q3_K still use serial block decoding.

The next device pipeline keeps FFN RMS, gate/up outputs, SwiGLU and down input on
the GPU, selected by `AMD_INFER_GPU_FFN=1` with resident FFN triples and shared
workspace. Streaming triples use the existing fallback. `check-ffn` passed four
actual layers times four input patterns (81,920 values), maximum error 9.537e-7
against CPU scalar RMS/SwiGLU around independently validated packed GEMVs.
Full-model regression and fresh throughput measurements are separate gates.
This pipeline launches several kernels; it is not a persistent megakernel.

Those FFN gates subsequently passed: all three full-model fixtures preserve the
eight generated IDs; maximum prefix-logit error versus the prior FP32 path is
6.437e-6, 5.960e-6 and 2.289e-5. Requested peak allocation is 13,489,199,256 bytes;
minimum observed HIP free memory is 2,423,914,496 bytes in the sampled run. This
test fitted every packed matrix, so it does not demonstrate streaming throughput.
`rust-ffn-benchmark-summary.json` contains three unsampled trials, with median
decode 2.681, 2.666, 2.635 token/s and prefill 1.810, 3.634, 27.045 seconds. All
trials retain reference output IDs and bit-identical reset/prefill repetition.
The launch parameters, executable/library SHA256s and raw per-request results
are retained beside the summary. FFN improves measured decode by about 7–15%
against the earlier native snapshot; Vulkan remains roughly ten times faster.
Q2_K/Q3_K element-parallel decoding is the next separately validated change.

Q2_K/Q3_K parallel decoding subsequently passed the same 5,976-call oracle
(maximum error 5.692e-7). The three-trial full-model result is retained in
`results/phase2/rust-qk-benchmark-summary.json`: median decode intervals are
3.242, 3.235 and 3.167 token/s, compared with Vulkan 25.567, 25.550, 25.749.
All eight generated IDs match the reference in every trial; prefix logits stay
bit-identical across resets and their differences from the earlier FP32 snapshot
are unchanged from the FFN-only change. Prefill medians are 1.493, 2.998 and
22.351 seconds. This is another approximately 20–21% native improvement, with a
remaining approximately eightfold decode gap. Matrix-call host wall time still
accounts for roughly 90% of a short request, including transfers and synchronizes.

Reproduce the native fixture run after coordinating GPU use and sourcing the
existing ROCm environment, from this repository in WSL:

```sh
AMD_INFER_GPU_EXCLUSIVE=1 AMD_INFER_GPU_DELTA=1 \
AMD_INFER_REUSE_WORKSPACE=1 AMD_INFER_ENFORCE_RESERVE=1 \
AMD_INFER_PARTIAL_RESIDENCY=1 AMD_INFER_GPU_FFN=1 \
AMD_INFER_EVAL_MODES=0 AMD_INFER_EVAL_REPEATS=3 \
target/release/evaluate MODEL_GGUF TOKENIZER_JSON \
results/phase2/cases results/phase2/rust-qk-benchmark
python3 tools/summarize-benchmark.py rust-qk-benchmark
```

Use a fresh results directory for a new experiment. `MODEL_GGUF` and
`TOKENIZER_JSON` above are placeholders for the exact paths recorded in
`qk-benchmark-launch.json`; preserve the recorded model SHA256. Do not rebuild
the HIP shared library while an existing process maps it. Optional
`AMD_INFER_PROFILE_KERNEL=1` changes timing and should be a separate profiling
run. Event slot 24 represents the entire resident FFN chain (RMS, gate/up,
SwiGLU, down); slots 0–23 represent ordinary standalone GEMV formats. FFN
projections are not double-counted as standalone GEMVs in that mode.

Latest measured allocation optimization: `AMD_INFER_REUSE_FFN_NORM=1` reuses
a 20,480-byte workspace buffer instead of allocating/freeing a device norm
buffer for each FFN invocation. Upload remains per layer; this is buffer reuse,
not permanently device-resident normalization weights. Independent FFN oracle
again passes all 81,920 values with max error 9.537e-7. Nine full-model requests
pass all output-ID and reset-repetition gates, with unchanged prefix-logit errors.
`rust-ffn-reuse-benchmark-summary.json` records decode interval medians 3.340,
3.346 and 3.264 token/s, prefill 1.443, 2.883 and 21.587 seconds. The incremental
gain is approximately 3%, not a major new throughput result. Requested peak is
13,489,219,736 bytes; free-memory sampling is deliberately disabled here, so no
new minimum-free measurement is claimed. Reproduce with the command above plus
`AMD_INFER_REUSE_FFN_NORM=1`, a fresh output directory and its corresponding
summarize command. Binary/library hashes and launch parameters are saved.

In the preceding separately timed profiling run, standalone GEMV plus FFN event
windows total 1.976 seconds, including FFN 1.278 seconds; matrix host wall is
3.436 seconds and request wall is 3.828 seconds. Event windows can include stream
scheduling gaps, so subtracting them does not isolate transfer-only cost. FFN
still dominates device work, while transfers/allocation/host launch/sync remain
material. The next gates should tune FFN GEMV bandwidth and retain more hidden
state/normalization data on-device before attempting persistent dispatch. The
current graph is a complete 64-layer hybrid runtime, not a complete megakernel.
