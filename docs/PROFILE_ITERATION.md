# Profile-led FFN and device retention experiments

This is the local RX 9070 XT / gfx1201, ROCm 7.2.3, Rust 1.98.1 snapshot.
The same 14,252,845,984-byte mixed-format GGUF is used throughout (SHA256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`).
No whole-model megakernel or improvement over Vulkan is claimed.

## End-to-end gates and comparison scope

Raw prompt fixtures contain 5, 10 and 74 tokens, batch/concurrency one, context
128, eight fixed greedy outputs, FP32 activations/KV/state. Each listed result
has three trials with reference output IDs checked. Decode below uses the seven
wall-clock token intervals, not eight divided by seven-step time. Model loading,
tokenization and artifact export are outside request timing; reset and token
selection are inside compute E2E. Vulkan E2E additionally includes HTTP transport.
GGML internally quantizes matmul activations; identical weight bytes do not imply
identical arithmetic. Earlier native runs have one startup forward warmup; the
last two rows add one complete request warmup per case, matching the Vulkan case
warmup protocol. Shared-desktop clock/scheduling variation remains a limitation.

| Native variant | English decode/s | Chinese decode/s | Long decode/s |
|---|---:|---:|---:|
| FFN workspace reuse, before persistent prototype | 3.340 | 3.346 | 3.264 |
| Generic persistent 512-block gate/up | ~2.74 | ~2.73 | ~2.67 |
| Actual-format specialized persistent 2048-block | 3.629 | 3.626 | 3.533 |
| Plus resident FFN norm bank | 3.710 | 3.715 | 3.608 |
| Plus append-only device attention cache | 3.653 | 3.644 | 3.630 |
| Specialized reference, full case warmup | 3.657 | 3.654 | 3.562 |
| Four warp rows/block, full case warmup | 3.680 | 3.757 | 3.651 |
| Pinned same-weight Vulkan reference | 25.567 | 25.550 | 25.749 |

These are separate serial runs, not randomized/interleaved trials. Small gains
need confirmation. Generic persistent and block shuffle-reduction variants did
not consistently improve throughput; optional flags preserve the original path.
Device attention retention is a state-management building block, with no clear
overall speed benefit in these fixtures. `results/phase2/*-summary.json` holds
raw medians and numerical checks; launch JSONs record exact environment/arguments.

## What the profile establishes

The 64 FFNs store 8,556,380,160 packed bytes, including 4,971,724,800 IQ4_XS
bytes. Gate/up has sixteen actual format pairs, not one IQ4_XS layout. GGUF rows
are contiguous packed 256-value blocks; gate/up are 17408x5120 and down is
5120x17408. Selected FFN event windows yield roughly 40–105 GB/s logical packed
payload rates. This is not physical bandwidth-counter data.

A separate 256 MiB read/checksum diagnostic has one warmup and eight event-timed
reads: most are 370–435 GB/s, one is 72.8 GB/s. It is a diagnostic, not engine
throughput; scheduling/cache effects remain. It supports investigating decode
and execution dependencies instead of assuming an 80 GB/s HIP read ceiling.

Resource/ISA inspection uses the **actual linked library**, resolved from
evaluate RUNPATH, extracted with llvm-objcopy and clang-offload-bundler. Original
IQ4_XS GEMV uses 15 VGPRs; specialized fused gate/up uses 53 and warp-row uses 60.
Their recorded scratch/spill allocations are zero. More VGPRs suggest an
occupancy tradeoff but are not a measured occupancy result. IQ4 ISA shows a
dependent global float-codebook read. The optional
`AMD_INFER_INLINE_IQ4_LEVELS=1` candidate reconstructs the same sixteen integer
levels from constants; its ISA removes that table load, at 18 VGPRs for standalone
IQ4 and 59 for IQ4/IQ4 fused gate/up. Quantization oracle passes all 5,976 calls
with max error 5.692e-7; FFN oracle passes 81,920 values with max error 9.537e-7.
Full-model three-trial decode medians are 3.373/3.333/3.566 tok/s for its
control and 3.599/3.598/3.505 tok/s for the inline variant. All generated IDs
match; the long case regresses about 1.7%, so this remains optional rather than
a universal improvement. Runs are serial and shared-desktop variability applies.

## Multi-step state, sampled trajectories and actual memory

DeltaNet passes 128 uninterrupted steps / 786,432 values, max error 2.794e-9.
Append-only device attention passes two 128-step rounds with reset, bit-identical
to the existing GPU attention (which separately passes the scalar CPU oracle).
It rejects the 129th cache append. Reset changes the valid length to zero; stale
rows outside the valid range are never read and the next request overwrites them.

At seed 42, temperature 0.7, top-k 40, all three full-model 32-token sampled
trajectories match the saved correct Rust path. The long prompt reaches position
105; retained-device-state last logits differ by at most 1.373e-4. This is one
seed and bounded raw-text validation, not general quality or Vulkan sampler
equivalence. A transport-disconnected partial run was preserved separately and
excluded from successful results.

The retained-state sampled run is fully weight-resident. Exact requested bytes:

| Allocation category | Bytes |
|---|---:|
| Packed matrices | 13,333,954,560 |
| 48 DeltaNet state/input/output buffers | 154,159,104 |
| 64-row resident FFN norm bank | 1,310,720 |
| 16 attention cache/query/output buffers | 17,956,864 |
| GEMV/FFN workspace and reusable norm scratch | 1,083,392 |
| Live total | 13,508,464,640 |
| Requested peak, including temporary embedding/dequant buffers | 13,508,487,320 |

Minimum observed HIP free memory is 2,392,092,672 bytes. Requested bytes exclude
driver rounding; free-memory changes include other applications. Allocation
guards continue to leave 2 GiB plus rounding allowance; budgeted FFN streaming
fallback remains available. Full-resident results are not streaming-throughput
results. All GPU buffers and streams are locally owned; no other apps are stopped.

## Reproduction and preserved source

### GDN device pipeline and projection retention

The convolution history, recurrent state, L2 normalization, gating and output
normalization now have an optional persistent device buffer. Actual parameter
layers 0, 17, 48 and 62 pass 128 uninterrupted steps followed by 64 steps after
reset against scalar CPU convolution/L2/Delta/RMS/gating: 4,718,592 values,
maximum absolute error 1.122e-7. The four input projections initially still
return through the host; this version is `AMD_INFER_GPU_GDN_PIPELINE=1`.

Three full-request-warmed trials, identical raw prompts and eight greedy output
tokens, give these seven-interval decode wall-time medians:

| Case | Previous path tok/s | Device GDN tok/s |
|---|---:|---:|
| English | 3.64915 | 3.73610 |
| Chinese | 3.64541 | 3.73360 |
| Long, 74 prompt tokens | 3.55108 | 3.64684 |

All trials match the same-weight Vulkan token IDs. These are about 2.4–2.7%
improvements on a shared desktop, still far below the ~25.6 tok/s Vulkan
baseline. Results and recorded launch environments are in
`results/phase2/rust-gdn-{control,pipeline}-benchmark*`.

`AMD_INFER_GPU_GDN_PROJECTED=1` additionally retains all four input projections
and the output projection's input on the device. One 5120-float hidden vector
upload and one 5120-float final output download replace intermediate transfers.
Nine kernels execute in order on the default stream without intermediate host
synchronization; this is a device pipeline, not a single full-model megakernel.
Four real layers, 32 uninterrupted steps and another 32 after reset, pass
1,310,720 final values with zero difference against separate GEMVs plus the
validated device GDN pipeline. Full-model A/B and 32-token sampling validation
are recorded separately; do not infer full-model performance from that test.
Profile event slot 25 measures the entire projected GDN chain, not GEMV alone.
The full-model gate subsequently passed three warmed trials for all fixtures:

| Case | Pipeline control decode/s | Projected decode/s | Control E2E s | Projected E2E s | Projected prefill s |
|---|---:|---:|---:|---:|---:|
| English | 3.72340 | 4.97413 | 3.18638 | 2.37768 | 0.94872 |
| Chinese | 3.72564 | 4.96598 | 4.48083 | 3.33625 | 1.90441 |
| Long | 3.62503 | 4.82733 | 21.26953 | 15.67478 | 14.20346 |

Every eight-token output matches Vulkan across all three trials. Reset prefix
logits repeat bit-for-bit; maximum prefix differences against the earlier FP32
path are 4.768e-6 / 9.418e-6 / 2.480e-5. These remain raw-text, context-128,
single-concurrency tests. The ~33% decode gain does not close the ~25.6 tok/s
Vulkan gap. Source, oracle log and build log are saved as
`projected-gdn-source.zip`, `projected-gdn-oracle.log`, `projected-build.log`;
full results and arguments are `rust-projected-{control,gdn}-benchmark*`.
The 48 GDN buffers request 172,308,480 bytes versus 154,159,104 for the former
Delta-only buffers; projected execution reuses these buffers and the workspace.

The projected version plus device attention retention subsequently passes the
same three 32-token sampled trajectories (seed 42, temperature 0.7, top-k 40)
against the saved correct FP32 runtime. All 96 sampled IDs match; final logit
maximum errors are 5.484e-6 / 4.530e-6 / 6.437e-5, including the long sequence
through position 105. This checks one seed, not broad generation quality.
`results/phase2/sampled-gdn-projected/summary.json` preserves exact results.

With the attention cache enabled, live requested bytes are 13,526,614,016 and
peak is 13,526,636,696, including temporary embedding buffers. Packed resident
weights remain 13,333,954,560 bytes. Minimum HIP free observed across the three
cases is 2,412,462,080 bytes; requested bytes exclude driver rounding, and free
memory includes other desktop applications. This run is fully resident, not a
partial-residency throughput result. The conservative allocation guard remains
enabled; no other application was stopped.

One English five-token/8-output request with HIP events covers 12 model forwards
and eight LM-head projections. FFN chains total 1.026955 s (768 calls), projected
GDN chains 0.507343 s (576 calls), other GEMV events 0.246881 s; FFN is ~57.7%
of these measured windows. Request wall time with instrumentation is 2.486610 s.
Attention, host operations, copies, launch/synchronization costs are not all
included in those event totals. The events change execution timing; the 4.757
tok/s diagnostic is excluded from headline throughput. This evidence selects
FFN quantized decoding/layout as the next optimization target. Actual-format
FFN logical packed bytes divided by these event windows gives about 100 GB/s,
not a physical-bandwidth counter. Results are in `projected-profile/`.

### Confirmed FFN row layout on the projected GDN path

An initial serial layout A/B saw roughly 8% cross-run short-case control drift.
The follow-up keeps one loaded model, warms both complete request layouts, then
runs six balanced ABBAAB trials per case (three per layout). This confirms the
four-warp-rows, 1024-block FFN variant versus the 2048-block reference:

| Case | Control decode/s | Warp-row decode/s | Gain | Warp-row prefill s | Warp-row E2E s |
|---|---:|---:|---:|---:|---:|
| English | 4.96965 | 5.18666 | 4.37% | 0.90651 | 2.27739 |
| Chinese | 4.95565 | 5.17211 | 4.37% | 1.81109 | 3.18281 |
| Long | 4.80974 | 5.02003 | 4.37% | 13.61979 | 15.03597 |

Every eight-token output matches the previously Vulkan-validated output. Within
each layout, all reset prefix logits repeat exactly; between layouts maximum
prefix errors are 4.768e-6 / 8.583e-6 / 2.480e-5. These are medians over only
three interleaved trials, not a confidence interval or exclusive-GPU result.
`interleaved-projected-ffn/summary.json` includes all rates, phase timings and
checks. `evaluate` accepts a validated six-entry `AMD_INFER_FFN_LAYOUT_SEQUENCE`
only for explicit device-FFN tests, preserving the normal one-to-three-repeat
runner behavior. Every trial records its layout and blocks outside E2E timing.

The final projected + warp-row + device-attention-cache combination also passes
all 96 sampled IDs against the saved correct path with seed 42 / temperature
0.7 / top-k 40. Last-logit maximum errors are 4.292e-6 / 1.526e-5 / 1.011e-4;
the longest trajectory covers 105 positions. Peak requested device bytes remain
13,526,636,696 and lowest HIP free memory is 2,404,597,760 bytes across those
cases. This is full packed-weight residency with the same reserve guard.
Here full residency means all main-graph GEMV matrices are resident. Input
embedding rows are still read on the host, dequantized by HIP, and returned as
host hidden vectors; global norms, residuals
and attention Q/K preparation remain part of the hybrid graph. MTP weights are
excluded from this 64-layer text path. This is not all-tensor GPU residency.
Results: `sampled-projected-rows/summary.json`. The greedy performance table
uses host attention for consistency with its control; the sampled state check
additionally enables append-only device attention, which has separate oracle
coverage. Do not mix the sampled run's timing with the performance table.

The latest warp-row event diagnostic confirms FFN chains at 0.923834 s versus
1.026955 s (~10.0% reduction), while GDN is 0.507614 s and other measured GEMV
windows total 0.247188 s. FFN still contributes 55.0% of those windows; GDN
contributes 30.2%. Its instrumented E2E is 2.394629 s and is excluded from the
un-instrumented interleaved performance table. This points to separating the
FFN gate/up and down event windows, then targeting actual IQ4_XS decode and row
layouts while retaining the FP32 oracle. The complete hidden-state device graph
also remains unfinished; CPU residual/norm and full-attention preparation still
introduce transfers and synchronization. `projected-row-profile/summary.json`
preserves call counts, memory and the diagnostic's exact launch environment.

```sh
python3 tools/interleave-ffn-layouts.py MODEL_GGUF TOKENIZER_JSON
python3 tools/check-sampled-ab.py MODEL_GGUF TOKENIZER_JSON --projected-rows
python3 tools/profile-projected-gdn.py MODEL_GGUF TOKENIZER_JSON --warp-rows
```

```sh
AMD_INFER_GPU_EXCLUSIVE=1 target/release/check-gdn MODEL_GGUF
AMD_INFER_GPU_EXCLUSIVE=1 target/release/check-gdn-projected MODEL_GGUF
python3 tools/run-row-layout-ab.py MODEL_GGUF TOKENIZER_JSON --gdn
python3 tools/run-row-layout-ab.py MODEL_GGUF TOKENIZER_JSON --projected
python3 tools/check-sampled-ab.py MODEL_GGUF TOKENIZER_JSON --projected
```

With the existing WSL ROCm environment sourced and the binary built:

```sh
python3 tools/profile-ffn.py MODEL_GGUF
python3 tools/run-device-retention.py MODEL_GGUF TOKENIZER_JSON
python3 tools/run-row-layout-ab.py MODEL_GGUF TOKENIZER_JSON
python3 tools/run-row-layout-ab.py MODEL_GGUF TOKENIZER_JSON --inline
python3 tools/check-sampled-ab.py MODEL_GGUF TOKENIZER_JSON
python3 tools/check-sampled-ab.py MODEL_GGUF TOKENIZER_JSON --retention
AMD_INFER_GPU_EXCLUSIVE=1 AMD_INFER_DELTA_NO_RESET=1 target/release/amd-infer check-delta
AMD_INFER_GPU_EXCLUSIVE=1 target/release/check-ops
AMD_INFER_GPU_EXCLUSIVE=1 target/release/amd-infer diagnose-read-bandwidth
```

Use fresh results paths for new experiments; the supplied named scripts reuse
their explicitly documented paths. Coordinate GPU use first. Do not rebuild
the mapped HIP library while an evaluate process runs. Model/tokenizer absolute
paths are saved in launch JSONs. Source ZIPs plus per-file manifests, executable
and actual library hashes, raw logs, code objects and ISA are retained under
`results/phase2`. No GitHub publication has occurred.
