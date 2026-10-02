# Phase6: wave-uniform quantized gate/up

The numerical gates pass. The candidate is opt-in and the phase5 runtime and
source snapshot remain the rollback baseline. No upstream publication is made.

## Code and arithmetic

`AMD_INFER_UNIFORM_ROW=1` selects the `UniformRow=true` specialization of
`scalar_warp_gate_up` in `kernels/iq4_xs.hip`. On gfx1201, each wave has 32 lanes;
its row index is already equal across the wave. `__builtin_amdgcn_readfirstlane`
expresses that equality to the compiler, allowing uniform address and metadata
work to use scalar registers. Quantization layouts, packed weights, FMA order
and the existing reduction tree are unchanged. The flag is off by default;
down projections and GDN projections are unchanged.

This is an independently written compiler-uniformity optimization, not a port
of an upstream packed GEMV. Static ISA and resource metadata are retained under
`results/phase6/uniform-kernel-code` and `uniform-kernel-analysis.log`.

The inspected b11284 upstream `mul_mat_vec_iq4_xs.comp` handles eight threads
per256-weight block, reads packed32 words and floating activation vectors, and
hoists the six-bit subblock scale outside its register accumulation. This could
reduce repeated scale/decode work, but changes rounding compared with the current
per-weight FMA tree. The current candidate preserves that tree. Neither source
inspection nor our benchmark establishes which shader dispatch the retained
Vulkan binary actually chose; no activation-Q8 claim is made.

| Gate/up formats (GGML IDs) | VGPR control / candidate | Estimated active blocks control / candidate |
|---|---:|---:|
| IQ4_XS / IQ4_XS (23/23) | 88 / 50 | 14 / 16 |
| IQ3_S / IQ4_XS (21/23) | 103 / 71 | 14 / 16 |
| Q5_K / Q5_K (13/13) | 118 / 72 | 11 / 16 |

HIP reports 2048 maximum threads per multiprocessor; these 128-thread block
estimates correspond to 87.5% -> 100%, and 68.75% -> 100%, respectively.
They are resource estimates, not measured dynamic occupancy. Both specializations
have no reported scratch or register spills. For 23/23, SGPR usage rises from
47 to 83. Static global-load instruction counts do not fall; sixteen vector
integer-to-float conversions become scalar conversions. This does not establish
dynamic decode cycles or reduced physical memory traffic.

`rocprofv3-avail` enumerates no agents/counters in this WSL environment because
`/sys/class/kfd/kfd/topology/nodes` is absent. The exact logs are retained in
`profiler-agents.log` and `profiler-counters.log`. No driver or security changes
were attempted. Actual occupancy, cache misses, physical bandwidth and clocks
are unavailable here.

## Format and shape evidence

`gemv-baseline-summary.json` records 38 distinct real matrix format/shape groups
and sixteen FFN gate/up format pairs. The independent CPU GGML dequant oracle,
five input patterns and selected rows all pass with maximum absolute error zero.
Large matrices are uploaded in bounded 16 MiB chunks; the 512 MiB read safety
limit and GPU reserve are retained. The initial oversized-read failure is saved.

All 64 candidate FFNs pass the existing independent preparation/GEMV comparison
on four patterns: 1,310,720 outputs, maximum error zero. The format-level balanced
ABBAAB adds 2,621,440 exact outputs. `format-interleaved-summary.json` records
three post-warm samples per arm per layer. Its event windows and logical packed
bytes estimate effective payload bandwidth, not physical bandwidth. One resident
layer has different cache and clock conditions from a full model. In this test,
21/23 improves from 156.16 to 196.16 logical GB/s; 13/13 from 211.81 to 332.89;
23/23 decreases from 203.77 to 195.92. Do not replace E2E evidence with these
mixed microbenchmark results or sum overlapping stages.

## Full-model balanced A/B

Both arms use device segments, identical weights, canonical arithmetic, FP32
states/KV, context capacity512 and B1. Only the uniform-row flag changes.
One loaded model, both complete-request warmups, 64-token sustained warmup,
then ABBAAB with three trials per arm; profiling is disabled. Eight greedy
outputs provide seven decode intervals. Rates include CPU selection and checks.

| Fixture | Control tok/s | Candidate tok/s | Relative gain |
|---|---:|---:|---:|
| Chinese | 8.365890 | 8.731103 | 4.37% |
| English | 8.359763 | 8.730091 | 4.43% |
| Repeated-number prompt, 74 tokens | 8.325729 | 8.693879 | 4.42% |

Raw trials, launch settings and binary/library hashes are under
`results/phase6/uniform-ab-sustained`. Prompt logits, sampled IDs for these
greedy runs, and reset prefixes are exact. Compute E2E includes reset, prefill,
first head, selection and decode; excludes loading, tokenization, warmup and
result output. This preserves the phase5 approximately8.35tok/s control.
The earlier short-fixture A/B had approximately7.68tok/s controls; it is retained
under `uniform-ab`, not used as the final absolute baseline.

## Completed numerical gates and limits

The serial workflow completed724 full-vocabulary steps, reset, actual processed
positions up to509, and480 sampled-step comparisons (seed42, temperature0.7,
top-k40). Every compared logit is byte-identical, maximum error0; sampled IDs and
reset agree. The fixed0.001 absolute gate is unchanged. State diagnostics compare
recurrent/convolution digests and hidden/logits
across 1/3/4/64-layer cases, including forced mixed fallback at layers2/17/31.
All twelve state trials pass. Reports are `uniform-validation/attention-full-comparison.json`,
`uniform-sampling/summary.json` and `uniform-state/device-oracle.log`.

`results/phase6/best-config.json` records the optional best environment and pinned
runtime hashes; `uniform-runtime` preserves that runtime. The source default
remains unchanged. Set `AMD_INFER_UNIFORM_ROW=0` to return to the stable device
path. `stage-summary.json` indexes profiles, numerical gates, performance and
memory; `kernel-vs-phase5.diff` preserves the precise kernel source change.
Requested peak allocation is13,579,250,072 bytes, unchanged from phase5. Actual
minimum HIP-free values are retained per request; requested bytes omit driver
rounding. The2GiB+16MiB reserve is unchanged.

Unchanged arithmetic does not eliminate the existing independent upstream Vulkan
differences. The native top128 log-prob and teacher-history comparisons retain
their separate precision limits. This is bounded512-capacity evidence, not a
long-context or official model-accuracy qualification. The approximately25.6tok/s
Vulkan reference remains faster; no victory or threefold speedup is claimed.

## Device-resident dispatch diagnosis

`uniform-engine-profile` repeats a five-token English prompt with both device
arms, full-request and64-token sustained warmup, ABBAAB, HIP events and host
counters. These diagnostics reduce measured rates to7.347/7.642tok/s; they do
not replace the uninstrumented8.36/8.73 headline. Across twelve forwards:

| Event window | Control seconds | Candidate seconds |
|---|---:|---:|
| FFN, including norm/gate/down | 0.638136 | 0.578807 |
| GDN | 0.542480 | 0.536468 |
| gate/up substage | 0.349253 | 0.291589 |
| down substage | 0.241704 | 0.239924 |

Gate/up logical packed-payload bandwidth rises192.74 ->230.85GB/s; down remains
146.30 ->147.39GB/s. Substages are contained in FFN and must not be added to it.
Event windows contain submission gaps, not exclusively useful device compute.
The unchanged down path and improved gate/up window support the targeted effect.
FFN and GDN remain the largest observed windows.

The public-ABI `profile_launch.cpp` observer records mean CPU submission durations
of3.670/3.464microseconds per `hipLaunchKernel`. Counts103202/43856 include all
warmups; control includes sustained preconditioning, so their totals are not
per-arm latency comparisons. Timer/observer overhead and HIP-event waits perturb
the instrumented run. CPU submission overlaps GPU work and does not measure
recoverable idle time. Host copies remain44H2D and32D2H calls per request in both
arms; no copy-removal improvement is claimed in this phase.

The bounded256-byte `probe_graph.cpp` confirms graph instantiate/replay on an
explicit nonblocking stream, with exact output. Capturing the legacy default
stream returns900. Its one-order1000-memset diagnostic shows no gain (direct
synchronized0.001854s, graph0.050424s); this is a capability probe, not inference
performance or a validated graph regression benchmark. Raw results are in
`graph-probe.log`; `dispatch-diagnosis.json` preserves the complete scope.

The next justified experiment is a stable-pointer local segment on an owned
explicit stream, with consistent kernel/event/copy dispatch and the same
724/reset/480 and uninstrumented ABBAAB gates. Current kernels use defaultstream0,
so graph integration cannot simply be switched on. Do not introduce global
spin barriers or promise engine speedup from the probe. No inference graph or
new persistent-kernel implementation has been added in this phase.

## Reproduction

Use the already configured local WSL Rust/HIP environment and recorded model and
tokenizer paths. These drivers require the retained local reference artifacts;
they are not a portable benchmark package. Coordinate GPU use and preserve old
output directories before rerunning.

```sh
cargo build --release --features hip,tokenizer --bins
python3 tools/profile-gemv-shapes.py
python3 tools/run-uniform-state.py
python3 tools/run-uniform-validation.py
AMD_INFER_PRECONDITION_TOKENS=64 UNIFORM_AB_LABEL=uniform-ab-sustained python3 tools/run-uniform-ab.py
python3 tools/run-format-interleaved.py
python3 tools/run-best-uniform.py results/phase2/cases results/phase6/replay-new
```

The sustained model A/B and format interleaving compare three post-warm samples
per arm in balanced ABBAAB order.
Independent CPU row dequant source and executable hashes are recorded in the
shape launch; actual per-run binary/library hashes identify each full-model run.
The replay helper verifies frozen runtime hashes and refuses an existing output
directory. Add `--rollback` to select the stable device path with the same runtime.
