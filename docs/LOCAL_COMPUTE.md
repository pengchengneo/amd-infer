# Phase8: DeltaNet head locality and multi-row quantized GEMV

The optional recommended configuration enables `AMD_INFER_WARP_GEMV=1` with
device segments and phase6 uniform gate/up. Head fusion, local graphs and explicit
stream stay off. Source defaults remain unchanged. The verified phase6 frozen
runtime and8.7 configuration remain intact. The incoming unfinished phase8 draft
was separately saved before repairing the misplaced resource-query definition.

## Implementation and arithmetic

`warp_quant_linear<Kind>` specializes all supported wire formats, gives each
32-lane wave one row and processes four rows per128-thread block. Each lane owns
four accumulators corresponding to the prior128-thread row. It retains the two
per-tile FMAs and reconstructs `(s0+s2)+(s1+s3)` before the same wave reduction.
Weights, FP32 activations, dequant multiplication and input rounding stay fixed.
Matrices with fewer than512 rows keep the original dispatch. GDN/attention
projections, LM head and FFN down use this optional kernel; fused FFN gate/up is
unchanged. No activation quantization or IQ4 block-scale regrouping was added.

The inspected MIT upstream Vulkan IQ4_XS shader uses eight threads per256-weight
block, packed32 weight words/vector activations and scale-hoisted accumulation.
Its row/column tiling and reduction differ from this project. The upstream
CUDA/HIP MMVQ source uses format/device dispatch and Q8 activation vec-dot paths;
that does not establish the activation precision or exact shader dispatch of
our retained Vulkan benchmark. Our implementation is independently written and
preserves its established arithmetic; references/licenses are in THIRD_PARTY.md.

## DeltaNet access and bounded persistent scope

There are48 recurrent layers,48 value heads/layer and128x128 FP32 state/head:
3MiB/layer,144MiB overall. In `[head,value,key]` layout, adjacent value lanes access
addresses512 bytes apart for fixed key. The source has two read/write passes,
nominally576MiB of state access/forward before cache/compiler effects; this is
not measured physical traffic. Column layout coalesces value lanes but requires
consistent state ownership, diagnostic layout and fallback handling across the
engine. Its current evidence is a synthetic cached probe, so it was not adopted.

`AMD_INFER_GDN_HEAD_FUSE=1` combines decay/prediction/update/query projection
with output RMS/gate. A block owns a disjoint head and all128 threads follow its
head loop. `AMD_INFER_GDN_HEAD_BLOCKS=24/32/48` permits a bounded one/two-head task
loop; block barriers protect the FP64 norm reduction and shared reuse. No global
barrier, spin wait, inter-block communication or across-token state residency is
introduced. It retains the physical state layout and core output buffer.

Fresh synthetic probe medians, four samples/arm and512 iterations/sample:

| Variant | Control us | Candidate us | Latency reduction |
|---|---:|---:|---:|
| column, two launches | 57.85 | 25.96 | 55.12% |
| row fused,48 blocks | 60.99 | 41.52 | 31.92% |
| column fused,48 | 61.06 | 24.85 | 59.31% |
| row fused,24 | 60.86 | 64.47 | -5.94% |
| row fused,32 | 60.75 | 69.79 | -14.88% |

All five candidates match core/gate bits at every step and logical state bits
every64 steps across two512-step passes and reset. These cached3MiB timings are
not full-model speed. Resource estimates report35 VGPR/1KiB LDS/no scratch for
fused heads. Smaller24/32 grids do not justify additional engine tuning.

The isolated48-block head experiment holds warp GEMV on in both arms. Decode
15.8958 -> 16.0595tok/s (+1.03%), but prefill
0.298718 -> 0.313843s and compute E2E
0.751193 -> 0.761545s regress. It stays experimental.
The combined configuration reaches15.91-16.05tok/s on the three short-output
fixtures; it is not substituted for the balanced recommended warp-only result.

## Full-model uninstrumented ABBAAB

One loaded model,64-forward sustained warmup, both full-request warmups, three
samples/arm. Same weights/arithmetic, B1, FP32 KV/state and context512. Eight
outputs give seven decode intervals including CPU selection/checks. Prefill is
sequential forward time; compute E2E includes reset/prefill/head/decode and
excludes load/tokenization/warmup/result output. All prompt and last logits,
generated IDs and reset prefixes agree by bytes with the frozen8.7 runtime.

| Fixture | Control tok/s | Warp tok/s | Gain | Prefill s control / warp | E2E s control / warp |
|---|---:|---:|---:|---:|---:|
| chinese | 8.7997 | 15.9199 | 80.91% | 1.0087 / 0.6158 | 1.8268 / 1.0674 |
| english | 8.7907 | 15.9107 | 80.99% | 0.5061 / 0.3150 | 1.3241 / 0.7681 |
| long | 8.7585 | 15.7984 | 80.38% | 7.5147 / 4.5183 | 8.3361 / 4.9736 |

## Numerical coverage and memory

Both warp-only and combined candidates independently complete the unchanged724
full-vocabulary steps, resets, capacities256/512 and actual positions through509.
Every compared logit is byte-exact, max error0, fixed0.001 gate unchanged. Each
also passes480 sampled-step comparisons, seed42/temperature0.7/top-k40, identical
IDs and resets. This is the existing Rust sampler, not an independent upstream
sampler qualification. Frozen CPU/same-policy and upstream-history fixtures are
retained. Existing independent upstream precision/generation differences remain.

The untouched CPU GGML dequant oracle checks38 real format/shape representatives
and five distinct inputs (three dense, zero, sparse),760 selected dot outputs,
max error0; all GPU outputs are also checked finite. Four actual GDN layers
compare4,718,592 values to the independent Rust CPU conv/L2/Delta/RMS/gate oracle,
max error1.0431e-7 at the unchanged operator gate. Twelve1/3/4/64-layer state
trials pass signed-zero-aware hidden/logit bits and recurrent/conv digests,
including layers2/17/31 forced mixed fallback. Digests are same-math diagnostics,
not independently exposed upstream state. Logs and exact scopes are indexed in
`results/phase8/stage-summary.json`.

Requested peak stays13,579,250,072 bytes. Long suites observe minimum free
2,330,193,920 bytes, above the unchanged2GiB+16MiB reserve. Requested accounting omits
driver rounding; observed free includes desktop applications.

## Profile, gain ceiling and remaining gap

A separate instrumented balanced English run measures these GDN windows over12
forwards; it reduces decode to7.27/11.43tok/s and is not the headline benchmark.

| GDN window | Control seconds | Combined seconds |
|---|---:|---:|
| qkv | 0.229855 | 0.087154 |
| z | 0.123246 | 0.040232 |
| alpha | 0.014546 | 0.014519 |
| beta | 0.021432 | 0.016481 |
| conv | 0.006029 | 0.005910 |
| l2 | 0.021290 | 0.018209 |
| delta_or_fused_head | 0.024567 | 0.025940 |
| norm_gate | 0.017523 | 0.004504 |
| output_projection | 0.151034 | 0.062981 |

The fused-head window includes output norm/gate; its following empty marker
still has event overhead. FFN down window falls
0.234847 -> 0.156315s. Gate/up stays
0.284813 -> 0.283848s. Substages belong
inside their enclosing FFN/GDN windows and cannot be added to them. No physical
occupancy/cache/bandwidth/clock counters are available in this WSL environment.

Current English latency is113.76 -> 62.85ms/token. Historical Vulkan25.6tok/s
is39.06ms/token: about23.79ms remains, with
68.2% of the historical latency gap removed. Vulkan's128
capacity and timing/arithmetic scope differ; this is context, not a new strict
SOTA comparison. The fresh diagnostic entire head segment is about3.51ms/forward.
Even an illustrative zero-cost segment moves the old control only to
9.07tok/s. State persistence alone cannot close the gap.

Unchanged gate/up is the largest remaining observed computational target at
about23.65 diagnostic ms/forward. An illustrative50% reduction subtracted
from warmed warp latency yields roughly19.6tok/s. This is sensitivity
analysis, not a prediction: event windows include submission gaps and overhead.
The next justified scope is a packed/block-scale-aware gate/up experiment with
an explicit matching-arithmetic oracle and unchanged full-model gates, rather
than broader persistence or another graph retry.

## Local reproduction and recovery

Use the existing AMD-Infer WSL toolchain and configured ROCm/DXG environment.
These local drivers require the retained GGUF, tokenizer and reference artifacts;
they are not a portable release benchmark. Coordinate GPU use and use fresh labels.

```sh
cargo build --offline --release --features hip,tokenizer --bins
python3 tools/run-compute-experiment.py ab --kind gemv --label replay-ab
python3 tools/run-compute-experiment.py state --kind pair --label replay-state
python3 tools/run-compute-experiment.py validation --kind gemv --label replay-quality
python3 tools/run-best-compute.py results/phase4/profile-cases results/phase8/replay
python3 tools/run-best-compute.py results/phase4/profile-cases results/phase8/recover --rollback
```

`best-config.json` and `compute-runtime` freeze the recommended warp path.
`head-experimental-config.json` retains the validated optional head variant.
Both graph and explicit-stream controls are0. All source defaults and phase6
files remain intact. Source diffs, snapshot, launch hashes, checks and final memory
are retained. No service was stopped, no installation/security change was made,
and nothing was published to GitHub or another external destination.

## Final frozen-runtime replay limits

The frozen standalone English replay gives14.6075tok/s; a repeat with only the
pinned project library on LD_LIBRARY_PATH gives14.5748tok/s. Both retain
64-forward and full-request warmup and byte-exact prompt/last logits and IDs.
The preserved phase6 runtime replay gives8.8194tok/s. The library-path
repeat does not explain the lower standalone rate, so no path/clock/cache cause
is asserted. These slower observations remain recorded, not discarded.

A final frozen-runtime English ABBAAB replication gives
8.0799 -> 14.6296tok/s
(+81.06%). This reproduces the relative gain, while both absolute arm rates are lower
than the earlier source run. The deliverable frozen path measures about14.6tok/s;
15.9 is not a per-call guarantee.
No physical clock/cache counters are available to diagnose the difference.

The controlled standalone latency is68.61ms/token versus the
historical39.06ms Vulkan context, a remaining29.55ms gap.
Its rate gain against the separate preserved-runtime replay is
65.3%; that sequential comparison has weaker controls than
the balanced single-model experiment. Frozen rates are indexed under
`frozen-replay`, `frozen-replay-controlled`, `rollback-replay` and `frozen-balanced`.
