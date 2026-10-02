# GPU context capacity and execution evidence, phase4

The append-only GPU attention cache now accepts128/256/512 capacity. Kernels
reserve512 shared scores and assign time rows with `t=lane; t<count; t+=256`,
retaining each dot and output accumulation order. Rust cache offsets use actual
capacity, bounds reject the next append, and a model cannot change its cache
backend during a sequence without reset. Default capacity128 is unchanged.
`AMD_INFER_CONTEXT_CAPACITY=256` or512 with `AMD_INFER_GPU_ATTN_CACHE=1` selects
the extended device cache; the CPU diagnostic cache remains available explicitly.

Validation uses the saved canonical flags, FP32 activation/KV/state storage,
main64 layers and the same mixed-format packed GGUF. This is still a Rust host
graph with HIP kernels and host q/k norm/RoPE preparation, not a full-device
engine or whole-model megakernel. IQ3_S packed-table experiment stays off.

## Correctness and actual positions

`check-ops` independently computes CPU attention for counts1/7/32/128/129/255/
256/257/511/512. Maximum error0. The256 and512 GPU caches each pass two complete
synthetic rounds and reset, exactly matching the checked attention operator;
capacity overflow is rejected. This covers the512-row boundary physically.

Full model, strict absolute gate0.001, unchanged:

| Run | Actual prompt / output | Highest actual position | Full-vocabulary comparisons | Max error |
|---|---:|---:|---:|---:|
| GPU256 free, English/Chinese/code | 156/64,152/64,145/64 | 218/214/207 | 192 steps vs saved CPU-attention free paths | 0, byte exact |
| GPU256 teacher, two reset trials each | same | same | 384 steps vs CPU-attention teacher | 0, byte exact |
| GPU512 teacher, two reset trials | 447/64 | 509 | 128 steps vs new CPU-attention teacher | 0, byte exact |
| GPU512 free vs CPU teacher | 447/64 | 509 | first20 steps with matching native history only | 0, byte exact |

GPU512 processes510 positions, not512 full-model positions. Capacity512 is
tested fully by the separate operator/cache oracle. The last sample is not fed
back. Full-model CPU teacher is a baseline, not proof through self-equality.
Both full graphs share checked HIP GEMV/GDN primitives; they are not independent
full CPU models. Actual token histories match before numerical comparisons.

An additional independent CPU DeltaNet oracle runs512 steps without reset:
3,145,728 output values, max2.7939677e-9; four whole-state snapshots,
3,145,728 state values, max6.519258e-9. The existing primitive tolerances were
not widened. This synthetic bounded-state test does not prove learned-model
state accuracy against upstream at arbitrary context lengths.

## Independent upstream quality

Reference is official native llama.cpp b11284, same GGUF and actual local
official template, greedy, one slot, ubatch1, FP32 KV, no flash attention.
Raw input IDs are verified against native tokenize. Both models are never
resident simultaneously. The owned server's launch configuration and idle
metrics are retained before stopping. Initial startup accidentally retained
context256 and changed ubatch; it received no test requests, was stopped idle,
and the corrected context512/ubatch1 configuration is used for all new data.

GPU256 directly matches the previously reported top128 scores: English/code
teacher greedy64/64; Chinese63/64 with free divergence step20. Chinese63/64
is a teacher alignment metric, not identical generated text. GPU512 English
teacher is62/64, with free divergence step19, position465: native token11834
is Rust rank2. Mean top128 overlap98.12%, score correlation0.999082,
log-prob RMSE0.113730 and maximum difference1.570243. All outputs hit64 limit;
no completed long-task accuracy claim follows.

The earlier code maximum1.648613 is step30/position174, vocabulary364,
native rank49, probability8.07725e-9; it is not selected. Code top1 max
log-prob difference0.047373, and native p>=0.1 region max0.095073. Its
early/middle/late top128 RMSE is0.117177/0.090587/0.045542. The near512 English
maximum1.570243 is step25/position471, rank41, probability4.27510e-7;
top1 max0.100094 and p>=0.1 region max0.350305. Its thirds RMSE is
0.103725/0.120620/0.116518. The longer prompt has more difference than the
shorter English probe; differing inputs and distributions prevent attribution
solely to accumulated state. Neither low tail probabilities nor high score
correlation dismiss remaining high-probability rank inversions. Upstream full
raw vocabulary logits/per-layer state are not available through this API.

**Arithmetic attribution correction:** earlier reports assumed Vulkan Q8
activations. Inspection of the existing local upstream
`../../llama-reference/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp`
shows floating activation vectors, packed32 reads and block-scale hoisting
after register FMA accumulation. This is source evidence, not a trace of the
pinned binary's actual selected shader. Vulkan activation quantization is
not established; reduction/scale ordering and norm/transcendental precision
remain plausible contributors, not an isolated causal diagnosis. Historical
raw reference failures and numerical outputs are preserved.

## Memory

512 KV for16 layers is67,108,864 bytes, query/gate/output scratch1,179,648.
GPU peak requested13,576,968,344 bytes (about12.64 GiB), excluding driver
rounding/context/code objects and CPU memory. Main packed matrices remain
13,333,954,560 bytes, excluding whole embedding/MTP. Observed free includes
other desktop apps; the unchanged2 GiB+16 MiB allocation reserve is enforced.
Every request records memory; use `stage-summary.json` for the minimum across
all full runs. No application is stopped to create space and streaming
admission remains available.

## Profile and next execution choice

One loaded model, same five-token English prompt/eight outputs, both routes
warmed, balanced ABBAAB, three trials each. CPU attention vs GPU cache, both
capacity512, same canonical arithmetic. All prefix logits byte-identical;
reset and IDs match. These are diagnostic profiles, not headline throughput:

| Host-clock profile, GPU route | Calls | Bytes | Wall seconds |
|---|---:|---:|---:|
| H2D | 2708 | 55,240,480 | 0.321711 |
| D2H | 2324 | 55,377,920 | 0.274070 |
| Explicit device sync | 2324 | — | 1.465495 |

The window has12 forwards (5 prefill +7 decode) and8 LM-head calls. Small-copy
averages are approximately119/118 microseconds. Copies account for about27%
of2.197756-second compute E2E. Sync wall includes useful device work and
cannot be called removable overhead. Host layer spans are attention0.513088,
GDN0.700864, FFN0.819352, initial norm0.003877 seconds; they include nested
device/copy waits and must not be added to the prior categories.

Paired event profile: GPU copy wall0.294837+0.242051, explicit sync0.127804,
and2120 blocking event/timed-linear calls1.371155 seconds. Event end waits
include kernel work; this explains where the non-event sync wall goes. Device
FFN0.593599, GDN0.503666; FFN gate/up0.350969, down0.229041, norm0.013240
seconds. Gate/up/down/norm are contained in FFN, not additive with it. Kernel
event instrumentation perturbs timing and excludes some uninstrumented work.

Short-context GPU cache is3.1–3.4% slower than CPU attention in these two
diagnostic runs because it adds576 H2D and192 D2H/sync calls. Long512 numerical
runs observe GPU decode about5.74 vs CPU4.66 tok/s on identical teacher history,
but those runs are sequential, not balanced performance A/B. Neither result
is a blanket speedup claim.

The next selected architecture is device-resident hidden/normalized/residual
vectors and GDN/FFN intermediates across contiguous layer segments, followed
by device q/k preparation and repeated decode command graphs. Preserve the
current graph as a fallback and require the same full-state/reset and independent
quality checks. HIP stream/graph kernel boundaries provide ordering; avoid a
global spin barrier over4096 blocks that can exceed resident occupancy.

Copy/sync removal alone cannot close the remaining roughly4x same-weight
Vulkan gap: useful device FFN/GDN work is substantial. The complementary
computational target is vector-packed, block-scaled GEMV, informed by the
observed upstream IQ4_XS shader, rather than more unproven table-load tweaks.
Moving scale outside element accumulation changes FP32 rounding order; any
candidate must be opt-in with an explicit independent CPU arithmetic oracle
and upstream teacher quality checks, not a widened error gate. This device
execution architecture and grouped GEMV are selected next work, not implemented
performance gains in phase4.

All commands/environments/hashes and raw output are retained under
`results/phase4`. `run-extended-attention.py` coordinates serial full-model
runs; `profile-extended-execution.py` produces paired profiles. Native API
requests and restart argv are retained in `context512`. No software was
installed, no unrelated processes stopped, no security settings changed,
and nothing published.
