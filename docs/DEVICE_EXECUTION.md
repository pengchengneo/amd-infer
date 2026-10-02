# Device execution, phase5

`AMD_INFER_DEVICE_SEGMENTS=1` enables a complete 64-layer device execution
path when the packed matrices are resident. It remains opt-in. Rust owns buffer
lifetimes, shape/capacity checks and graph scheduling. HIP retains hidden,
normalized inputs, residuals, recurrent/conv state, Q/K preparation, KV cache and
FFN intermediates between layers. Kernels execute on the default stream; there
is no grid-wide spin barrier, persistent whole-model megakernel or HIP graph.

The existing mixed path remains selectable with the flag unset/zero. If a layer
has an omitted streamed FFN or other nonresident matrix, execution downloads
hidden, executes the existing mixed layer and uploads hidden at the next eligible
device layer. `AMD_INFER_DEVICE_SKIP_LAYERS=2,17,31` is a diagnostic way to exercise
these transitions with resident weights. This forced transition check is not a
streaming-performance measurement.

The path requires GPU attention cache, stable RMS, projected GDN and GPU FFN.
Tensor/host tracing is rejected for this experimental path rather than silently
changing execution. RoPE coefficients retain the Rust FP32 libm policy and are
uploaded once per position; Q/K norm and rotation application are on device.
The attention-preparation kernel uses ordered FP64 norm and split FP32 rotation
arithmetic. Residual additions explicitly round to FP32. GEMV accumulation,
quantized weights and GDN/FFN arithmetic are unchanged.

Embedding row reads/dequant upload/readback, once-per-position coefficients,
final hidden readback, CPU final norm, LM-head input upload and logits readback
remain. This is a local experimental full-model path, not a released portable
engine. No model accuracy or whole-model megakernel claim is made.

## Verification

`results/phase5/device-long/attention-full-comparison.json` records all 724 frozen
full-vocabulary comparisons, with byte identity/max absolute error0 at the
unchanged0.001 gate. It covers Chinese/English/code free generation, same-upstream
teacher histories, two resets, capacity256 and512. The longest full-model run
processes510 positions, highest position509; this is not a long-context claim.
The512 operator/cache boundary was independently checked in phase4 and the same
attention kernel is reused.

`final/device-oracle.log` records single-layer, three-layer, attention-containing
four-layer and full64-layer tests. Eight fixed-history tokens are checked against
the mixed path in two complete device trials and one forced-transition trial.
Hidden and full logits are byte-identical. Every executed GDN recurrent/conv
state digest also matches after every token. Digests are diagnostic bit-pattern
evidence, not an independently exposed upstream-state oracle. Forced transitions
at layers2,17,31 preserve state and output. Old raw-state DeltaNet independent
512-step checks remain in phase4.

`sampling/summary.json` records the existing Rust sampler at seed42,
temperature0.7/top-k40,32 outputs, English/Chinese/74-token repeated-text prompts,
ABBAAB and reset repetitions. This is same-Rust-sampler consistency, not Vulkan
sampler equivalence. Full sampled logits are retained.

Independent Vulkan teacher comparisons reuse the frozen native responses from
phase3/4; they are recomputed against the new device logits under
`independent256` and `independent512`. Numerical differences remain: English/code
256 teacher greedy64/64, Chinese63/64, near512 English62/64. Chinese free first
divergence20 and near512 English19 remain. The64-output long tasks are truncated;
do not claim complete-answer quality. Upstream top128 log-prob differences and
arithmetic-policy limits are preserved in those reports. Original historical
FP32-reference failures remain failures; the gate was not enlarged. Upstream
Vulkan activation quantization is not established.

## Performance and profile

`ab-performance/summary.json` is the headline experiment: one model load, both
complete-request warmups, fixed same-weight/canonical math, GPU attention for
both arms, context capacity512, B1, FP32 KV/state,8 greedy outputs and ABBAAB
with three trials per arm. Decode rate uses7 intervals and includes CPU selection
and finite-logit checks. Prefill is sequential forward time. Compute E2E includes
state reset, prefill, LM head and decoding, but excludes model loading,
tokenization, warmup and result-file writes. Raw load/warmup/TTFT/prefill/decode/E2E
metrics are retained. Saved execution counters verify full residency:768 device
layers/0 mixed layers for the5-token English candidate request, and the inverse
for its control. Actual streaming throughput is not compared here.

Balanced decode medians are5.872->8.393 tok/s Chinese,5.846->8.380 English,
5.830->8.346 for the74-token repeated-text fixture, about43% gain. Prompt logits,
generated IDs and reset results match. The historic same-weight Vulkan25.6 tok/s
reference remains substantially faster; its capacity128, arithmetic and API
timing scope differ, so it is context rather than a new strict A/B result.

Host and HIP-event profiles are separate diagnostic runs, not headline timings.
For the5-token/8-output request (12 forwards and8 LM heads), host-visible H2D/D2H
calls fall from2708+2324=5032 to44+32=76, explicit sync2324->32. Copy wall falls
from0.296453+0.250585=0.547038 seconds to0.005014+0.007366=0.012380 seconds.
Without events, sync wall remains1.312314 seconds in the candidate: this includes
useful queued GPU work, not1.31 seconds of removable synchronization overhead.
Counts omit D2D, allocation and kernel-launch calls; do not claim all driver calls
were measured. Candidate host layer-wall zero fields are unmeasured, not zero
computational work.

With events, candidate FFN window0.637743 and GDN0.541944 seconds remain large.
FFN norm0.047668, gate/up0.348629 and down0.241446 are contained in FFN and must
not be added to it. Events include queue/host-submission gaps and perturb the
path; candidate decode falls to7.412 in this instrumented experiment. These are
event windows rather than isolated physical kernel throughput. Gate/up logical
compressed-payload bandwidth is about191->193 GB/s; down about154->146 GB/s.
There is no verified significant GEMV-bandwidth improvement in this stage.
No physical memory-bandwidth or GPU clock counters were collected.

The next computational target is packed/block-scale-aware quantized GEMV.
IQ4_XS accounts for66 of128 FFN gate/up matrices and3.125GB of5.610GB gate/up
packed payload per forward. Preserve arithmetic order where possible, and use
an explicit matching-arithmetic CPU oracle plus independent upstream teacher
checks if regrouping changes rounding. The inspected upstream MIT shader and
license attribution are recorded in phase4/THIRD_PARTY.md; no such port was
added here. More copy removal or a global persistent spin barrier is not justified
as the primary fix. A bounded stream/graph prototype can later A/B remaining
submission overhead; no graph speedup is currently claimed.

## Memory, provenance and reproduction

Packed resident bytes13,333,954,560. Requested allocation peak13,579,250,072
(about12.65GiB), with reserve enforcement still2GiB+16MiB. Minimum HIP-free in
the long runs about2.14GiB; observed free includes other desktop applications.
Requested bytes omit driver allocation rounding. KV512 for16 layers is64MiB;
device hidden/norm/preparation adds about2.18MiB over phase4. Safety thresholds
were not lowered and other applications were not stopped.

The long tests use the pinned initial phase5 runtime in `device-runtime`.
Final Rust changes add diagnostic timing/counters/state digests and test-only
forced transitions; HIP library SHA256 remains
`40221835c818fa06df9c045c3b54d2cbdac3d6fd4489de1d3c60c7263c62f1ea`.
Final runtime is separately retained for the supplemental oracle/A/B/sampler.
Per-run launch records identify the actual binary, environment and library.
Do not assume every run used the same evaluate executable hash.

Using the existing configured AMD-Infer WSL/Rust/HIP environment, build with
`cargo build --release --features hip,tokenizer --bins`. Serial local drivers:

```
DEVICE_ORACLE_LABEL=review python3 tools/run-device-oracle.py
python3 tools/run-device-long.py
python3 tools/run-device-ab.py
python3 tools/run-device-sampling.py
python3 tools/report-device-stage.py
```

Drivers use locally retained phase2 launch configuration and phase3/4 reference
artifacts; the GGUF and tokenizer must already exist at recorded paths. Preserve
existing run directories before rerunning. Results and stage index are under
`results/phase5`; source/runtime snapshots are retained. No installation,
unrelated process stop, security-setting change or publication occurred.
