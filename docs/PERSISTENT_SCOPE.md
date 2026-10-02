# Phase14: bounded fusion, remaining budget and persistent execution

Phase14 keeps the validated phase13 runtime as the best recoverable opt-in
configuration. No engine kernel or Rust execution path changed. A standalone
single-block residual/RMS fusion was implemented and measured twice; its modest
benefit does not justify promotion without full-model quality and matched A/B.
The requested new long-request steady-state comparison could not begin because
the unchanged HIP reserve gate rejected model initialization on the shared
desktop. There is no new claim of exceeding either reference.

## Measured local prototype

The device-layer FFN boundary currently updates the hidden vector, copies it to
the workspace and runs stable RMS normalization before the projections. The
prototype combines that boundary in one 256-thread block. Each lane performs the
same rounded FP32 residual addition, then the original lane-strided FP64 sum;
the same 128-to-1 shared-memory reduction, epsilon and final FP32 products follow.
All barriers are block-local. No other block must become resident or publish
data for this kernel to complete.

Two serial 48-trial ABBAAB probes run 512 operator steps per trial, using four
input patterns: ordinary dense values, alternating values over a broad exponent
range, zeros and tiny/subnormal values. Each trial resets hidden state. After
512 steps, hidden and normalized outputs match the original ABI chain bit for
bit. Across both runs, 901,120 final element results are compared. This does not
test every intermediate step, a full network, the independent numerical oracle
or sampled trajectories. The first control trial is excluded by a fixed summary
rule; every raw timing, including cold control/candidate transients, is retained.

| Run | Control median across patterns, us/layer | Fused median, us/layer | Saved, us/layer | 64-layer extrapolation, ms/token |
|---|---:|---:|---:|---:|
| First | 12.660–12.832 | 10.196–10.246 | 2.436–2.637 | 0.156–0.169 |
| Repeat through archived entry | 12.554–12.648 | 10.163–10.231 | 2.340–2.450 | 0.150–0.157 |

These are cached API/submission/dependency-chain event windows. They include
enqueue/scheduling gaps and are not pure hardware execution counters. The
64-layer figure is a microbenchmark extrapolation, not a measured full-model
gain. At the previous roughly 43 ms/token latency it corresponds to about
0.35–0.40% potential throughput improvement. Even the earlier impossible
zero-cost removal of the entire boundary chain was bounded around 0.82 ms/token.
This candidate stays standalone; no new engine flag or default was introduced.

Reproduce with a new output directory:

```sh
python3 tools/phase14-prototype.py \
  --runtime-config results/phase13/experimental-config.json \
  --output results/my-residual-prototype
```

The C++ source is project-authored and links the recorded existing HIP library.
It uses the established arithmetic to compare against the old public ABI. No
upstream engine source was copied into this prototype.

## Resource and synchronization limits

The actual gfx1201 WSL backend reports 32 HIP multiprocessors, wave size 32,
2,048 threads per multiprocessor, 1,024 threads per block and 64 KiB shared memory
per block/multiprocessor. Both `CooperativeLaunch` and
`CooperativeMultiDeviceLaunch` report zero. These are current local query
results, not a statement about every AMD GPU or ROCm execution environment.

The prototype uses 15 reported registers/thread, 2 KiB shared memory and zero
local scratch. The HIP occupancy API gives eight active 256-thread blocks per
multiprocessor, a device-wide **resource upper bound** of 256 blocks.
The sixteen current grouped gate/up kernels report 55–112 registers, no shared
memory/scratch and 12–16 active 128-thread blocks per multiprocessor: bounds of
384–512 blocks. The two queried standalone grouped linear kernels report
36–37 registers, no shared memory/scratch and a 512-block bound.
These estimates are not physical measured occupancy or a residency guarantee
for a shared/preemptible desktop. A combined kernel must be compiled and queried
again; its resources cannot be inferred by reusing the component bounds.

The current 4,096-block FFN launch is safe because its row tasks are independent
and it has no cross-block barrier. Adding an ordinary all-grid spinning barrier
would let resident blocks wait for blocks that cannot be scheduled. Restricting
the grid to a theoretical occupancy count does not supply the missing
cooperative-launch guarantee in this backend. No such barrier was implemented.

Larger fusion encounters actual dependencies:

- RMS needs the entire updated 5,120-element hidden vector before every gate/up
  row uses its normalized input.
- Down projection needs all 17,408 gate/up activations.
- Attention/GDN output projection and the next residual depend on complete
  previous projections; recurrent state remains FP32 with captured layout.

Single-block ownership solves a small reduction boundary. Applying it to all
matrix rows would sacrifice most device parallelism. Replicating RMS across
every 4,096 row block would multiply work and nominal input access; it is not an
assumed free synchronization replacement. Across-token persistence would also
need the host sampler/logit contract or a separate device sampler API, plus
reset/capacity and preemption handling. Phase14 implements none of those changes.

If a larger persistent design is chosen later, a safer starting design for this
backend is a bounded, nonblocking dependency queue: workers process ready row
tasks, publish completed outputs with explicit release/acquire ordering, and
enqueue consumers only after prerequisites finish. Workers must never occupy
the device while waiting for a task assigned to an unscheduled block. Queue
capacity/backpressure, outstanding-work accounting, actual combined register/LDS
limits, spills and liveness must first be proved in an isolated two-stage
prototype. Merely replacing an all-grid barrier with an atomic counter is not
such a proof. Keep launches bounded per token; do not change watchdog/driver
settings. This is a design option, not a validated implementation.

## Remaining budget and decision

The existing phase13 matched data imply a 1.262–1.934 ms/token gap to the FP32
reference and a 3.074–4.151 ms/token gap to the default MMVQ reference. The
FP32-to-default reference difference is 1.765–2.217 ms/token and reflects an
activation-arithmetic difference between reference arms. Those numbers are
historical phase13 measurements reused for planning, **not** new phase14 runs
or physical per-module attribution. The 0.150–0.169 ms local saving would cover
only about 8–13% of the smaller gap and 4–6% of the larger gap if it transferred
unchanged to full-model execution. It cannot by itself meet the target.

The previous instrumented budget identifies quantized gate/up, down and the
other projections as the large remaining path. Its event windows overlap their
containers and include observer/submission overhead; do not sum nested windows
or subtract them from unobserved decode. The existing public tiling comparison
is documented in `COOPERATIVE_FFN.md`, `COOPERATIVE_MATRICES.md` and
`FP32_REGROUP.md`, with source provenance/licenses in `THIRD_PARTY.md`.
Those references do not establish a new phase14 measured gain. No precision,
activation quantization or model change was made to match the faster default arm.

The next bounded choice is to recover a complete 512-context matched baseline
and an observer-only profile with the same short–long–same-short order, then
select at most one same-FP32 matrix-tiling/shared-input candidate using actual
resource and cold-weight results. It must save at least roughly 1.3–1.9 ms/token
to reach the same-FP32 reference, and roughly 3.1–4.2 ms/token to reach both
reference arms, before allowing a margin for drift. Small boundary fusion is
not the main target. No defensible full-model percentage is assigned to an
unbuilt persistent queue or megakernel. Pursuing that option requires a broader
execution/synchronization design decision rather than repeated small tuning.

## Memory blockage and ranking drift

Both frozen phase13 baseline attempts fail before sustained warmup at the
original 2 GiB + 16 MiB gate. The first attempted 2,200-byte allocation observes
2,146,394,112 bytes free; the retry observes 2,143,244,288 bytes. Their immediate
shortfalls are about 17.9 and 21.0 MB, but lazy recurrent/cache/device buffers
still remain. These shortfalls are not a sufficient-memory estimate for a whole
run. The requested peak remains 13,579,250,072 bytes and allocator/driver
rounding is additional. The attempted setup loaded the full 13,333,954,560-byte
matrix set; it was not an accepted streamed configuration.

The final benchmark entry is also run end to end. Its first HIP arm rejects a
2,200-byte allocation at 2,138,001,408 bytes free, before warmup. It retains
`complete: false`, the full-residency admission and the original failure log,
and starts no subsequent native GPU arm. This verifies failure handling rather
than adding an accepted throughput result.

Read-only Windows evidence finds no project inference process. At the recorded
idle check, ADL reports 1,781,530,624 bytes global dedicated usage and the
per-process counter's largest entry is DWM, followed by desktop applications.
Per-process counters can include shared allocations and must not be summed as
physical usage. No unrelated process was stopped or setting changed.

Without the requested long-request steady-state measurements, thermal,
scheduling and desktop drift cannot be causally assigned or declared resolved.
The new benchmark records short-request performance both before and after the
long request, reload brackets around the native arms and every trial. Its
conservative ranking rejects excessive spread/drift and uses worst HIP versus
best reference, instead of peak throughput. The native FP32/default arms remain
separate. See `RUNNING.md` for the final entry and failure-preserving protocol.

The unchanged phase13 source engine/frozen hashes retain their prior quality
qualification. Phase14 does not claim a newly rerun 724-step, reset, 480-sampling
or independent long-context suite: there is no integrated candidate and full
model loading is blocked. All existing gate values, frozen configs, libraries,
quality evidence and phase6/8/9/10/11/13 rollback artifacts are preserved.
Full runtime and quality verification status are saved under `results/phase14`.
