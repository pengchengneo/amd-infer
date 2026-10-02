# Phase7: explicit-stream local HIP graphs

This experiment is opt-in: `AMD_INFER_EXPLICIT_STREAM=1` and
`AMD_INFER_LOCAL_GRAPHS=1`. Both are off by default. Phase6's frozen8.7tok/s
configuration/runtime remains the best validated path. The completed balanced
A/B shows regression, so this prototype is not promoted.

## Captured work and dynamic data

The runtime owns one explicit nonblocking HIP stream per host thread, selected
before initial allocation. Every project kernel, timing event and D2D copy uses
that stream. H2D/D2H and reset memset use the same stream followed by stream
synchronization, preserving the prior synchronous ABI. This is B1, single-thread
runtime ownership; no cross-thread or multi-GPU execution guarantee is made.
Legacy stream0 remains selectable. No default-stream capture retry or workaround
is used.

48 GDN graphs capture five actual packed GEMVs, convolution/history update,
q/k normalization, DeltaNet recurrent-state update and output normalization/gate:
nine kernel nodes per graph.64 FFN graphs capture RMS, fused gate/up and down:
three nodes per graph. The complete cache is112 graphs/624 kernel nodes.
Each capture records work without executing it; one subsequent graph launch
executes that token's state update exactly once.

Graph kernels dereference the stable device buffers on each replay; current
inputs are written before launch on the same stream. The cache keys include
every device pointer, format, FFN dispatch argument and relevant arithmetic flag.
They do not freeze token values. Attention preparation/RoPE/KV append and its
changing position/count remain ordinary explicit-stream dispatch outside graphs.
No captured scalar position or KV-count argument is reused across tokens.
Reset clears state/history and resets KV counts outside capture; graph cache
survives. Mixed fallback supplies fresh workspace inputs through the same ABI.

Cache size is bounded128. Capture is skipped near the unchanged2GiB+16MiB
reserve; post-instantiation free memory is checked too. Skips and failures are
reported. Allocation ranges track referenced buffers: freeing an unrelated
temporary embedding buffer preserves graphs, while freeing referenced storage
invalidates its graph entries before reuse. Unknown ownership invalidates all
entries conservatively. Stream work is drained before freeing allocations.

## Evidence so far

`results/phase7/state/device-oracle.log` records twelve actual graph-vs-normal
trials:1/3/4/64 layers, eight distinct tokens each, reset and forced mixed layers
2/17/31. Hidden, full logits and recurrent/conv digests agree exactly, maximum
error0, fixed0.001 gate unchanged. Full64-layer trial replays896 graphs over
eight tokens; later reset trials capture zero new graphs and reuse all112.
These are same-arithmetic state diagnostics, not independently exposed upstream
state or model-accuracy proof.

The serial724-step and480 sampled-step workflows completed with byte-exact
logits, unchanged0.001 gate, identical sampled IDs and reset. They reuse
the existing saved CPU/same-arithmetic and upstream-history fixtures, including
actual positions up to509. Full results, captures/launch counts and memory are
recorded under `graph-validation` and `graph-sampling`. Independent upstream
precision/generation differences remain separately recorded; this experiment
does not change reference precision or widen tolerances.

The strict repeat under `reload/device-oracle.log` adds signed-zero-aware bit
comparison for all twelve trials, followed by model destruction and reload.
After destruction the graph cache is empty and invalidations are recorded; the
new model captures112 fresh graphs. Its hidden/logits and recurrent/conv digests
match ordinary dispatch exactly. Unrelated embedding temporary frees do not
recapture graphs during the long tests. All measured long requests reuse112
graphs, with zero captures/failures/skips/invalidations inside the request.

## Performance and setup accounting

The graph0/1 A/B uses identical explicit-stream, device-resident, uniform-row
configuration; it isolates graph dispatch rather than the separate legacy-to-
explicit stream transition. One loaded model, both complete-request warmups,
64-token sustained warmup, then ABBAAB, three samples per arm. Prefill, seven
decode intervals for eight outputs, and compute E2E retain the existing scope.
Uninstrumented results belong under `graph-ab`; launch instrumentation is a
separate run and must not replace headline rates.

`.graph.json` counts only the measured request. Counter reset retains graphs;
`.preceding-graph-counters.json` and `initial-warmup-graph.json` retain excluded
warmup/capture setup. Capture setup time covers begin/body/end/instantiate and
post-instantiation reserve check; preceding stream drain is excluded. Whole
warmup wall includes useful execution and other overhead. Amortization must
compare setup cost with observed per-forward/request savings; a negative saving
has no positive break-even point. Cache entry/node counts alone are not speedup.

| Fixture | Graph off / on tok/s | Prefill off / on seconds | Compute E2E off / on seconds |
|---|---:|---:|---:|
| Chinese,10 prompt tokens | 8.7310 / 8.3176 | 1.0192 / 1.0775 | 1.8430 / 1.9399 |
| English,5 | 8.7304 / 8.3133 | 0.5073 / 0.5432 | 1.3308 / 1.4075 |
| Repeated numbers,74 | 8.6937 / 8.2782 | 7.5678 / 7.9945 | 8.3954 / 8.8608 |

Warmed decode regresses4.73–4.78%; prefill and E2E also regress. The first112
captures cost0.0932442seconds outside the measured requests; no positive amortized
break-even exists for these fixtures. Normal explicit-stream controls reproduce
the8.7 path, but this is not a separate legacy-vs-explicit transition A/B.

The separate public-ABI observer records an English request with twelve forwards
and eight LM heads:10964 ordinary kernel launches become3476 ordinary launches
plus1344 graph launches, a56.04% reduction in observed launch API calls. All624
captured kernel nodes still execute per forward; this does not reduce useful
kernel work. Submission wall falls0.007229 ->0.005945seconds in the diagnostic
run.132 explicit stream synchronizations remain in both arms, with useful GPU
work included in their wait durations. Observer counters are reset after model
reset, so they cover prefill/head/decode; compute E2E additionally includes reset.
Other driver/configuration helper calls are not counted. Events from phase6 are
not additive with these spans. Actual GPU-internal scheduling regression remains
unresolved because WSL hardware counters are unavailable; no cause is invented.

Requested allocation peak remains13,579,250,072bytes. The512 teacher run reports
minimum free2,274,271,232bytes, above the unchanged reserve. Requested accounting
excludes graph/driver allocations and rounding; observed free includes desktop
applications and cannot isolate graph allocation cost.

`stage-summary.json` indexes numerical coverage, launch diagnostics, memory and
runtime hashes. `experimental-config.json` records this tested configuration
with `recommended:false`; `graph-runtime` freezes its executables/library.
Use the unchanged phase6 best-config/uniform-runtime to recover the best path.
No graph or explicit-stream default is enabled.15 unit tests, formatting and
Python compilation pass. Source diffs and a final source snapshot are retained.

## Safe next scope

No whole-model persistent kernel, graph containing changing attention arguments,
or global spin barrier is introduced. FFN gate/down dependency crosses output
rows, so a persistent whole-FFN fusion needs an explicit safe synchronization
design. A bounded alternative is head-local DeltaNet update plus output norm/gate
with a block barrier; it must preserve rounding/state and be supported by an
operator profile before implementation. Do not equate local fusion with a
persistent whole-model megakernel.

Build and serial local drivers (existing model/reference artifacts required):

```sh
cargo build --release --features hip,tokenizer --bins
python3 tools/run-graph-state.py
python3 tools/run-graph-validation.py
python3 tools/run-graph-ab.py
```

Preserve run directories and coordinate GPU use before rerunning. No install,
unrelated process stop, security change or publication is involved.
