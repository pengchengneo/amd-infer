# Bounded-sequence arithmetic diagnosis

The prior fast source is frozen in
`results/phase2/specialized-down-final-source.zip` and its SHA256 manifest.
Historical benchmarks and failing trajectories remain unchanged. New numerical
options are opt-in; the original kernel paths remain selectable.

## Verified local contraction-control revision

`noncontract-boundary` and `noncontract-reset` preserve the subsequent local
FP contraction-control experiment separately from all prior results.
Three fixtures (English 96+32, code/Unicode 64+64; capacity 128; seeds 42/123)
now have **max absolute error 0 on all 160 steps / 39,731,200 logits** against
the independent same-policy hybrid reference. Two fresh trials per fixture
add 320 sampled outputs: every per-step logit is byte-identical across reset
and to that reference. Selected per-layer tensors at positions 0/110/111 also
match exactly, including layer-0 recurrent/conv state. This is bounded-context
evidence, not long-context qualification or a full CPU model oracle.
An additional offline byte comparison covers all 960 full-vocabulary steps
across both grid configurations and their three trajectories (including signed
zeros), with no difference. See `noncontract-full-byte-validation.json`.

The actual old/new gfx1201 ISA is retained. Old scalar conv has one FP32 mul
and three FP32 FMAs despite source helper calls; new conv has four separate
FP32 mul and three add operations. See `contraction-isa-audit.json`.
Attention's whole-function FMA counts include compiler division/math routines;
they alone do not locate dot-product contraction.

**Historical policy still does not pass the unchanged 0.001 gate:** English
max 0.0013527870 (step 3); code 0.0017127991 (step 47); Unicode 0.0005828142
(no failed steps). All sampled IDs match, but that does not replace the numeric
gate. The separate report is `noncontract-boundary/historical-error-analysis.json`.

The reference change specifies a different precision policy, not a tolerance
change. Independent CPU norm/activation/attention compute FP64 math then round
to FP32; candidate kernels implement that same specified policy. Main activation,
KV, state and packed weights remain FP32/unchanged quantization. Both full graphs
share previously independently checked HIP GEMV/Delta primitives; they are not
fully independent at those primitives. Independent CPU Delta/state checks exist
separately. Old/native-FP32 and new/FP64-to-FP32 **CPU preparation alone** differ
on token 727: raw projections exact, mixed input max 3.72529e-9; zero initial
history rules out accumulated state as the source of that policy difference.
See `cpu-policy-first-stage.json`. The old FFN reference additionally used GPU
FP32 RMS, unlike the new independent CPU FP64 RMS. All historical files remain.

Performance was measured with diagnostic collection disabled, one loaded model,
both full-request warmups and ABBAAB three trials per variant. Eight greedy
outputs give seven decode intervals; main matrices are resident. Control is the
frozen fast math/cache-off route. Candidate includes the new precision policy
and GPU attention cache, so gains do not isolate contraction control.

| Fixture/prompt tokens | Control tok/s | Candidate tok/s | Candidate prefill s | Candidate compute E2E s |
|---|---:|---:|---:|---:|
| English/5 | 5.788326 | 5.913721 | 0.794547 | 2.000625 |
| Chinese/10 | 5.742900 | 5.905712 | 1.579417 | 2.787944 |
| Long fixture/74 | 5.466262 | 5.781753 | 11.929394 | 13.162233 |

All IDs match saved Vulkan-validated fixtures; repeated prefix logits are exact
per variant. No regression was observed relative to the same-run frozen route.
Preceding contracted-build numbers below are separate runs; no isolated speedup
claim is made from cross-run differences. See `interleaved-noncontract/summary.json`.
Vulkan still measures about 25.6 tok/s with different activation arithmetic.
Both saved Vulkan and Rust decode rates use seven intervals for eight outputs.
Rust compute E2E covers state reset, prefill, LM head, CPU selection and decode;
it excludes model load, warmup, text tokenization and result-file/text output.
Vulkan request E2E includes local HTTP/SSE handling, so their E2E scopes differ.
The saved Vulkan requests supply identical raw prompt IDs, bypass prompt cache,
and use FP32 KV, B1, ubatch1 and context128.

## First divergence and historical comparison

The code fixture fails the historical 1e-3 absolute logit gate at generation step
47 (processed position 110, input token 830). Vocabulary entry 10137 differs:
historical reference 2.455742835998535 vs fast candidate 2.4539642333984375.

The first captured divergence is earlier: position 0, token 727, layer 0.
Input normalization and all four GDN projections are byte-identical. Prepared
GDN q/k/v/beta differ by at most 5.9604645e-8; core by 3.7252903e-9; gated
output/state by 2.3841858e-7. At the first position convolution history is zero,
so this initial v/beta difference does not require an FMA contraction explanation:
CPU and GPU native FP32 activation math already differ. Later convolution and
attention also have different contraction/reduction orders. Layer-0 state error
at position 110 reaches 1.7166138e-5. Raw tensors and logical state are saved in
`results/phase2/boundary-layer-trace` and `boundary-first-stage-trace`, including
`prepared-parts.json`.

**Audit correction:** the historical full-model reference was a mixed path with
`GPU_FFN=1, PERSISTENT_FFN=0`: GPU FP32 RMS, separate 128-lane GEMVs and GPU
SwiGLU. It was not a CPU-normalized FFN oracle. Its GDN preparation and attention
were on CPU, with the independently checked GPU Delta primitive. Neither path
uses GGML activation quantization. Thus Vulkan activation quantization cannot
explain this Rust/Rust discrepancy.

Partial precision changes did not satisfy that historical comparison:
FP64 RMS alone: code max 0.0015096664; RMS plus scalar GDN: 0.0024251938;
all scalar-order options: 0.0025873184. Their full reports remain under
`context-boundary-down`. Errors do not decrease monotonically when cancellation
and the reference's mixed precision differ. These are not passing results.

## Explicit numerical policy and independent reference

The new configuration defines FP32 activation/KV/state storage, FP64 norm sums
and inverse calculation rounded to FP32, and FP64 exp/log1p rounded to FP32
before subsequent FP32 operations. The first passing configuration intended
separate FP32 conv/attention multiply/add rounding, but the actual gfx1201 ISA
audit found that HIP rounding helper calls still contracted into FP32 FMA.
Those passing results describe the contracted build, not a bit-exact scalar
arithmetic contract. Its source is preserved in `canonical-contracted-source.zip`;
the kernel snapshot was reconstructed by reversing only the subsequent local
contraction-control patch. The actual original device code is also preserved in
`canonical-final-gfx1201.elf` and its disassembly.

The current experimental build uses local `#pragma clang fp contract(off)` and
plain multiply/add expressions. Independent mul/add instructions have been
observed in the new convolution ISA. Completed full validation is preserved under
`noncontract-boundary` and `noncontract-reset`, summarized above. Performance is
checked separately above. Choosing a specified FP64-to-FP32 policy does
not alter the threshold or erase the native-FP32 reference differences.
FFN keeps four warp-owned rows but uses four
accumulators per lane to reconstruct the original 128-lane projection tree.
There is no global barrier. CPU and GPU norm reduction orders may still differ;
byte-exact full-model logits are not assumed.

Opt-in flags:

```
AMD_INFER_CANONICAL_MATH=1
AMD_INFER_STABLE_RMS=1
AMD_INFER_SCALAR_GDN=1
AMD_INFER_SCALAR_ATTENTION=1
AMD_INFER_SCALAR_FFN=1
AMD_INFER_FFN_WARP_ROWS=1
AMD_INFER_PERSISTENT_FFN=1024
AMD_INFER_DOWN_VARIANT=1
```

`CANONICAL_MATH` is fixed at process startup. CPU uses Rust/libm FP64 functions;
GPU uses HIP/OCML FP64 functions and separately implemented rounding/order.
The numerical reference explicitly disables GPU FFN and GDN preparation and
GPU attention: CPU conv/L2/RMS/activation/attention with independently checked
HIP GEMV and Delta primitives. Delta remains the same checked HIP primitive in
both full graphs; the separate CPU Delta/state oracle remains available.
This is a precision-aligned hybrid reference, not a complete CPU or BF16 oracle.

Evidence for choosing this policy, rather than changing a threshold:

* All 64 FFN layers, four inputs each, 1,310,720 outputs: **max abs 0** against
  independent CPU norm/activation plus checked matrix primitives.
* At position 0/token 727/layer 0, raw projections, prepared inputs, core,
  gated output, recurrent/conv state, FFN and hidden are now byte-identical.
* The unchanged 1e-3 gate is checked on every logit at every generated step,
  with historical data kept separately. Generated-token equality is additional
  evidence, not the numeric gate.

## Prior contracted-build full-sequence and reset evidence

Same padded raw English/code/Unicode fixtures, 96+32/64+64/64+64 prompt/output
tokens, seeds 42/123, 127 processed positions per case, capacity 128.
160 outputs / 39,731,200 logit values against the precision-aligned reference:

| Case | Max absolute error | Max relative L2 | Failed steps |
|---|---:|---:|---|
| English | 0.000329256058 | 1.34882e-5 | none |
| Code | 0.000897347927 | 3.10984e-5 | none |
| Unicode | 0.000374972820 | 7.88781e-6 | none |

Original code step 47: max abs 0.000882148743. Strict gate remains **1e-3**.
Two additional complete trials per case: 320 sampled outputs, all per-step
logits byte-identical after full state resets; both trials pass the same gate.
This remains bounded context with only three fixtures/two seeds. It does not
qualify long contexts or eliminate every possible implementation bug.

Evidence: `results/phase2/canonical-boundary/candidate-error-analysis.json`,
launch JSON, per-step logits/tensors; `canonical-reset/summary.json`;
`canonical-all-ffn.log`, `canonical-ops.log`.

## Prior contracted-build E2E and hotspot evidence

One loaded model, both complete requests warmed, ABBAAB, three trials per
variant, 8 fixed greedy outputs, seven measured decode intervals. Comparison
includes the arithmetic configuration plus GPU attention cache (control cache
off, candidate on); it does not isolate FFN alone. Main matrix storage is fully
resident, 13,333,954,560 bytes. Embedding is CPU row read / HIP dequant / host
hidden; residual and attention preparation remain hybrid CPU operations.

| Fixture | Frozen-path control tok/s | Aligned candidate tok/s | Gain | Candidate prefill s | Candidate compute E2E s |
|---|---:|---:|---:|---:|---:|
| English 5 | 5.665119 | 5.795374 | 2.30% | 0.811767 | 2.039061 |
| Chinese 10 | 5.662498 | 5.793510 | 2.31% | 1.612994 | 2.843146 |
| Long fixture 74 | 5.464358 | 5.768826 | 5.57% | 11.920576 | 13.153206 |

All greedy IDs match the saved same-weight Vulkan-validated fixtures; repeated
prefix logits are deterministic per variant. Vulkan is still about 25.6 tok/s,
with different activation arithmetic. No CPU-offloaded vLLM victory claim.
Full sequence checks capture tensors/logits and are excluded from performance.

Final HIP event profile (12 forwards: 5 prefill + 7 decode, excludes warmup
because counters reset per request): FFN 0.615874 s,
GDN 0.503916 s, respectively 45.17%/36.96% of measured events. FFN parts:
norm 0.013371 s; gate/up/activation 0.373067 s; down 0.229435 s.
Previous profile gate/up was 0.491835 s, though profiles are separate diagnostic
runs; only the interleaved table supports an E2E gain claim. Parts overlap FFN
and are excluded from total-event time. Profile wall throughput is not headline
performance.

Requested peak allocation 13,526,636,696 bytes; measured minimum free
2,377,949,184 bytes in this profile. Requested bytes exclude driver rounding;
free includes desktop applications. Safety reserve is unchanged.

Evidence: `interleaved-scalar-arithmetic/{launch.json,summary.json,run}` and
`scalar-arithmetic-profile/{launch.json,summary.json,run.log}` under phase2.

## Reproduction and next work

In the coordinated WSL HIP/Rust environment:

```sh
cargo build --release --features hip,tokenizer --bins
python3 tools/check-canonical-boundary.py results/phase2/repro-boundary
python3 tools/check-canonical-reset.py results/phase2/repro-boundary results/phase2/repro-reset
python3 tools/interleave-ffn-layouts.py "$MODEL_GGUF" "$TOKENIZER_JSON" --scalar-arithmetic --root results/phase2/repro-perf
```

The correctness scripts use the existing local model/tokenizer paths and saved
fixture metadata; launch JSON records exact environment. Keep old reference
and fast-path logs separate. Next: broader seed/sequence coverage, then split GDN
projection/state timings and investigate gate/up resource/bandwidth limitations.
Move norm/residual and attention preparation onto GPU only with the same numeric
policy and repeated trajectory/reset gates. Whole-model megakernel remains
unfinished; no code is published.

## Gate/up grid experiment (same arithmetic)

The existing persistent row-task kernel uses four warp-owned rows per block;
increasing grid size reduces repeated row tasks per block without changing
quantization decoding, reduction tree, intermediates or storage. HIP already
limits the grid to 128..4096. An attempted 4352 configuration was rejected;
its failure log is retained. The new driver validates that existing range.

One-load, warmed ABBAAB comparison of 1024 vs 4096 blocks, three trials per
fixture, with both routes using the verified precision/cache configuration:

| Fixture | 1024 tok/s | 4096 tok/s | Change |
|---|---:|---:|---:|
| English | 5.768382 | 5.853634 | +1.48% |
| Chinese | 5.323219 | 5.355111 | +0.60% |
| Long fixture | 5.781650 | 5.827318 | +0.79% |

All prefix logits are byte-identical across both variants and all generated IDs
match. Separate full-model validation of 4096 now passes all 160 sampled steps
and 320 reset steps byte-for-byte against the saved independent same-policy
reference. See `gateup-grid4096-boundary` and `gateup-grid4096-reset`.
The 1024 source/performance snapshot remains preserved; 4096 is a verified small
optimization with only 0.6–1.5% observed E2E decode gains, not a major advance.
See `interleaved-gateup-grid4096`.

One-load warmed English ABBAAB event profiling confirms gate/up median
0.374439114 -> 0.351225522 seconds (-6.20%); whole FFN 0.617458753 ->
0.594339236 (-3.74%). Down 0.230019484 -> 0.229924070 and GDN 0.505254978 ->
0.505587717 are effectively unchanged. These 12-forward diagnostic windows
include 5 prefill + 7 decode calls, exclude warmup, and perturb throughput.
Profile throughput is not a headline E2E result. See
`interleaved-gateup-grid4096-profile/summary.json`.
Raw real-format payload analysis finds 5,609,553,920 gate/up bytes per forward:
66 IQ4_XS and 28 IQ3_S matrices among 128. The preceding profile implies about
180.44 logical compressed GB/s (5 prefill + 7 decode forwards, warmup excluded).
This is not a physical bandwidth measurement. See `ffn-actual-payload.json`.

The new per-layer FFN diagnostic verifies 64 layers / 4 inputs /
1,310,720 outputs with max error 0, using CPU norm/activation plus independently
checked matrix primitives. Post-first-call medians reveal that IQ4_XS/IQ4_XS
accounts for 25 layers and 32.96% of diagnostic gate/up time (194.54 logical
GB/s, 88 VGPR); IQ3_S/IQ4_XS for 11 layers and 20.29% (125.77 logical GB/s,
103 VGPR). Full format resource metadata shows 88–134 VGPR, zero spill/scratch.
This is a one-layer-at-a-time synthetic-input diagnostic, not full-resident E2E
or measured occupancy. See `gateup-format-profile-summary.json` and
`noncontract-resource-summary.json`. The next concrete target is decode/load
instruction scheduling on the IQ3_S/IQ4_XS path, compared with IQ4_XS/IQ4_XS;
do not infer a hardware bandwidth ceiling from these logical ratios.

## Allocation reconciliation for the new revision

All six interleaved profile records reconcile exactly with allocated source
dimensions (`noncontract-memory-accounting.json`):

| Category | Requested bytes |
|---|---:|
| Main packed matrix arena | 13,333,954,560 |
| GDN small weights plus FFN norm bank | 9,218,048 |
| GDN recurrent/conv history plus attention KV | 173,670,400 |
| Persistent GDN/attention scratch plus shared linear workspace | 9,771,008 |
| Runtime tracked live total | 13,526,614,016 |
| Peak temporary embedding Q3_K row + FP32 output | 22,680 |
| Runtime tracked requested peak | 13,526,636,696 (12.5977 GiB) |

The embedding is Q3_K, 5120 values: 20 blocks * 110 bytes + 5120 * 4 bytes.
The source-derived category split exactly matches tracked allocation totals;
it is not separate per-category hardware instrumentation. HIP observed minimum
free is 2,378,727,424 bytes. Requested counts exclude driver rounding and
context/code-object allocations; the free sample includes other desktop apps.
The reserve check and budgeted streaming fallback remain unchanged. Both grid
variants have the same allocations. Main residency excludes embedding/MTP,
and the runtime still performs host normalization/residual/attention preparation.
