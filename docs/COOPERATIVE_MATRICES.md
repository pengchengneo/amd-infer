# Phase10 cooperative quantized matrices

The accepted optional path extends exact-math cooperative loading to IQ3_S
gate/up and IQ4_XS/IQ3_S generic GEMV, including FFN down and projections.
`AMD_INFER_COOP_IQ3S=1` and `AMD_INFER_COOP_QUANT_GEMV=1` are added to the
phase9 configuration. Main defaults remain0. Formats, FP32 activations/KV/state,
dequant multiplication order, ordered FMAs and reduction tree remain unchanged.
Each row stays inside one wave; there is no cross-block barrier. Head fusion,
graphs and explicit stream remain0. No full-model persistent megakernel was added.

## Frozen uninstrumented comparison

One resident model,64 warm forward tokens, both requests warmed, ABBAAB three
trials/arm,8 fixed outputs/7 intervals, context512, same raw prompts. Compute
includes forward/LM head/full vocabulary read; wall includes selection. No ADL
polling ran during this final benchmark. All prompt/last logits and IDs match
the frozen phase9 policy byte for byte. TTFT and E2E medians improve for all cases.

| Prompt | Phase9 control tok/s | Candidate tok/s | Paired gain | Removed ms/token | Remaining Vulkan gap ms |
|---|---:|---:|---:|---:|---:|
| chinese | 16.446 | 17.717 | 7.73% | 4.36 | 17.50 |
| english | 16.454 | 17.750 | 7.88% | 4.44 | 17.26 |
| long | 16.339 | 17.611 | 7.78% | 4.42 | 17.73 |

The Vulkan column uses the preceding phase9 matched b11284 reference25.59-25.67
tok/s, not a new native measurement in this phase. Its short histories match
token IDs but independent logits remain different. The remaining gap is a
whole-request latency difference, not an exclusive per-module attribution.
Only this paired gain counts as phase10 algorithm improvement; the earlier
14.6-to16.4 absolute change is not credited here. Absolute drift remains unresolved.
One-shot GPU samples and actual library resolution/hashes are retained. There
is no earlier matched14.6 counter trace or demonstrated thermal cause.

## Budget and selection

The full decode-only observer covers FFN, nine GDN parts, seven attention parts
and LM head. It excludes prefill and initial prompt head. All-module instrumentation
raises total latency to roughly94-98ms versus about56-61ms without instruments.
Its event windows contain enqueue/scheduling gaps. Nested FFN/GDN containers
must not be added; GDN's whole container includes nine observer reads/layer.
These numbers rank work, and do not exactly split the17-22ms native gap.

| Diagnostic window ms/token | Phase9 control | Candidate |
|---|---:|---:|
| FFN_gate_up | 20.96 | 19.82 |
| FFN_down | 12.75 | 11.06 |
| GDN_qkv | 7.15 | 6.63 |
| GDN_z | 3.32 | 3.03 |
| GDN_output_projection | 5.35 | 5.08 |
| attention_q | 4.03 | 3.95 |
| attention_core | 0.48 | 0.50 |

Gate/up remains the largest window. The64 actual-layer micro screen gives
5.19% gate/up and14.88% down speed improvements; full-model paired timing is
the acceptance criterion. IQ3_S loads64 codebook words/tile across32 lanes and
distributes their bytes with wave shuffles. IQ4_XS reuses the phase9 packet and
register-codebook design in generic GEMV. This reduces duplicate instructions,
not the unique matrix payload. The matrix payload is13,333,954,560B/forward;
IQ3_S plus IQ4_XS account for66.24%. Candidate generic kernels use37/50 registers
for IQ4/IQ3 and zero LDS/scratch; occupancy fields are runtime estimates.

Short-context attention core is about0.50ms. Near512, with447 prompt tokens and
64 teacher outputs reaching position509, attention-only instrumentation reports
core3.11ms/token. The latter is a different observer scope; projection numbers
from it must not be compared directly with the all-module trace. Eliminating
all short core work would conditionally save under1% of total latency; eliminating
the near512 core would save about5% in its instrumented request. Neither is an
achievable kernel assumption. Long attention deserves a later layout screen,
while short-prompt throughput remains dominated by quantized matrices.

LM head is Q5_K,874,086,400B, and this phase does not change it. Uninstrumented
LM-head host windows are about2.42ms/token;
even a zero-cost head would only remove roughly4% of current total latency.
GDN delta plus norm/gate is about3.36ms in the full observer. A local persistent
kernel can only remove part of that window, and the prior head-fusion candidate
regressed prefill. This evidence did not justify another persistent implementation.

The next recommended bounded screen is FP32 GEMV load/block scheduling with
the existing numerical policy, followed by long-attention layout. Upstream
Vulkan uses grouped codebook values/block-scale dot products; CUDA/HIP MMVQ
uses Q8 activation dots and type-dependent RDNA4 warps. Adopting their arithmetic
is a larger numerical choice. No Q8 activation or changed weight/KV format is
enabled here; present its quality/memory evidence for a user decision before
adopting such a policy. Inspected pinned-source hashes and MIT attribution are
retained in `results/phase10/payload-and-references.json` and `THIRD_PARTY.md`.

## Quality, memory and real generation

- 38 format/shape groups x5 patterns,760 selected outputs versus untouched GGML
 CPU dequant plus the explicit FP32 FMA oracle: max_abs0.
- 64 actual FFN layers x8 inputs,2,621,440 values: CPU RMS/SwiGLU plus
 validated GEMV reference,max_abs0. Phase11 audited that the standalone strict
 arm-bit guard then covered coop-gate-up,not coop-matrix;see FP32_REGROUP.md
 for the corrected guard and rerun. Whole-model bit checks below remain valid.
- Original724 long/reset steps and480 sampled comparisons: max_abs0, original
 0.001 gate unchanged; highest processed position509. Existing independent
 long-history divergence remains visible. Sampling uses the same Rust policy.
- New128/256 fixed-output runs compare every full248320-value step:384 extra
 comparisons, max_abs0. The real greedy CLI emits256 tokens with identical
 text/IDs/final logits; it stops at EOS or the requested limit.
- Invalid output513, unsupported context129 and prompt+output overflow are
 rejected before model open. Supported capacity remains128/256/512; no dynamic
 cache growth or HTTP service is added. A nonempty prompt must share the capacity
 with outputs, so512 context does not mean512 available output tokens.

The CLI now writes its final vocabulary once, while text and token files continue
to stream. Its256-output candidate generation wall time is15.66s versus16.82s
control, including host selection and file writes, without benchmark warmup.
Model loading is about45s separately. These are usability diagnostics, not the
frozen17.7tok/s headline. Raw prompts own their chat template; no automatic
template/hidden thinking-mode substitution occurs.

Requested GPU peak remains13,579,250,072B. Candidate diagnostic min-free is
2,360,705,024B, above the2,164,260,864B reserve. Requested bytes exclude driver
rounding; measured free includes desktop applications. No other process was
stopped, and no driver/security/power setting was changed.

## Recoverable local deliverables

`results/phase10/best-config.json` and `matrix-runtime` freeze binaries/library
with hashes. The library was selected by each executable's actual dynamic
resolution, not directory glob order; a stale build variant remains untouched.
Cargo now watches the new attention observer header; its rebuild reproduced
the exact HIP library hash. Source snapshot/hashes and checks are retained.

From the existing repo in AMD-Infer with the established ROCm environment:

```sh
python3 tools/run-best-matrix.py results/phase2/cases results/phase10/my-replay
python3 tools/run-best-matrix.py results/phase2/cases results/phase10/my-rollback --rollback
python3 tools/run-generation.py PROMPT.txt results/phase10/generated.txt --max-new-tokens 256 --capacity 512
```

Replay output paths must be new; generation outputs belong under project results.
The rollback selects unchanged phase9; older phase8/phase6 entrypoints and hashes
are also preserved. Results remain local. No commit, GitHub publication or upload
was performed. See `results/phase10/summary.json` for the new evidence only.
