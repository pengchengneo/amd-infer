# Phase12: independent FP32 quality and activation-quantization diagnosis

The four native/HIP greedy disagreements in the 512-capacity story are primarily caused by the pinned Vulkan backend quantizing eligible FP32 inputs to Q8_1. Disabling MMVQ before the native process starts removes all four disagreements. No HIP arithmetic, quantized weight format, precision contract, quality threshold, recommended flag, or frozen runtime changed in this phase.

## Independent quality evidence

Same GGUF SHA256 `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`, actual prompt IDs and teacher IDs, context512, batch/ubatch1, FP32 KV/state, FA/MTP disabled. Pinned llama.cpp b11284, local commit `25747b08e7a0f9a59a2089ce6b98d2229b76042a`, existing native DLLs, retained MIT notices. The independent model graph and kernels use no AMDInfer implementation. The CPU model in earlier self checks shares HIP primitives and is not the independent full-model oracle here.

| Unobserved FP32 native comparison | Stable HIP control | Existing phase11 opt-in |
|---|---:|---:|
| Story free greedy IDs | 469/469 | 469/469 |
| Story teacher greedy IDs | 469/469 | 469/469 |
| Full-vocabulary maximum absolute error, each trajectory | 0.0003790855 | 0.0007534027 |
| Original0.001 gate, each trajectory | 469/469 pass | 469/469 pass |
| Minimum top40 overlap, each trajectory | 40/40 | 40/40 |

The prompt is43 tokens and the469 output limit fills512 capacity; highest processed position510. This is a truncated story without EOS, not a completed1000-word task. All native/HIP input histories agree. Independent sampling adds3 cases x3 trials x32 outputs:288 IDs match, and all18 saved prompt/last full-vocabulary boundaries pass0.001. Twelve project-authored bilingual arithmetic, Python, logic, JSON and reading fixtures have independently computed or explicit ground truth: all configurations score12/12; prompt IDs match. These small diagnostics do not establish general accuracy, statistical quality, or SOTA capability. Prior724/reset/480 checks remain backed by unchanged frozen runtime hashes; they were not rerun as new independent coverage.

Temperature1 full-vocabulary probability comparisons have no top-k truncation. Against production default-MMVQ native, maximum total variation was0.0552142, at story step191. Against independent unobserved MMVQ-disabled native, stable HIP maximum TV is0.00000413063 and opt-in maximum TV0.00000593059; opt-in mean KL(native||HIP) is2.3630e-12. This supports arithmetic agreement for the tested histories, not bit identity or universal model equivalence.

## The first branch and local primitives

At output index222 (the223rd token, processed position264), HIP chooses token78831, ` Ellison`; default native chooses23720, ` Chen`. Each token ranks second in the other backend. HIP margin0.00631714; default native margin0.00162697. Their chosen probabilities are approximately0.29520 and0.29401. This is a near-tied fictional surname, so branching alone does not establish poorer task quality. Step191 still had a material5.52% distribution shift and required investigation. The3.3325 largest raw-logit error at step51 belongs to ` Fa`, native probability6.37e-13; it was not treated as a quality verdict.

Native `ggml_vk_should_use_mmvq` selects Q8_1 activation matvec for eligible AMD cases; Q8_0 weights and Q6_K have different selection rules. `quantize_y` and `quantize_q8_1.comp` establish the actual activation format. Embeddings match exactly at all11 captured positions. On the first layer, norm inputs match to FP32 rounding, Q8_0 alpha/beta projections differ around1e-6, but quantized QKV/z already differ around0.02. Untouched upstream GGML CPU dequantization plus independently written FP64 row dots match HIP QKV within3.63e-6 across selected rows at7 positions. Q8_1 input simulation explains most native error, including five positions reproduced near1e-6; two positions retain rounding-boundary residuals up to0.00183. It is a diagnostic simulation, not a bit-exact recreation of the native quantizer. The controlled full-model MMVQ toggle provides the stronger causal evidence.

An independent FP64 DeltaNet specification checks actual inputs and previous states, not a shared HIP oracle. HIP16 cases across4 layers and positions0/1/264/265: max post-state3.14e-6, core1.22e-6. Native28 cases: max post-state2.32e-6, core5.00e-7. Head mapping is `head%16` in both implementations and state layout is [head][value][key]. The GGUF RoPE base10000000,64 rotated dimensions, sections11/11/10/0 and RMS epsilon1e-6 agree with the textual path. There is no demonstrated indexing, reset, cache or DeltaNet-state bug in these probes.

## Observers and reproducibility

New default-off `AMD_INFER_DEVICE_TRACE=1` enables readbacks of device hidden, FFN stages, DeltaNet stages/state and attention preparation/cache at selected positions. It changes no kernels or state updates. All469 full-vocabulary HIP teacher outputs are byte exact against the unobserved frozen opt-in trajectory. Device buffers and memory reservation are unchanged.

Native callbacks can change graph fusion. Default-MMVQ callback trace differs from unobserved native by up to3.32055 and changes one argmax at step353. It is retained only for primitive diagnosis. Under effective MMVQ disable, callback max effect3.0041e-5 and no argmax changes; final quality always uses unobserved runs. Native strided tensors are read using validated pinned ne/nb metadata. Conv history compares the post-update source view `conv_state_last`, not the pre-copy destination buffer. Layer comparison covers64 layers x11 positions. The64-layer logit lens uses118 independently dequantized head rows at3 positions; its ranks are explicitly within a final top40 union, not full-vocabulary early-layer ranks or early-exit accuracy.

On Windows, setting `$env:GGML_VK_DISABLE_MMVQ` inside an already-running PowerShell process did not affect the native CRT environment copy. The first two nominal FP32 attempts reproduced default results byte for byte and are rejected in `ineffective-control-rejection.json`. Accepted directories end `startup-env`: the parent sets the variable before spawning the child. No DLL or driver settings change. Native result-directory misplacement was corrected by moving only the newly generated directory, with provenance recorded.

## Fresh serial performance and limits

All GPU jobs run serially. Headline rates exclude loading, tracing, file writes and sampling; native compute includes forward/LM-head/full-vocabulary read. Same raw cases,8 outputs,64 warm tokens plus per-case warm,512 context:

| Case | Native FP32 compute tok/s | HIP control | HIP opt-in | Opt-in gap ms/token |
|---|---:|---:|---:|---:|
| chinese | 24.5302 | 16.1564 | 20.7804 | 7.3563 |
| english | 24.3700 | 16.1716 | 20.7428 | 7.1754 |
| long | 24.4962 | 17.5329 | 22.4354 | 3.7497 |

Short-case ABBAAB arms are stable and retain28.27–28.62% gain from the existing opt-in. Long8 control changes regime16.04->17.53 within the sequence, so its median ratio is not accepted as stable evidence. An additional64-output run has stable English+28.12% and long+27.94% pairs; Chinese arms are unstable and retained. No new speedup is claimed in phase12. The prior default-native25.6–25.8 target uses Q8_1 activations, while this fresh activation-FP32 reference is24.37–24.53. They are separate serial runs, so their rate difference is not a precise isolated quantization-speed attribution.

The remaining same-length3.75–7.36ms cannot be assigned wholly to DeltaNet. Existing event-instrumented budget prioritizes FFN gate/up, down and quantized projections, but observer overhead makes its absolute totals unsuitable for subtracting from these uninstrumented gaps. Absolute HIP short/long regimes remain unexplained; there is no new evidence proving clock, thermal or driver causality. No clock/power/driver/security setting changed. Physical reserve2GiB+16MiB and native256MiB WDDM headroom gates pass in every run.

## Decision and saved evidence

Retain FP32 activations/accumulation and original weight formats. Keep phase11 opt-in and phase10 stable fallback, with phase9/phase6 hashes preserved. Implementing a Q8 activation path would change the precision/quality contract and requires a separate user decision; it is not implemented. Wider independent accuracy benchmarks and controlled regime/remaining-kernel attribution remain future work.

Primary evidence: `results/phase12/summary.json`, `fp32-independent-quality.json`, `native-fp32-comparison.json`, `independent-primitives.json`, `hip-independent-gdn.json`, `layer-logit-lens.json`, `matched-hip8/summary.json`, `matched-hip64/summary.json`, native observer reports and original tensors/logits. Project-authored reproduction scripts are saved under `tools/phase12-*`. Native wrappers require the existing b11284 runtime and process-start environment control; they do not redistribute native DLLs. `final-source.zip` and manifest preserve this source state. No external publishing occurred.
