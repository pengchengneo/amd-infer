# Phase11 FP32 grouped quantized GEMV

Validated opt-in only. `AMD_INFER_FP32_REGROUP=1` changes FP32 association for IQ4_XS/IQ3_S, including compatible fused gate/up pairs. GGUF weight formats, activation/KV/state precision, context capacity, memory reserve and canonical defaults are unchanged. Flag default0, exact phase10 and phase9 runtime hashes remain recoverable.

## Measured outcome

Same process64-token warmup, case/arm warmup and ABBAAB; load excluded, reset included; all three retained trials per arm. Repeated after the independent native run with the same frozen runtime. Absolute control drift is retained; ratios are the algorithm evidence.

Two initial runs reproduced roughly28% gains in all cases. The post-native short cases moved together from17.7/22.8 to16.1/20.6,still about28%; its long case switched regimes within ABBAAB and its17.6% overall median is retained but cannot establish a stable ratio. A subsequent confirmation is preserved as well. No slow records are removed. The table below describes the pre-native stable regime,not a guaranteed sustained22.8tok/s; its5ms native gap expands toward9.5ms in the slower regime. Absolute-speed uncertainty remains a separate unresolved problem.

A development-only argument validator permits12 balanced trials in one loaded model. Both cycles are retained; the later stable long cycle measured17.568 control and22.477 candidate tok/s,+27.94%,with each arm spread below2%. Its exact logs and the unchanged earlier driver source are retained. The production frozen runtime is untouched; no compute/state/timing path was altered by this driver extension.

| Case | Control tok/s | Regroup tok/s | Paired gain | Fresh native tok/s | Gap ms/token |
|---|---:|---:|---:|---:|---:|
| chinese | 17.765 | 22.800 | 28.34% | 25.767 | 5.050 |
| english | 17.767 | 22.799 | 28.32% | 25.753 | 5.031 |
| long | 17.630 | 22.529 | 27.79% | 25.689 | 5.459 |

TTFT/E2E, every trial and the post-native repeat are in `results/phase11/summary.json`. Native uses the same hashed GGUF, raw prompt IDs,512 context, FP32 KV/state,B1/ubatch1,FA off,MTP off,64 warm tokens and pinned b11284; forward/head/read and host selection wall are reported separately. The current SOTA goal is still unmet. Remaining latency is an aggregate difference; without native module traces it cannot be assigned exclusively to GEMV, state access or host submission.

## Why this candidate

Seven original-order variants tested row counts8/16/2, byte-permute IQ4 codebook, IQ3 LDS codebook, uint2 reads and packet prefetch. Every screen output was bit-identical. The best cold generic proxy was rows2,+2.67%; actual64-layer FFN gate/up slowed2.51% and whole-model paired rates slowed0.52–0.69%, so it was rejected. The other weighted cold gains were at most0.35% or negative; no stable full-model benefit was claimed.

Pinned public Vulkan sources were inspected locally; hashes, MIT license and commit are in `upstream-reference-comparison.json`. Grouped scale application and packed/vector reads inspired the design comparison. The project implementation uses32 lanes per row,8 contiguous values per lane,two float4 inputs,8-value FP32 inner dots,one outer scale FMA and a fused gate/up projection. No shader implementation was copied. No inter-block/global barrier is used.

The full-output grouped micro proxy was1.76x hot/1.83x cold,max_abs2.62e-6; this was a candidate screen, not the model rate. The independent38-format/shape CPU dequant/FMA oracle passed,max_abs9.69e-8. Actual64-layer FFN paired diagnostic error was2.38e-6; gate/up and down rate gains were40.25% and55.85%, respectively, under event instrumentation. Full-model paired results above are the acceptance evidence.

## Fixed quality and longer generation

Original724 long/teacher/reset oracle comparisons and480 sampled comparisons passed the unchanged0.001 gate; an additional384 full-vocab steps passed. New469-step free and469-step teacher trajectories passed against the frozen control. Across these self checks, worst full-vocab error was0.000587463379; tokens remained identical and every top40 set had40/40 overlap. The512-context long-prompt checks reach509; the43-token story plus469 outputs reaches processed position510. These are bounded fixtures, not a guarantee for unseen prompts or larger contexts.

The actual CLI produced469 matching text/IDs,without EOS. Generation wall including selection/streaming writes was29.035s control and23.512s candidate; load was about45s separately. The story is truncated at the requested token limit and does not complete1000 words. `quality-long` retains text,metrics,all vocab steps and teacher evidence.

`native-comparison.json` reports independent free-history first branching and every same-prefix teacher logit/top40 error,argmax agreement and0.001 pass count for both arms. Existing independent upstream differences remain; numerical equivalence and SOTA quality are not claimed. Self cumulative error gates have not been widened.

The new independent story diverged at output index222. On all469 fixed-prefix teacher steps,control/candidate both agreed with native argmax465 times; top40 overlap fell to32/40. Worst full-vocab errors were3.332515/3.332699 and top40 errors1.903432/1.903490;none of469 native comparisons passed0.001. This extends the measured upstream discrepancy beyond the earlier shorter fixtures. Regroup did not change the four disagreeing argmax positions or observed free output,but this is not a solved upstream-compatibility issue.

The strict standalone FFN bit-check branch was audited: it previously covered coop-gate-up but not coop-matrix. Phase11 includes coop-matrix/gemv-tune and rechecked64 layers with mode6,max0. Existing phase10 full-model bit checks remain valid. FP32 regroup uses the original numeric gate,not the bit check.

## Recovery, memory and next decision

Requested model peak remains13,579,250,072B. Measured HIP diagnostic free minimum was2,332,803,072B; native global physical-free minimum was2,205,155,328B. Both retain the2GiB+16MiB reserve; native also checks256MiB WDDM budget headroom. Synthetic cache scrub allocations are development-only and do not enter model execution. Read-only GPU bookends are retained; no process was stopped or driver/security/power setting changed.

From the existing repo/ROCm environment, opt in explicitly:

```sh
python3 tools/run-best-regroup.py results/phase2/cases results/phase11/my-replay
python3 tools/run-best-regroup.py results/phase2/cases results/phase11/my-control --rollback
python3 tools/run-best-matrix.py results/phase2/cases results/phase11/my-phase9 --rollback
python3 tools/run-generation-regroup.py PROMPT.txt results/phase11/generated.txt --max-new-tokens 469 --capacity 512
```

The profile in `budget-regroup` ranks residual kernels but its event windows include observer/scheduling costs and exceed uninstrumented latency. Next work should obtain comparable native module evidence and target the remaining format/output-head or submission cost with a bounded paired experiment. Removing the unchanged observed head window is only an optimistic local bound,not enough evidence to assign the whole remaining gap. Further exact-order micro sweeps already failed to improve the full model in this phase. All deliverables remain local; no commit,push,publication or DLL redistribution occurred.

Current instrumented windows:FFN gate/up15.10ms,down7.73ms,GDN qkv5.17ms,out4.29ms,attention q3.39ms,head host2.27ms. Their sum is not a steady-state budget:the fully instrumented request takes74–75ms versus44–49ms uninstrumented. Even an ideal elimination of that observed head window would be below the aggregate5–10ms native gap;it is an optimistic local screen,not a promised achievable2.27ms gain. The highest-value next decision also needs independent upstream state/activation tracing at the first branching and worst-error positions,because the expanded3.33 logit discrepancy remains unresolved before any public-quality equivalence claim.
