# Phase13: FP32 GDN column state and head fusion

The opt-in candidate reduces decode latency by3.06–3.35% in two frozen ABBAAB cycles, saving1.31–1.44ms/token. Its engine default remains off. No whole-model megakernel or across-token persistent execution is implemented.

## Scope and implementation

`AMD_INFER_GDN_COLUMN=1` changes the physical recurrent state from `[head,value,key]` to `[head,key,value]`. Adjacent value lanes now access adjacent FP32 elements; each lane retains both original128-key accumulation loops. The state footprint remains3MiB per GDN layer/144MiB across48 layers. `AMD_INFER_GDN_HEAD_FUSE=1` keeps the head normalization and gate in the same128-thread block, preserving FP64 norm reduction, FP32 state and scalar arithmetic. Each block owns disjoint heads; all barriers are block-local with uniform head loops. The48-block configuration is measured here;24/32-head loops remain experiments, not promoted settings.

The Rust pipeline captures its column flag at construction/reset and passes it explicitly through new layout-aware C ABI entrypoints. A mid-request environment change cannot reinterpret live state. Diagnostic snapshots transpose only the state portion back to canonical row order and leave convolution history unchanged; ordinary decode does not perform this transpose. The graph cache key includes layout, although local graphs remain off. Legacy C ABI paths preserve row behavior.

## Frozen performance

Model SHA256:40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199. RX9070XT/gfx1201,ROCm7.2.3,WSL AMD-Infer,Rust release;native pinned b11284. Same512context,FP32KV/state,batch1,256warm tokens,64fixed outputs/63decode intervals,one full warm request per arm. Load,warmup and diagnostics excluded from headline decode. Native compute and selection wall rates are saved separately. GPU jobs ran serially.

| Case | HIP control before / after native | HIP candidate before / after native | Native FP32 median | Native default median |
|---|---:|---:|---:|---:|
| chinese | 22.63745 / 22.52857 | 23.32966 / 23.27879 | 24.15156 | 25.22663 |
| english | 22.65309 / 22.57040 | 23.39155 / 23.32710 | 24.10311 | 25.20368 |
| long | 22.40201 / 22.35399 | 23.13479 / 23.08075 | 24.15924 | 25.52666 |

All paired HIP arm spreads are below0.6%. The older same-FP3224.37–24.53 reference and default~25.7 remain recorded in `performance.json` with their original8output/64warm protocol. The fresh64output/256warm native FP32 trials span24.09–24.44 and default MMVQ25.18–25.54. Do not merge those regimes or conceal arithmetic differences.

## Remaining gap and bounds

The measured full-model GDN saving is1.31–1.44ms/token. Candidate versus fresh same-FP32 native remains roughly1.3–1.9ms/token; the native FP32-to-default MMVQ arithmetic difference adds roughly1.7–2.3ms/token. `performance.json` calculates each case and cycle directly, with no claim of native per-module physical attribution. The original3.75–7.36ms range involved shorter earlier runs and absolute drift; it is retained as a prior regime, not asserted as the current residual gap.

A direct ABI residual→D2D→stable RMS chain profile settles around12.8us/layer after a62us cold transient; even eliminating all64chains gives only0.819ms/token, an unattainable zero-cost upper bound. The cached synthetic full GDN profile suggests~1.90ms across48layers for column+fusion, but its row controls had19–37% spread. The actual model gain, not this synthetic number, selects the candidate. Quantized GEMV remains the large target in the existing module budget; observed event-window sums include launch/observer gaps and cannot be subtracted from unobserved decode latency. Unified warmup produces stable paired ratios but does not fully explain absolute drift. No power,security or driver settings were changed.

## Quality and recovery

The interrupted `quality-724` directory is preserved and excluded. `recovery.json` records actual Windows+WSL recovery, no surviving inference task and unchanged frozen hashes; `quality-724-resumed` replays the incomplete suite. Original0.001absolute gates remain unchanged:724saved CPU-oracle comparisons,768phase11-self comparisons,reset and256/512context coverage;480sampling comparisons are bit-exact between candidate and the current same-FP32 control. Actual-weight GDN CPU checks compare4,718,592values(max1.043e-7);1/3/4/64layer device checks include exact state digests,mid-request flag changes and forced fallback.

The469-token story is rechecked in free/teacher mode and CLI, reaching highest processed position510 within512context. Candidate full-vocabulary outputs are bit-exact against frozen phase11. The pinned independent phase12 FP32 Vulkan469-step oracle is reused transparently; phase13 freshly replays the HIP candidate against it. The36new external/fixed/long cases also receive fresh independent native FP32 execution:24evenly spaced BIG-bench object-counting inputs,8computed/explicit fixed answers and4long lookups at428prompt tokens. Prompt IDs,all8output IDs and288full-vocabulary boundaries are compared.

The36case accuracy is {'control': {'correct': 24, 'total': 36}, 'candidate': {'correct': 24, 'total': 36}, 'native_fp32': {'correct': 24, 'total': 36}}; independent maximum absolute error=0.000144958496, minimum top40 overlap=40. The469step independent maximum absolute error is 0.00075340271. Exact outputs and this small adapted task set are diagnostics; they are not a complete BIG-bench score,SOTA demonstration, universal numerical equivalence or comprehensive quality guarantee. Evaluate uses fixed lengths and ignoresEOS; answer scoring truncates before the first EOS.

## Reproduce and roll back

From the repository with the established WSL environment, use `python3 tools/run-best-column.py CASES results/NEW-RUN --new-tokens 64 --capacity 512`. `--rollback` selects the unchanged frozen phase11 runtime. The config and runtime hashes are under `results/phase13/experimental-config.json` and `column-runtime`. Original phase10/9/6 fallbacks remain unchanged. Source and all new scripts are archived locally; no GitHub publication was performed.

Primary data source: https://github.com/google/BIG-bench/blob/cbbbbaa0a279c7e066ec60b443bc33f6812fbacb/bigbench/benchmark_tasks/object_counting/task.json . Its Apache-2.0 license and source notices are retained in the local source snapshot.
