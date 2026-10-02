# Independent upstream quality probes

This records the phase3 snapshot. GPU256/512 cache execution is added and
validated in the later `GPU_CONTEXT_EXECUTION.md`; old raw results remain.

The phase3 suite uses the exact existing GGUF, local official tokenizer/chat
template, `enable_thinking=False`, raw prompt IDs, greedy selection and no
repetition penalty. Upstream is official llama.cpp Windows Vulkan build b11284,
commit 25747b08e, one slot, ubatch1, FP32 KV, flash attention off, context256.
The launch and zero-request stop record are saved. Rust and upstream never hold
their full models on the GPU simultaneously.

The three measured prompts contain 156 English, 152 Chinese and 145 code tokens.
Each requests 64 outputs. Repeated factual notes are recorded in the preparation
script; this is a task probe, not a public accuracy benchmark. The maximum
processed positions are measured separately from output length. A 64-token limit
can truncate prose or code and does not establish completion-level quality.

The Rust extension to context256 uses CPU attention preparation and CPU KV for
all positions; GPU Delta/FFN and quantized matrix operations remain enabled.
The default GPU attention/cache stays capped at128. The extended diagnostic
constructor and forward path reject attempts to select that GPU attention path.
It does not qualify GPU long-context performance or contexts beyond this test.
Capacity512 is accepted only for the same CPU attention diagnostic route; it
has not been exercised here and is not a validated512-position result.

`tools/run-upstream-quality.py` pins the previously verified stable HIP library
and saves its SHA256 and executable SHA256, full launch environment, exit status,
per-step full-vocabulary Rust logits and generated IDs. Free generation and
teacher-forcing are separate runs. Teacher-forcing feeds the upstream's actual
token at each step, while retaining the Rust greedy choice before forcing.

Upstream `/completion` stores top128 pre-sampling log probabilities for every
step. These are normalized with the full upstream vocabulary, but upstream raw
full-vocabulary logits are unavailable through this API. Rust log probabilities
use its full logits and a double-precision log-sum-exp. The analysis reports
top-k overlaps, top128 score correlation, normalized log-probability differences,
greedy agreement on identical histories, and the first free-generation token
divergence. Correlation alone is insufficient; score differences are retained.
Different reduction/scale ordering and precision can contribute to differences.
The earlier assumption of Vulkan activation quantization was not established;
the inspected local IQ4_XS Vulkan shader uses floating activation vectors.

Only matching histories are used for numerical comparisons. Free-generation
logits after the first token divergence are not numerically compared. Text is
decoded from the saved IDs with the official tokenizer, preventing HTTP-client
Unicode decoding artifacts from being mistaken for model text errors.

Results and per-step evidence live in `results/phase3/quality-long/comparison.json`.
The completed long suite gives:

| Case | Prompt / output | Highest actual position (zero based) | Matching free prefix | Teacher greedy agreement | Mean top128 overlap | Mean score correlation | Max top128 log-prob difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 156 / 64 | 218 | 64 | 64/64 | 99.15% | 0.999884 | 0.224954 |
| Chinese | 152 / 64 | 214 | 20 | 63/64 | 98.96% | 0.999791 | 0.446258 |
| Code | 145 / 64 | 207 | 64 | 64/64 | 98.71% | 0.999415 | 1.648613 |

The Chinese step20 divergence is a rank inversion: upstream chosen token109574
is Rust rank2; upstream top1/top2 margin0.094908, Rust reversed margin0.063660.
Both decoded continuations discuss the pressure/boiling mechanism coherently,
but wording then differs. This observation does not isolate a specific kernel
or establish the cause of the independent arithmetic difference. English's
two complete opening sentences are sensible and identical, but do not finish
the six-sentence task. Code ends inside its docstring; identical partial code
does not pass syntax/function validation. Every long response hits the64 limit.
Mean top128 log-prob RMSE is0.048291/0.056809/0.084531 respectively. Differences
are retained rather than declaring numerical equivalence from high correlation.

Each Rust run remains finite throughout all192 decoded steps and processes
219/215/208 total positions; full matrix residency13,333,954,560 bytes is used.
CPU attention/cache context256 is excluded from claims about GPU long-context
performance. Requested GPU peak13,508,679,832 bytes and observed minimum free
2,402,070,528 bytes (over all six long requests) retain the reserve. CPU KV
memory and driver rounding are not part of those requested GPU byte counts.

Reproduce by first starting the saved native server, preparing fixtures,
requesting raw IDs with the saved `.request.json`, recording idle metrics and
stopping that owned server, then sourcing the existing ROCm environment and
running the two Python quality tools. The test is serial and leaves other apps
untouched. Historical phase2 reference failures remain retained independently.

## Supplemental complete code probe

The supplemental complete short code case uses37 prompt tokens and24 outputs,
reuses the existing upstream response, and matches all24 token IDs including
EOS. Teacher-forced greedy agreement is24/24, mean top128 score correlation
0.999495, mean log-prob RMSE0.083623 and max difference0.982787. Both complete
`triangular(n)` functions parse and pass eight nonnegative inputs from0 through
1,000,000, with only a restricted arithmetic AST executed. This validates that
small task only; the long `summarize_scores` function is still truncated and
unvalidated. See `quality-code-complete/comparison.json` and `function-check.json`.

## IQ3_S packed table load experiment

The opt-in `AMD_INFER_IQ3S_PACKED_GRID=1` branch replaces an IQ3_S four-byte table
entry's byte-addressed load with an aligned32-bit load and byte extraction.
Only the11 G21/U23 FFNs are selected. Quantization and multiplication/reduction
order are preserved. This experiment is not a full-model megakernel.

All64 real FFNs, four inputs each, 1,310,720 outputs pass against CPU norm/activation
and the previously independently checked matrix primitive: max error0. One-load,
both routes fully warmed, ABBAAB with three trials each and8 greedy outputs:

| Fixture | Default tok/s | Packed tok/s | Change | Default / packed compute E2E seconds |
|---|---:|---:|---:|---:|
| English | 5.855798 | 5.859382 | +0.061% | 2.018529 / 2.019410 |
| Chinese | 5.843214 | 5.856403 | +0.226% | 2.816326 / 2.807733 |
| 74-token fixture | 5.927655 | 5.925802 | -0.031% | 12.805826 / 12.833086 |

All prefix logits are byte-identical across routes and repeated resets, and all
generated IDs match. Timing is7 decode intervals for8 outputs, includes CPU
selection; compute E2E excludes load/warmup/tokenization/file IO. This short
performance fixture is separate from the actual219-position quality test.
No convincing E2E benefit is established; the original byte-load path stays
default and the experiment remains opt-in. Do not describe the diagnostic
FFN pass or tiny timing variation as a throughput improvement. Raw launches,
per-stage events, all trial records and full summary are under
`results/phase3/iq3s-packed`. Stable runtime copies and phase2 snapshots remain.
