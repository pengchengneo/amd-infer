# Phase9 cooperative FFN gate/up

The accepted optional configuration uses wave-owned IQ4_XS packet words and a
wave-register codebook. Both COOP flags must be1. Other formats keep their prior
decoder. All dequant multiplications, the two ordered FMAs per accumulator,
128-lane reconstruction and SwiGLU remain unchanged. Main defaults stayoff;
head fusion and graphs stay0. No cross-block barrier or full-model megakernel.

Frozen runtime: `results/phase9/gate-runtime`; configuration:
`results/phase9/best-config.json`. `tools/run-best-gate.py CASE_DIR RESULT_DIR`
replays it; `--rollback` selects unchanged phase8. The existing phase6 recovery
entrypoint and hashes are also preserved. Do not overwrite existing result paths.

## Uninstrumented frozen comparison

One resident model,64 forward warm tokens, both complete requests warmed,
ABBAAB three trials per arm,8 fixed outputs/7 intervals, raw identical prompts,
context512, FP32 KV/state. Decode compute includes forward, LM head and full
vocabulary read, excludes CPU token selection; wall and E2E are also retained.
No periodic ADL observer ran during this headline measurement.

| Prompt | Control tok/s | Candidate tok/s | Gain | Current Vulkan tok/s | Remaining ms/token gap |
|---|---:|---:|---:|---:|---:|
| chinese | 15.887 | 16.422 | 3.37% | 25.675 | 21.94 |
| english | 15.884 | 16.403 | 3.27% | 25.591 | 21.89 |
| long | 15.773 | 16.291 | 3.28% | 25.609 | 22.34 |

All TTFT and E2E medians improved. Prompt/last full-vocab logits and generated
IDs were bit exact versus preserved policy. Full trial arrays, timing boundaries
and version hashes are in `results/phase9/frozen-ab64`.

## Why this route

IQ4_XS occurs in66 of128 gate/up matrices and about56% of their unique packed
payload. An aligned word per lane cooperatively loads the128 packet bytes per
tile; shuffles distribute the packed bytes to low/high nibble consumers. Exact
codebook floats are loaded once per wave then selected with register shuffles.
This reduces redundant instructions, not the unique matrix bytes read from VRAM.
Representative linked ISA:860→778 static instructions, global b32 loads26→13,
global u8 loads24→8, DS permutations10→42. Candidate59 VGPRs, zero LDS/scratch;
runtime occupancy estimates are not measured occupancy. See `codeobjects`.

The bounded packet-only screen improved gate/up aggregate speed2.91%, and full
decode about1.4–1.5%. Adding codebook reuse improved aggregate gate/up speed8.23%
and full-model decode3.27–3.37%; this is the variant accepted after full coverage.
Each screen used64 real layers ×8 FFN runs,2,621,440 outputs, max_abs0 against
CPU RMS/SwiGLU plus independently validated GEMV, with strict control/candidate
bit checks. It did not replace the pinned CPU GGML dequant oracle.

## Current matching Vulkan

Pinned b11284 native public C API, same9070XT and full-file-hashed GGUF, context512,
B1/ubatch1, FP32 KV/state, flash attentionoff,64 warm tokens plus case warm.
Sequential prefill requests only the final prompt LM head. Model load excluded;
reset and full-vocab CPU read included. Greedy8 and sampled32 are fixed-length.
The project C# port of seed42/temp0.7/top40 policy passed576 stored Rust-sampler
steps; native math remains independent. Host selection implementations differ,
so both compute and wall rates are shown.

| Prompt | Frozen phase8 HIP compute / wall sampled tok/s | Current Vulkan compute / wall sampled tok/s |
|---|---:|---:|
| chinese | 16.014 / 14.499 | 25.550 / 23.325 |
| english | 15.977 / 14.463 | 25.601 / 23.400 |
| long | 15.857 / 14.384 | 25.621 / 23.392 |

The sampled HIP rows are a fresh frozen phase8 baseline, not optimized candidate
sampling speed. Current short native token sequences match, but full-vocab native
prompt/last max_abs spans roughly0.14–1.00. Prior long-history disagreements remain;
the0.001 gate was not widened. No claim of independent numerical equivalence.

Vulkan's public free-memory API returns WDDM **budget headroom**. Applying a
physical2GiB reserve to that metric first aborted the test. A64MiB suballocation
trial did not help and was not adopted. The final check uses that runtime's
reported physical heap capacity minus Microsoft's global dedicated VRAM usage
through read-only ADL; it retains physical2GiB+16MiB and separately requires256MiB
WDDM headroom. Minimum physical free2,206,203,904B, WDDM about603–633MB. A first
conservative check wrongly mixed smaller HIP capacity with native occupancy;
all failed attempts remain saved. No other process was stopped or settings changed.

## Absolute reproducibility and quality

Four identical-byte executable/library-origin combinations gave15.84–15.99 tok/s.
Actual HIP/HSA/ROCDXG paths, hashes, compiler versions and starting source are
saved. Later the same resident process showed14.6 on short and15.8 on long requests;
256 warm tokens did not eliminate the shift. Final64-warm frozen arms were faster
again. ADL diagnostics recorded changing clocks and temperatures, but provide no
matched earlier counter record or proven causal explanation. Their overlap is
diagnostic only; the final headline excludes polling. Preserve both regimes and
the prior14.6 evidence; do not promise16.4 or discard slower results.

Full724-step long/reset coverage and480 sampled comparisons passed max_abs0,
unchanged gates, capacity256/512 and highest processed position509. Near512 free
generation compares20 steps until existing independent-history divergence; two
64-step teacher/reset paths cover the rest. Default14 unit tests and format/AST
checks pass. Requested peak remains13,579,250,072B, measured HIP min-free2,343,768,064B.

## Remaining gap and sensitivity

Current faster-regime candidate latency is about61ms; native about39ms, leaving
21–22ms/token. This change removes about2ms/token, not the full gap. Diagnostic
gate/up windows changed23.66→21.41ms/forward. The profile
uses HIP events and reserve queries and is not the headline; nested stage times
overlap and must not be summed as exclusive GPU costs. `summary.json` gives an
Amdahl sensitivity for halving/removing the remaining gate window. These are
idealized arithmetic scenarios, not measured rates or guaranteed upper bounds.
Gate/up remains a large target; broader GEMV/FFN and native arithmetic differences
still need work before claiming SOTA parity. Nothing was published.

For a concrete conditional bound, applying the21.41ms gate window to the
uninstrumented61ms total gives about19.7–19.9tok/s if half the gate cost vanished,
and25.0–25.3 if all vanished. These cross-scope scenarios assume unchanged other
costs; zero gate cost is not a realizable kernel. A separate same-instrumented-run
sensitivity is saved in `summary.json` to avoid mixing measurement scopes.
The diagnostic whole FFN window is38.04ms/forward, with norm3.57 and down13.06;
its containing window and subwindows must not be added together. The measured
uninstrumented improvement remains about2ms/token.
