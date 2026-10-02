# Run the existing HIP engine and reproduce the comparison

The source build and the validated phase13 configuration execute the existing
Rust/HIP engine. Phase14 does not introduce a new engine or a promoted kernel.
Use explicit input files: `generate` handles greedy text generation, while
`evaluate` supplies the fixed-length benchmark. The documented local target is
RX 9070 XT / gfx1201, Qwen3.8-27B mixed IQ4_XS GGUF, batch one and FP32
activations, KV and recurrent state. Supported context capacities are 128, 256
and 512; prompt tokens plus the requested output limit must fit the capacity.

## Build and local files

Run from the repository in WSL AMD-Infer with the established Rust/ROCm
environment. The recorded toolchain is Rust 1.98.1 and ROCm 7.2.3. Python 3 is
required for the launchers; Jinja2 is additionally needed only for chat-template
rendering. No launcher downloads model files or installs packages.

```sh
export ROCM_PATH=/opt/rocm-7.2.3
cargo build --release --features hip,tokenizer --bins
cargo test --lib --features hip,tokenizer

MODEL=/path/to/Qwen3.8-27B-UD-IQ4_XS.gguf
TOKENIZER=/path/to/qwen-config/tokenizer.json
```

The benchmark requires this model SHA256:
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
The local file is 14,252,845,984 bytes. Model weights, the official tokenizer and
native reference DLLs are separate user-supplied files. The project preserves
its existing third-party licenses; the launchers do not redistribute those files.

`configs/rx9070xt-iq4-xs-fp32.json` is a portable opt-in source-build preset based
on the validated phase13 policy. It is distinct from source defaults, which stay
off for experimental flags. Each launch records the actual executable and linked
HIP library hashes. Local frozen replay additionally checks every hash in the
saved validated config. Frozen runtimes under `results/` are local artifacts,
not files provided by a fresh source checkout.

## Real generation

Create a UTF-8 prompt file. Raw mode forwards its contents unchanged, including
any role delimiters already present. No BOS or chat template is added implicitly.
Use `--dry-run` to tokenize and check capacity without allocating GPU model
buffers or creating the output directory.

```sh
python3 tools/run_engine.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --prompt-file fixtures/benchmark/a_short.txt \
  --output results/my-raw-run --max-new-tokens 64 --context 512 --dry-run

python3 tools/run_engine.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --prompt-file fixtures/benchmark/a_short.txt \
  --output results/my-raw-run --max-new-tokens 64 --context 512
```

For one user message, explicitly select chat mode and supply the local official
`tokenizer_config.json`. The launcher executes that file's actual chat template.
`--thinking off` produces its closed-thinking generation prefix; `low`, `medium`
and `xhigh` pass the corresponding reasoning setting to the official template.
An optional `--system-file` supplies a system message. These settings affect the
prompt and available output budget; they do not change arithmetic precision.

```sh
python3 tools/run_engine.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --template chat --tokenizer-config /path/to/qwen-config/tokenizer_config.json \
  --thinking off --prompt-file /path/to/question.txt \
  --output results/my-chat-run --max-new-tokens 64 --context 512
```

For exact local phase13 frozen replay, add:

```sh
  --runtime-config results/phase13/experimental-config.json
```

The new directory contains `prompt.txt`, prompt IDs, `launch.json`, `run.log`,
`exit.json`, `output.txt`, generated IDs, the final vocabulary and scoped timing
metrics. Generation stops at model EOS or the requested limit. Its decode wall
includes the initial LM head, sampling and streaming output writes, has no warmup
and is **not** the steady-state benchmark rate. `--verify-model` also records a
full model hash; without it the CLI records size but leaves model SHA unverified.
Preflight and run failures preserve clear errors; existing output directories
are refused.

## Final matched benchmark entry

`tools/benchmark.py` fixes batch one, FP32 KV/state, context 512, 256 sustained
warmup tokens, one complete warm request per case and three trials per arm.
Each trial emits 64 fixed outputs and measures 63 forward/LM-head intervals.
The default fixtures are the same five-token short request, a 447-token long
request, then the identical short request. Each request resets model state.
Compute throughput includes full-vocabulary retrieval; selection wall throughput
is saved separately. Load, warmup and artifact export are outside headline decode.
Fixed benchmark output lengths deliberately ignore EOS.
The benchmark fixes the full 13,333,954,560-byte matrix residency set: it disables
automatic FFN omission and validates the admission record, so desktop VRAM
changes cannot silently turn a trial into a streamed-weight configuration.

The default two-cycle sequence is strictly serial:

1. HIP before, native FP32, native default MMVQ, HIP after.
2. HIP before, native default MMVQ, native FP32, HIP after.

The native path uses the pinned b11284 C ABI and Windows Vulkan0. Supply a WSL
path to its Windows DLL directory and a known `launch.json` hash manifest. The
wrapper creates a separate worker process: MMVQ disable is inherited at startup
for the FP32 arm; it is unset at startup for the default arm. Both use FP32 KV,
batch/ubatch one, context 512, no flash attention and no observation callback.
The default MMVQ arm quantizes eligible matmul activations and is a practical
performance reference with different activation arithmetic. It is not the
independent FP32 numerical oracle.

```sh
NATIVE=/mnt/c/path/to/runtime/b11284
python3 tools/benchmark.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --runtime-config results/phase13/experimental-config.json \
  --native-runtime "$NATIVE" \
  --native-manifest results/phase13/native-bench-fp32/launch.json \
  --output results/my-matched-benchmark --dry-run

python3 tools/benchmark.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --runtime-config results/phase13/experimental-config.json \
  --native-runtime "$NATIVE" \
  --native-manifest results/phase13/native-bench-fp32/launch.json \
  --output results/my-matched-benchmark
```

Windows PowerShell 7 must be accessible from WSL (`pwsh.exe` by default, or
`--powershell /mnt/c/path/to/pwsh.exe`). For a fresh source build, omit
`--runtime-config`. A newly built b11284 reference can be recorded without a
prior manifest; that records DLL hashes but does not certify their identity
against a historical frozen reference. The ABI guard still checks structure
sizes and defaults. The inspected reference source is commit
`25747b08e7a0f9a59a2089ce6b98d2229b76042a`.

Without `--native-runtime`, the entry runs only the two HIP brackets and reports
drift; it cannot declare a win over references. `--cycles 1` is a diagnostic
shorter run and cannot satisfy the two-cycle stable-win classification.

The report retains every trial and median, inter-bracket drift and short-request
drift after the long request. Its conservative performance classification
requires less than 1% within-arm spread and bracket drift, exact HIP boundary
logits/IDs, and the worst HIP trial exceeding the best trial of **both** references
for every case in both cycles. One fast trial cannot produce a positive stable
classification. The original 0.001 prompt-boundary FP32 numerical gate is also
checked. This is a performance classification, not a statistical confidence
interval, SOTA claim or automatic config promotion. It does not replace the
724-step/reset/480-sampling/long-context quality suites. A failed arm halts the
sequence and writes `status.json` with `complete: false`; partial metrics are
never ranked as a complete benchmark.

## GPU coordination, memory and rollback

The two launchers share a nonblocking project file lock. Older scripts and other
applications do not participate. `AMD_INFER_GPU_EXCLUSIVE=1` is an assertion of
coordinated use, not a driver reservation. Run all GPU experiments serially and
coordinate any existing model service. Neither launcher stops a process.

The HIP allocation gate stays at 2 GiB + 16 MiB. The native comparator retains
its physical-memory and WDDM headroom guards. On the current shared desktop,
phase14 failed twice during HIP initialization, before warmup; therefore no new
full-model ranking or steady-state profile was obtained. The first failed small
allocation was only 18–21 MB below the gate, but later lazy state/cache buffers
were not yet all allocated. Freeing just that amount is not evidence that the
complete run will fit. See [phase14 resource assessment](PERSISTENT_SCOPE.md).

Existing frozen rollback commands are unchanged:

```sh
# Phase11 before column-state fusion:
python3 tools/run-best-column.py CASE_DIR results/rollback-phase11 --rollback
# Earlier phase10 matrix policy:
python3 tools/run-best-regroup.py CASE_DIR results/rollback-phase10 --rollback
# The original phase6 ~8.7 configuration/runtime remain archived locally.
```

The new real-generation launcher also accepts any validated frozen config via
`--runtime-config`; phase11/10 generate binaries can be selected that way.
No public repository was published by this work.
