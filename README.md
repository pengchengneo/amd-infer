# AMDInfer

Experimental Rust/HIP inference for AMD Radeon RX 9070 XT (`gfx1201`). This
development snapshot supports the confirmed Qwen3.8-27B text GGUF layout,
mixed packed quantization, batch one, greedy generation and diagnostic sampling.
Activations, KV and DeltaNet state are FP32. Context capacity is 128, 256 or 512;
vision, MTP, serving and general-model support are not implemented here.

## Current evidence and limitations

The opt-in phase13 device path measured **23.08–23.39 tok/s** in two local
interleaved A/B cycles with 256 warmup tokens and 64 fixed outputs / 63 decode
intervals. It remains behind the pinned Vulkan FP32 activation comparator
(roughly 24.1 tok/s medians) and default MMVQ comparator (roughly 25.2–25.5 tok/s).
Default MMVQ uses eligible Q8_1 activation quantization; identical GGUF weights
do not make its arithmetic identical to this engine's FP32 path.

The grouped FP32 dot association is opt-in, as are column-major recurrent state
and head-local fusion. Original source defaults are preserved. There is no
whole-model megakernel, across-token persistent engine or SOTA claim. Phase14
only measured a standalone boundary fusion; its 0.15–0.17 ms/token saving is a
microbenchmark extrapolation, not a full-model improvement. Shared-desktop VRAM
prevented its fresh full-model steady-state measurements at the original reserve.

The original 0.001 numerical gates, 724-step/reset coverage, 480 sampled-step
comparisons and long512context diagnostics are documented in the
[phase13 report](docs/GDN_COLUMN_STATE.md). The independent FP32 story oracle
is a transparently reused prior469-step run;36 diagnostic tasks additionally had
a fresh FP32 reference. All three arms scored24/36 on that adapted set, including
12/24 public counting cases. These tests do not establish general/SOTA quality.
Source provenance and notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

The [sanitized measured snapshot](docs/benchmarks/phase13.json) preserves raw
trial rates, comparison scopes and quality limitations. Model weights,
tokenizer files, native DLLs, original experiment logs and frozen executables
are local artifacts and are not shipped in this source checkout. Historical
documents may reference those local artifacts or machine-specific experiment
scripts; the maintained public launchers below use user-supplied paths.

## Build

The recorded toolchain is Rust1.98.1 and ROCm7.2.3. CPU tests and GGUF inspection
do not require ROCm:

```sh
cargo test --locked --lib --features tokenizer
cargo run --release -- inspect /path/to/model.gguf --tensors
cargo run --release -- budget /path/to/model.gguf
```

For GPU execution in a configured Linux/WSL ROCm environment:

```sh
export ROCM_PATH=/opt/rocm-7.2.3
cargo build --release --locked --features hip,tokenizer --bins
MODEL=/path/to/Qwen3.8-27B-UD-IQ4_XS.gguf
TOKENIZER=/path/to/qwen-config/tokenizer.json
python3 tools/run_engine.py --model "$MODEL" --tokenizer "$TOKENIZER" \
  --prompt-file fixtures/benchmark/a_short.txt --output results/my-run \
  --max-new-tokens 64 --context 512 --dry-run
```

Remove `--dry-run` to execute. Raw mode forwards the exact prompt; explicit chat
mode uses the user's local official tokenizer chat template. The source-build
preset in `configs/rx9070xt-iq4-xs-fp32.json` opts into the measured phase13
policy and records actual executable/library hashes. It does not provide a
frozen-binary or cross-machine performance guarantee. Python3.9+ is required;
chat rendering additionally uses Jinja2. No launcher downloads weights.

See [RUNNING.md](docs/RUNNING.md) for real CLI output/timing, chat settings,
rollback and the final matched benchmark. The benchmark fixes full matrix
residency,512context,256warmup,64outputs, short–long–same-short requests and
serial HIP/native brackets. It records every trial and drift rather than peak
rates. Windows Vulkan comparison additionally requires user-supplied pinned
b11284 DLLs and Windows PowerShell7 accessible from WSL. GPU jobs must be
coordinated with other workloads; the project lock is not a driver reservation.
The 2GiB+16MiB allocation reserve remains enabled. The launchers never stop
another process.

## Design and validation

- [Device execution](docs/DEVICE_EXECUTION.md)
- [FP32 grouped quantized GEMV](docs/FP32_REGROUP.md)
- [GDN column state and head fusion](docs/GDN_COLUMN_STATE.md)
- [Persistent scope and resource/liveness limits](docs/PERSISTENT_SCOPE.md)
- [Independent upstream FP32 quality](docs/UPSTREAM_FP32_QUALITY.md)

The portable benchmark is a performance comparison, not a replacement for full
quality qualification or an automatic release/promotion mechanism. New kernel
changes need operator oracles, full-model comparison, reset/long-context checks
and a recoverable baseline. Pinned GGML reference sources/tables retain their
MIT license and source commit. The project is MIT licensed.
