#!/bin/sh
set -eu
: "${LLAMA_REFERENCE_DIR:?point to llama.cpp b11284 checkout}"
mkdir -p results
cc -O2 -mf16c -ffunction-sections -fdata-sections \
  -I"$LLAMA_REFERENCE_DIR/ggml/include" -I"$LLAMA_REFERENCE_DIR/ggml/src" \
  tools/reference_quant.c "$LLAMA_REFERENCE_DIR/ggml/src/ggml-quants.c" \
  -Wl,--gc-sections -lm -o results/reference-quant
