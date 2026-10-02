#!/bin/sh
set -eu
: "${LLAMA_REFERENCE_DIR:?point to pinned llama.cpp b11284 checkout}"
c++ -O2 -std=c++17 -I"$LLAMA_REFERENCE_DIR/include" -I"$LLAMA_REFERENCE_DIR/ggml/include" \
  tools/reference-model.cpp -Lresults/llama-cpu/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,"$(pwd)/results/llama-cpu/bin" -o results/reference-model
