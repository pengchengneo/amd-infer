GGUF/GGML formats and the IQ4 non-linear codebook follow ggml-org/llama.cpp,
licensed under MIT. Numerical oracles use the unmodified b11284 implementation
in a separate checkout. This repository does not claim ownership of GGML formats.

Source: https://github.com/ggml-org/llama.cpp/tree/b11284/ggml
Copyright (c) 2023-2026 The ggml authors. Upstream license:
https://github.com/ggml-org/llama.cpp/blob/b11284/LICENSE

HIP kernel and Rust runtime here implement that wire-format specification.

The phase5 hidden/residual and attention-preparation kernels are project-authored.
The upstream MIT IQ4_XS Vulkan shader was inspected as a design reference; the
exact inspected source and hash are retained in results/phase4. No grouped or
block-scaled GEMV port was added in phase5. Existing GGML table and format
attribution above remains applicable.

The phase6 wave-uniform row specialization and public-HIP launch/graph diagnostic
tools are project-authored. No upstream block-scale regrouping or packed-load
shader code was ported in this phase. The existing format/codebook attribution
continues to apply.

Phase7's stream ownership, graph cache/invalidation and public-HIP diagnostic
observers are project-authored against the installed official HIP headers.
No upstream graph or persistent-kernel implementation was copied.

Phase8 head-local DeltaNet fusion, four-row warp GEMV, dispatch controls and
benchmark drivers are project-authored. GGML format/codebook attribution remains
unchanged. Design comparison inspected the pinned MIT Vulkan IQ4_XS shader and
its reduction base, and the CUDA/HIP MMVQ dispatch source; no kernel code was
copied from these files. The project kernel preserves its existing FP32
activation/dequant/FMA policy, without adopting the upstream Q8 activation path.

Inspected references:
- https://github.com/ggml-org/llama.cpp/blob/b11284/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq4_xs.comp
- https://github.com/ggml-org/llama.cpp/blob/b11284/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl
- https://github.com/ggml-org/llama.cpp/blob/b11284/ggml/src/ggml-cuda/mmvq.cu

Phase9 cooperative IQ4_XS packet/register-codebook gate/up, public-ABI native
benchmark, sampler port and read-only ADL observers are project-authored.
GGML format/codebook attribution and licenses remain unchanged. No upstream
shader/kernel implementation or AMD SDK sample code was copied. The installed
Windows/AMD runtime DLLs are used in place and are not redistributed. Public ABI
references for the read-only observer and native benchmark:
- https://gpuopen-librariesandsdks.github.io/adl/overdrive8_8h.html
- https://gpuopen-librariesandsdks.github.io/adl/group__ADAPTERMAINAPI.html
- https://github.com/ggml-org/llama.cpp/blob/b11284/include/llama.h

Phase10 cooperative IQ3_S word ownership, generic IQ4_XS/IQ3_S GEMV extensions,
attention observer and bounded generation are project-authored. No upstream
kernel/shader implementation was copied. The existing MIT GGML table/format
attribution remains. Design comparison additionally inspected the pinned source:
- https://github.com/ggml-org/llama.cpp/blob/b11284/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iq3_s.comp
Reference hashes and commit are in local results/phase10/payload-and-references.json.

Phase11 row/codebook/vector-read experiments, FP32 grouped IQ4_XS/IQ3_S dots, fused gate/up and long-generation public-ABI wrappers are project-authored. Pinned MIT Vulkan IQ4_XS/IQ3_S and reduction sources were inspected as design references; no shader implementation was copied. Existing GGML tables/formats and MIT attribution remain applicable. Exact source/license hashes and commit are in local results/phase11/upstream-reference-comparison.json. Installed AMD/Windows/llama runtime DLLs remain in place and are not redistributed.

## BIG-bench diagnostic data (phase13)

The local phase13 quality fixtures sample and adapt BIG-bench object_counting data from https://github.com/google/BIG-bench at commit cbbbbaa0a279c7e066ec60b443bc33f6812fbacb. The source task, canary, repository Apache-2.0 LICENSE and README are retained under `results/phase13/external-quality/source`, with SHA256 hashes and sampled indices in `provenance.json`. Role tokens, closed thinking segment and answer-only instructions were added; the result is not a canonical BIG-bench benchmark score. No BIG-bench implementation code was copied into the engine. Preserve the source notices and license when redistributing these data artifacts.
