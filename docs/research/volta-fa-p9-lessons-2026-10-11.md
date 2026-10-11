# What volta-fa's small-batch matmul work (P-9) teaches the A770 (2026-10-11)

volta-fa's P-9 is a Volta tensor-core matmul for quantised weights at 5–16 tokens, llama.cpp patch 0008. It was
measured on a rented V100 on 2026-10-11 (`~/Github/volta-fa/docs/design/P9-small-batch-qmatmul.md`). This repo
already carries volta-fa's attention and conversion lessons (`llama-cpp-arc-fa-dispatch.md` §5). Nothing below has
run on an A770. Items marked **hypothesis** are reasoning from V100 data and llama.cpp code at the shared pin
`79e2e74`, not measurements.

## 1. Measure the matmul before committing to attention work

- **Where the time went.** On the V100, the llama-superfast-v100 fork was 85–89% faster than upstream on
  Qwen3.8-27B at 8–16 tokens. About 95% of that came from its quantised matmul, not attention: 68.5 → 32.8 ms of a
  78.7 ms upstream step. volta-fa had assumed attention
  (`volta-fa/docs/research/qwen-smallbatch-and-vecpack-heads-v100-2026-10-11.md`).
- **What upstream did per batch size.** It streamed weights at 570–775 GB/s at 1 token but 205–365 GB/s at 9–16
  tokens. Patch 0008 alone gave Qwen +34–39% at 8–16 tokens.
- **This repo today.** The roadmap is almost entirely FlashAttention, and neither `bench/phase0.sh` nor
  `bench/patches/run.sh` measures MUL_MAT performance.

**Suggested for Phase 0:**
- `test-backend-ops perf -o MUL_MAT` at n = 1..16 per weight type on SYCL0 and Vulkan0, converted to GB/s against
  the A770's 560 GB/s (`volta-fa/bin/qmatmul-table.py` does this conversion for CUDA output).
- A per-kernel time breakdown (unitrace or VTune) of one 8-token and one 16-token step on a real model.
- Then rank attention against matmul by measured share.

## 2. The SYCL backend already reorders weights (checked at the pin)

`ggml-sycl.cpp` sets `opt_feature.reorder` for every Intel GPU (`ext_oneapi_architecture_is(intel_gpu)`) and
reorders Q4_0, Q8_0 and Q2_K–Q6_K for its MMVQ/DMMV paths (`ggml_sycl_supports_reorder_*`, plus ESIMD variants). So,
unlike on CUDA, the "repacked weights" question is partly settled upstream on SYCL.

Phase 0 should record which path each matmul takes at 1–16 tokens:
- reordered MMVQ/DMMV, MMQ, or oneMKL/oneDNN GEMM;
- with `GGML_SYCL_DISABLE_OPT` on and off;
- and Vulkan's mul_mat_vec / coopmat paths.

## 3. A small-batch XMX matmul maps naturally (hypothesis)

- **Shape.** Volta's `mma.m8n8k4` puts the 8 tokens of a tile on the M side. The A770's `joint_matrix` 8×8×16 at
  sub-group 8 has the same shape class.
- **Layout.** On the V100, staging the weights through shared memory, in their native layout, reached 0.76–0.90×
  of a fully repacked layout with decode and math included (`volta-fa/bench/micro/qload.cu`). On SYCL, the reordered
  layout above may be the natural input instead.
- **Activations.** The A770's hardware also has int8 DPAS, which Volta lacks, but this repo's invariants allow only
  `joint_matrix` with f16/bf16 inputs and f32 accumulation. Feeding q8_1 activations to XMX would need a manager
  decision. Within the invariants, FP16 activations with a power-of-two scale per slice (§4) are the natural input,
  as on Volta.
- **Decode tricks.** volta-fa's exact ones work on any FP16 ALU: OR 4-bit codes into 0x6400 (1024 + q), then one
  half2 FMA; for Q6_K, codes at bits 4–9 and one FMA with sc/16 and −96·sc.

This is worth a design doc only if Phase 0 (§1) shows a gap at 5–16 tokens.

## 4. Numerics: the q8_1 range limit applies to SYCL too

- **Where the limit is.** `block_q8_1` stores d and the block sum of the raw activations in FP16. SYCL's
  `quantize.hpp` writes `ds = half2(half(d), half(sum))`, so sums above 65504 become inf at activations around 1e5,
  and d underflows around 1e-5. Vulkan's `block_q8_1` also has an FP16 `ds` (`types.glsl`); whether its sum
  overflows the same way was not checked.
- **What volta-fa measured.** Its range tests (`volta-fa/third_party/patches-local/0009-*.patch`,
  `test_mul_mat_range`) passed 18/35 on the CUDA current path and 35/35 with patch 0008, which scales each 256-value
  slice by an exact power of two before converting to FP16.
  - **Suggested:** port them as a local-only patch and run them on SYCL0 and Vulkan0. They fail on unmodified
    upstream, so they must not go into an upstream test list.
- **Accuracy method.** volta-fa added a development switch that routes quantised matmuls to dequantise + FP32 GEMM
  (a reference). It then compared llama-perplexity KL divergence of each path against that reference: mean KLD
  0.0017 for q8_1 vs 0.0001 for FP16 activations on Qwen. That is more informative than perplexity against upstream,
  which is itself approximate. The same switch can be built on SYCL with oneMKL SGEMM.

## 5. Measurement lessons from P-9

- **The benchmark keeps small matrices in cache, and the A770's cache is large.** test-backend-ops perf repeats one
  matrix, so it stays in L2. The A770 has 16 MB of L2 (Intel's spec; confirm on the box), so even a 4096×4096 Q4_K
  matrix (≈ 9.4 MB, a whole Llama-3.1-8B attention projection) fits. Kernel benchmarks there will flatter memory
  behaviour much more than on the V100 (6 MB). Confirm every matmul gain inside the model, using per-kernel times by
  shape.
- **Small ops jump between runs** (20–70 µs on the V100, both paths). Use the minimum of 3 or more interleaved rounds.
- **Thresholds differ per type and per N.** On the V100 the crossover against the mat-vec path was 5–9 tokens
  depending on the type, and small matrices lost. Measure each type separately before routing.
- **Find out which side you are waiting on.** volta-fa used development switches: weights from L2 only, and loads
  only with padded SLM so occupancy stays the same. Each took about 80% of the total time, which showed the native
  kernel was near its limit. The same experiment applies to any XMX prototype.
- **Process:**
  - When the built kernel departs from its design doc, write it down at the time. volta-fa's activation scaling
    drifted silently and had to be restored.
  - Before releasing a rented or borrowed machine, diff its results folders against the local copies.
