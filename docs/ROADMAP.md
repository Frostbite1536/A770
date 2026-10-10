# Roadmap

Phases have explicit exclusions. Exit criteria for Phases 0–3 are measured on a real A770.

**Starting premise (2026-10-10):** llama.cpp already runs FA on the A770 through two backends. SYCL uses oneDNN SDPA or MKL GEMMs (XMX) for ≥ 32 tokens, and the non-XMX TILE/VEC kernels below that. Vulkan runs its scalar or coopmat1 shaders. Any change has to beat the faster of the two on specific shapes (INV-PERF-002). volta-fa's lessons that carry over are listed in `docs/research/llama-cpp-arc-fa-dispatch.md` §5. Its Volta tensor-core specifics do not carry over.

## Pre-hardware track (now → A770 access)

**Goal**: Make day one on the hardware about measuring, not setup.
**Includes**:
- P-1 Static toolchain without a GPU: ocloc + IGC + icpx, no sudo (`bin/setup-dev.sh`); per-kernel acm-g10 report with DPAS/FP64 counts (`bin/xe-report.py`); compile probes that fixed the architecture invariants (`docs/research/xe-hpg-static-probes-2026-10-10.md`), kept as ctest fixtures. ✅
- P-2 Reference + tests: CPU attention reference, tiled numerics model, tolerances and error budget carried over from volta-fa (backend-independent), passing under g++ and icpx. ✅
- P-3 Desk research: how llama.cpp picks the FA kernel on the A770 in SYCL and Vulkan. ✅ first pass (`docs/research/llama-cpp-arc-fa-dispatch.md`). Candidates (unmeasured):
  1. SYCL TILE streams K/V G/ncols2 times for G = 3, 5, 6, 7, **including F16 single-token decode**.
  2. SYCL selection and TILE dispatch compute the GQA predicate twice, differently.
  3. SYCL converts quantised K/V to F16 for B = 3..31 with the 256-work-item-per-row geometry volta-fa fixed in CUDA (0005/0006).
  4. Nothing below 32 tokens uses XMX on the A770.
  5. Vulkan's Intel decode fast path is integrated-GPU-only on Xe1.
- P-3b Static compile of llama.cpp's SYCL FA kernels for acm-g10. ✅ `bin/llama-sycl-fa-static.sh`; upstream baseline in `bench/results/llama-sycl-fa-static-20261010/`. Finding: every TILE instance at D 128/256 spills 7–15 KB at SIMD32/128 GRF (research §6).
- P-10 TILE register pressure: does `grf_size_automatic` (256 GRF) remove the spills, at what thread cost? Static experiment (ticket p10), then an A770 A/B.
- P-4 Planning calculator and decode cost model for Xe-HPG. ✅ `bin/plan-tiles.py`, `bin/decode-gqa-model.py` (datasheet constants, labelled; recalibrate from Phase 0). Note from the planner: at 128 GRF a SIMD8 sub-group holds only 4 KB of live data.
- P-5 Phase 0 measurement kit: `bench/phase0.sh` runs FA support/correctness/perf via test-backend-ops on **SYCL0 and Vulkan0**, plus the llama-bench grid (depth, KV quant, small batches, ubatch, MKL/oneDNN on/off). Drafted, **not run**.
- P-6 Padded GQA in the SYCL TILE kernel + dispatch (port of volta-fa P-6). **Designed** (`docs/design/P6-sycl-padded-gqa.md`): unlike Volta's MMA kernel, the SYCL TILE kernel has no phantom-head support, so the kernel's work-group indexing changes too. Correctness harness first: SYCL on the OpenCL CPU device (ticket `p6a-sycl-cpu-harness`, in progress with a Sol worker); implementation ticket `p6b` after it.
- P-8 Conversion geometry for short non-contiguous rows in `ggml-sycl/convert.cpp` (port of volta-fa 0005/0006). Not started.
**Excludes**: any performance claim; tuning by guesswork; Battlemage/Xe2 code paths.

## Phase 0: Baseline and gap analysis (first sessions on hardware)

**Goal**: Find where the existing A770 paths are actually slow, per backend.
**Includes**: run the P-5 kit; `test-backend-ops -o FLASH_ATTN_EXT` perf on SYCL0 and Vulkan0; llama-bench over depth, batch 1–16, KV quant, ubatch, on models with G = 4, 5/7 and D = 128/256 that fit in VRAM; VTune GPU hotspots on the FA kernels; record ReBAR, kernel driver, IGC.
**Excludes**: writing new kernels.
**Exit**: `bench/BASELINE.md` + `profiles/phase0-*.txt`, and a written go/no-go per shape naming the measured bottleneck and the faster backend. Re-check upstream (SYCL FA changes quickly) so we don't rebuild what has landed.

## Phase 1: Dispatch fixes and/or kernel (only for "go" shapes)

**Goal**: A correct change that beats the faster backend on the Phase 0 gap shapes.
**Includes**: P-6/P-8 patches first (cheap, upstreamable); an XMX small-batch kernel only if the gap remains compute-bound; static report → VTune loop until spills are 0 and DPAS > 0.
**Excludes**: multi-GPU.
**Exit**: parity tests pass; `test-backend-ops` correctness on SYCL0; microbench shows a gain (or honestly fails to) with artifacts in `bench/results/`.

## Phase 2: llama.cpp integration

**Goal**: End-to-end prefill/decode gain on real GGUF models.
**Includes**: patch series on the pinned commit; `llama-bench` before/after (3+ interleaved rounds); perplexity check.
**Exit**: end-to-end numbers + perplexity delta published, or an upstream PR.

## Phase 3: Multi-GPU (only with two A770s)

**Goal**: Reduce inter-card stalls.
**Includes**: peer-access capability probe (INV-MGPU-001); bandwidth test (peer vs staged); llama.cpp `GGML_SYCL_DEV2DEV_MEMCPY` modes 0/1/2 measured; overlap.
**Exit**: measured tok/s change on a real dual-card run, or a documented negative result.

## Phase 4: Publish

README results table, reproduction script, write-up using only INV-PERF-001-compliant numbers.
