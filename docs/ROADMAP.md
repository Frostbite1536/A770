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
- P-10 TILE register pressure. Static result: 256 GRF reduces but never removes the spills and halves threads/XVE; some D=256 configs get worse. Kept as an optional A770 A/B variant (`third_party/patches/p10-draft-*`). Structural option: a SIMD16 TILE variant (Phase 1, only if Phase 0 shows TILE time is spill-bound).
- P-11 FP16 output accumulation in SYCL FA TILE (research §7, from the P100 project; design `docs/design/P11-sycl-fa-fp16-accumulation.md`). Patch 0005 adds biased-V cases with an FP32 CPU reference. On the CPU SYCL device at A770 launch geometry, TILE fails them at ≥ 4096 keys per half2 accumulator (NMSE up to 2.3); VEC passes (p11/p11b). Two fix candidates in `third_party/patches/p11c/`: F (FP32 flush per KV tile, spill −576…+2592 B) and S (≤ 1024 keys per block, host-side only); both pass 16/16 with existing coverage unchanged (p11c). Accept-if fixed 2026-10-11; A770 A/B (variants `fp32f`/`cap`) and the A770's real `max_wg_per_cu`: not run.
- P-4 Planning calculator and decode cost model for Xe-HPG. ✅ `bin/plan-tiles.py`, `bin/decode-gqa-model.py` (datasheet constants, labelled; recalibrate from Phase 0). Note from the planner: at 128 GRF a SIMD8 sub-group holds only 4 KB of live data.
- P-5 Phase 0 measurement kit: `bench/phase0.sh` runs FA support/correctness/perf via test-backend-ops on **SYCL0 and Vulkan0**, plus the llama-bench grid (depth, KV quant, small batches, ubatch, MKL/oneDNN on/off). Drafted, **not run**.
- P-6 Padded GQA in the SYCL TILE kernel + dispatch (port of volta-fa P-6). Design `docs/design/P6-sycl-padded-gqa.md`: unlike Volta's MMA kernel, the SYCL TILE kernel has no phantom-head support, so the kernel's work-group indexing changes too. Correctness harness **done** (p6a: `bin/sycl-cpu-tbo.sh`, SYCL on the OpenCL CPU device; 85/85 subset pass, TILE negative control fails then recovers). **Patches 0001/0002 done** (p6b): CPU-device correctness 197/197, failing controls recover; acm-g10 static unchanged SIMD/GRF, spill bytes up to +8.4% on already-spilling instances (owner judges after the A770 A/B). A770 timing not run (day-one A/B, `bench/patches/run.sh`).
- P-8 Conversion geometry for short non-contiguous rows in `ggml-sycl/convert.cpp` (port of volta-fa 0005/0006). **Patches 0003/0004 done** (`docs/design/P8-sycl-kv-conversion.md`): bit-exact microbenchmark `bench/micro/dequant_nc_sycl.cpp`, CPU-device correctness, acm-g10 static SIMD16/no spill. A770 timing not run (day-one A/B, `bench/patches/run.sh`).
- P-9 Day-one kit. ✅ `bench/patches/run.sh` (upstream/p6/conv/conv8 + opt-in `grf`, Vulkan reference; correctness, perf, llama-bench; CPU-smoke-tested plumbing only), `docs/A770_BOX.md`, `prompts/a770-session-1.md`.
- Lessons from volta-fa P-9 (V100 small-batch matmul, 2026-10-11): `docs/research/volta-fa-p9-lessons-2026-10-11.md`. On the V100 the small-batch gap was ~95% matmul, not attention. Tickets p12–p15 ✅ merged 2026-10-16 (CPU-device verified, **not run** on an A770): Phase 0 `mmperf` and MUL_MAT/range tests, opt-in `prof` and `kld`; see `docs/HANDOFF.md`. Suggested: MUL_MAT perf at n = 1..16 on SYCL0/Vulkan0 and a per-kernel step breakdown in Phase 0 (the kit measures no MUL_MAT perf today); volta-fa's activation-range tests as a local-only patch (SYCL's q8_1 stores its sum in FP16). Upstream SYCL already has a small-batch XMX dequant-GEMM (`fused-gemm.cpp`, ≤ 64 tokens, Arc-A SIMD8 combinations), so the question is measuring it, not building one.
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
