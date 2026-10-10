# Architecture

## Components

```
        llama.cpp (third_party/, pinned commit)
           │ ggml SYCL backend hook        │ ggml Vulkan backend (baseline only)
           ▼                               ▼
 ┌─────────────────────┐        ┌──────────────────────┐
 │ include/a770/       │        │ src/mgpu/ (later)    │
 │ shared contract     │        │ peer access, staged  │
 └─────────┬───────────┘        │ copies, overlap      │
           ▼                    └──────────┬───────────┘
 ┌─────────────────────┐                   │
 │ src/attention/      │◄── split-KV / TP partials
 │ SYCL / OpenCL C     │
 │ XMX (DPAS SIMD8)    │
 └─────────────────────┘
```

| Component | Responsibility | Depends on | Interface |
| --- | --- | --- | --- |
| `src/attention/` | Attention kernels for the gap shapes Phase 0 finds; XMX via SIMD8 DPAS, online softmax in FP32 | SYCL (icpx 2026.1), `include/` | launch wrappers |
| `src/mgpu/` | Peer access check, staged fallback, event-based overlap | Level Zero / SYCL | later (Phase 3) |
| `include/a770/attention.h` | Tensor layout, mask/causal semantics, reference API: the contract every kernel and test shares (identical to volta-fa's) | none | header |
| `tests/` | CPU reference, tiled numerics model, parity tests, xe-report fixtures | `include/` | ctest |
| `bench/` | Phase 0 kit, result files, analysis | llama.cpp | `bench/phase0.sh`, `bench/analyze.py` |
| `bin/xe-report.py` | Static acm-g10 resources and DPAS/FP64 counts per kernel | ocloc + IGC, icpx | CLI |
| llama.cpp integration | Patches against the pinned commit | all | `third_party/patches/` |

## Data flow (attention, when a kernel is written)

1. Q tile loaded to GRF/SLM; loop over K/V tiles.
2. `S = Q·Kᵀ` with DPAS 8×8×16, FP32 accumulate (INV-NUM-005).
3. Online softmax: update running max `m`, sum `l` (FP32), rescale accumulator.
4. `P` cast to FP16 (or BF16) → `O += P·V` with DPAS.
5. Write O (and LSE if needed). No N×N tensor reaches global memory (INV-NUM-001).

## Key design decisions (to be filled in with evidence)

| Decision | Status | Evidence |
| --- | --- | --- |
| SYCL `joint_matrix` vs OpenCL C builtins vs ESIMD | open, leaning SYCL `joint_matrix` (llama.cpp's backend is SYCL; wrong shapes fail loudly) | `docs/research/xe-hpg-static-probes-2026-10-10.md` |
| Fix llama.cpp's dispatch first vs write a kernel | **first: dispatch** (padded GQA, one predicate, conversion geometry: the volta-fa P-6/P-8 lessons) | `docs/research/llama-cpp-arc-fa-dispatch.md` §5 |
| FP16 vs FP32 accumulate for S | **decided: FP32** (INV-NUM-005) | `docs/research/error-budget-2026-10-09.txt` |
| 128 vs 256 GRF per kernel | open, per kernel | `bin/xe-report.py` `thr` column, then VTune occupancy |
| SYCL vs Vulkan as the integration target | open: SYCL is where the code is ours to change; Vulkan is part of the baseline | Phase 0 |

## Failure modes and limits

- No A770 → CPU tests and static acm-g10 reports only; hardware results are "not run" (INV-PROC-002).
- The compiler can drop XMX silently (SIMD16 DPAS). The DPAS count in `bin/xe-report.py` is the guard.
- 16 GB (or 8 GB) VRAM limits model size and context more than on a 32 GB V100; pick Phase 0 models to fit.
- Resizable BAR off, or the wrong kernel driver, can dominate any result. Both are recorded with every run.
