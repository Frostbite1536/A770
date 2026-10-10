# Xe-HPG (acm-g10) static compile probes, 2026-10-10

Compiled on the dev laptop (no Intel GPU) with the pinned static toolchain: ocloc 26.35.39758.10 + IGC 2.41.5, and icpx 2026.1.1 (`bin/setup-dev.sh`). Each probe compiled a small kernel for `-device acm-g10` with IGC shader dumps on and read the `.zeinfo` metadata and the final `.asm`. These probes are the evidence behind INV-ARCH-001/002/004. The fixtures that still guard them are in `tests/xe/`, run by `tests/test_xe_report.py`. Nothing here ran on a GPU: these are **compiler facts, not hardware measurements**.

| # | Probe | acm-g10 result | Fixture / test |
| --- | --- | --- | --- |
| 1 | OpenCL `intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc)` at `intel_reqd_sub_group_size(8)` | SIMD8, 1 `dpas` instruction | `dpas_simd8.cl` / `xe_report_dpas_simd8` |
| 2 | Same builtin with `short8 a` at sub-group 16 (the Xe2/Xe-HPC form) | **Build succeeded, 0 `dpas`, no warning**. IR shows `__builtin_IB_sub_group16_fdpas_*`, `hasDPAS=false`, and the store writes registers that were never written. The same source with `short8 a` at sub-group 8 also gives 0 DPAS. For contrast, `-device bmg`: the SIMD16 form emits DPAS and the SIMD8 form fails to build | `dpas_simd16.cl` / `xe_report_dpas_simd16_silent` |
| 3 | 48 live `float16` per work-item at sub-group 16 | `[RetryManager] Start recompilation`, "compiled SIMD16 allocated 128 regs and spilled around 1474", zeinfo `spill_size: 47168` | `spill.cl` / `xe_report_spill` |
| 3b | Same kernel with no declared sub-group size | IGC retries and lands on **SIMD8** (still spilling 22016 B); a light kernel with no declaration got SIMD32. SIMD width follows register pressure unless pinned (INV-ARCH-004) | (manual) |
| 4 | `cl_khr_fp64` + `double` | "unsupported OpenCL extension 'cl_khr_fp64'", build fails (bmg: builds, 3 `:df` ops) | `fp64.cl` / `xe_report_fp64` |
| 4b | `intel_sub_group_2d_block_read_16b_8r16x1c` | "use of undeclared identifier", build fails (bmg: builds) | `block2d.cl` / `xe_report_block2d` |
| 5 | `-cl-intel-256-GRF-per-thread` (OpenCL) / `-options -ze-opt-large-register-file` (SYCL AOT) | Accepted for acm-g10: `grf_count: 256`, `eu_thread_count` 8 → 4. Probe 3 still spills 43072 B | `xe_report_large_grf` |
| 6 | `__local float t[N]` | 64 KB: `slm_size: 65536`. **96 KB also compiles** (`slm_size: 98304`): IGC does not enforce a per-work-group SLM limit at compile time | `slm.cl` / `xe_report_slm` |
| 7 | SYCL `joint_matrix` M=8,N=8,K=16, f16 → f32, `reqd_sub_group_size(8)` | SIMD8, 2 `dpas`; bf16 inputs: same | `joint_matrix.cpp` / `xe_report_sycl_joint_matrix` |
| 7b | SYCL `joint_matrix` M=8,N=16,K=16 at sub-group 16 (Xe2 shape) | `IGC: Internal Compiler Error: Floating point exception`, build fails | (not kept: an ICE is not a stable contract) |
| 8 | `icpx -fsycl -fsycl-targets=spir64_gen -c` | Warns the `-Xsycl-target-backend` argument is unused: AOT happens at device link. `bin/xe-report.py` builds a `-shared` library so no `main()` is needed | — |

## What this means for kernel work

- **DPAS = 0 is the Arc version of volta-fa's "Ampere bias".** Models (and people) reach for the SIMD16 forms that Battlemage and PVC examples use. On acm-g10 the OpenCL builtin silently produces garbage, so every XMX kernel is checked for DPAS count > 0 at SIMD 8 (`--expect-dpas`).
- The SYCL path fails loudly (ICE) for the wrong shape, which is safer. It still needs the report: a `joint_matrix` that IGC lowers without DPAS would not error either.
- IGC's spill retry changes the SIMD width unless the kernel pins it, and that changes the per-work-item fragment layout. Pin it.
- SLM and occupancy budgets are ours to enforce. 256-GRF mode is a real trade (half the threads).

## Not checked yet

- ESIMD (`sycl::ext::intel::esimd::xmx::dpas`), and whether ESIMD's LSC messages without 2D are all legal on acm-g10.
- The XMX shapes and types that the oneDNN/MKL paths in llama.cpp use (library code, not compiled here).
- Anything about timing, bandwidth or occupancy at runtime: not run.
