# Toolchain

## Version pins (INV-ARCH-003)

| Piece | Pin | Why |
| --- | --- | --- |
| oneAPI DPC++ (icpx) | 2026.1.x | llama.cpp's SYCL CI moved to the oneAPI 2026.1 toolkit (its SYCL.md reports 3.1× prompt processing on Arc B570 vs the 2025.3 build). CMake refuses other versions. |
| compute-runtime (ocloc) | 26.35.39758.10 | Static reports. acm-g10 is a regular (not legacy) platform in this release. |
| IGC | 2.41.5 (build 22716) | The compiler inside ocloc and the driver. It decides SIMD width, spills and DPAS emission, so reports and timings carry its version. |
| oneDNN | ships with the oneAPI 2026.1 toolkit | llama.cpp's SYCL FA uses oneDNN SDPA for ≥ 32 tokens. Deep Learning Essentials no longer includes oneDNN from 2026.0. |
| llama.cpp | `bench/LLAMA_COMMIT` (same pin as volta-fa) | INV-PERF-002 |

Pins were current on 2026-10-10. Re-check the release notes before bumping: a pin change re-runs the `tests/xe` probes and invalidates earlier static reports.

## Dev machine (no Intel GPU)

No sudo, no system oneAPI. Everything lives in the gitignored `.venv/` and `tools/`, installed by one script (pinned, checksum-verified downloads):

```bash
bin/setup-dev.sh                 # .venv (cmake, ninja, shellcheck, numpy) + tools/neo: ocloc + IGC (~360 MB extracted)
WITH_DPCPP=1 bin/setup-dev.sh    # + tools/oneapi-2026.1: icpx/DPC++, oneMKL SYCL, OpenCL CPU runtime from conda-forge (~4.4 GB)
```

What works here:

- First, once per shell: `export PATH=$PWD/.venv/bin:$PATH`.
- **CPU build + tests** (reference attention, tiled numerics model, analysis scripts, xe-report fixtures): `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build`. With the default host compiler only CPU targets are built. With `-DCMAKE_CXX_COMPILER=$PWD/tools/oneapi-2026.1/bin/icpx` the same tests pass under icpx (`-fp-model=precise`).
- **Static acm-g10 kernel reports**: `python bin/xe-report.py <file.cl|file.cpp>` gives exact SIMD width, GRF mode, threads per XVE, spill, private, SLM and DPAS/FP64 instruction counts per kernel, from the same IGC the driver uses. `.cl` needs only `tools/neo`; `.cpp` (SYCL) needs `WITH_DPCPP=1`.
- **llama.cpp's SYCL kernels** can be AOT-compiled for acm-g10 with the same tools (configure llama.cpp with `-DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=$PWD/tools/oneapi-2026.1/bin/icpx -DGGML_SYCL_DEVICE_ARCH=acm-g10`, `tools/neo/bin` on `PATH`, and `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` for per-kernel dumps). *Not done yet* (roadmap P-3b).
- Lint: `shellcheck -x bin/*.sh`, `bin/check-forbidden.sh`.

What does not work here: running anything on an Arc GPU, timing, VTune/unitrace metrics, multi-GPU.

Possible, not set up: this laptop has an Intel HD P630 (Gen9, `i915`). Gen9 is a *legacy* compute-runtime platform (needs the 24.35 `legacy1` packages and a system install) and has no XMX. It could run **non-XMX** SYCL code (e.g. a patched llama.cpp TILE dispatch through `test-backend-ops`) for correctness, the way volta-fa used its Maxwell GPU. It cannot stand in for A770 timing or for anything XMX.

`tools/neo/bin/ocloc` is a wrapper that sets `LD_LIBRARY_PATH` for the extracted IGC libraries. Put `tools/neo/bin` on `PATH` when icpx needs to call `ocloc` for AOT.

## A770 box (when available)

1. **Resizable BAR on** (BIOS: Above 4G decoding + Re-Size BAR). Arc A-series performance depends on it. Record the state.
2. Linux kernel driver: record which one is bound (`lspci -k`: `i915` or `xe`), and the kernel version.
3. Intel client GPU user-space stack (compute-runtime + IGC + Level Zero) per Intel's client GPU install guide. Record versions (`clinfo | grep 'Driver Version'`, `sycl-ls --verbose`).
4. oneAPI toolkit 2026.1 (`source /opt/intel/oneapi/setvars.sh`), then `cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx`.
5. Vulkan: Mesa ANV version and whether `VK_KHR_cooperative_matrix` is exposed (`vulkaninfo | grep -i cooperative`).
6. Record `sycl-ls --verbose` (local memory size: confirms INV-ARCH-002's 64 KB), board vendor, 8 GB or 16 GB, power limit, in `bench/BASELINE.md`.
7. Profilers: VTune (GPU Compute/Media Hotspots), `unitrace` (intel/pti-gpu), `intel_gpu_top`. See docs/PROFILING.md.
8. Multi-GPU: record the PCIe topology (`lspci -tv`), IOMMU mode, and whether Level Zero reports peer access between the cards.
