# Ticket p8-sycl-kv-conversion: faster non-contiguous K/V → F16 conversion in llama.cpp's SYCL backend

Your current directory is a git worktree of A770 (branch `p8-sycl-kv-conversion`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. These are non-XMX SYCL kernels. Do NOT use SIMD16 DPAS, 2D block loads, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC. Every kernel you write declares its sub-group size (`[[sycl::reqd_sub_group_size(N)]]`, INV-ARCH-004). If you think you need something forbidden, STOP and explain.
- The toolchain lives in the MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/neo`, `tools/oneapi-2026.1`), read-only, by absolute path; `A770_TOOLS=/home/p51/Github/A770/tools` for `bin/xe-report.py` if p6c's option exists in your base, else run the main checkout's `bin/xe-report.py` on your file by absolute path. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write. Read-only git is fine; `git diff`/`git apply` inside the scratch clone are fine.
- Do not loosen tolerances; do not edit or delete existing test cases. Results must be bit-identical to upstream conversion; any difference is a failure to report, not to explain away.
- Never invent numbers. CPU-device timings are NOT A770 timings: record them only as "CPU device, mechanism check", or not at all.
- Never kill a process you did not start. Long builds: rerun after a timeout.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/llama-cpp-arc-fa-dispatch.md` §3: before the SYCL TILE FA kernel runs with a quantised K/V cache (3–31 query tokens), `launch_fattn` (`ggml/src/ggml-sycl/fattn-common.hpp:954-1012`) converts the K/V views to F16 with `dequantize_block_nc_sycl` / `convert_unary_nc_sycl` (`ggml/src/ggml-sycl/convert.cpp:553-650`): one 256-work-item work-group per (row, head), 2 values per work-item for dequantize, so a 128-value row leaves 192 of 256 work-items idle. volta-fa fixed the same geometry in CUDA; read its patches as the reference (read-only): `/home/p51/Github/volta-fa/third_party/patches/0005-CUDA-fewer-idle-threads-when-converting-short-non-co.patch`, `0006-CUDA-vectorized-dequantization-of-non-contiguous-q4_.patch`, design `/home/p51/Github/volta-fa/docs/design/P8-quantised-kv-decode.md` (Changes 2 and 3), and its microbenchmark `/home/p51/Github/volta-fa/bench/micro/dequant_nc.cu`.

SYCL on this laptop's CPU: `bin/sycl-cpu-tbo.sh` and `docs/TOOLCHAIN.md` "SYCL on the CPU" (from p6a; in your base).

Harness notes (from p6a, `bench/results/sycl-cpu-fa-baseline-20261010/README.txt`): the "p6a subset" is the exact one-line filter `P='...'` in that README (85 cases, 85/85 pass on the unpatched clone). Export `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/<ticket-name>-sycl-cache` (your own dir; never another worker's) to reuse CPU-JIT programs across runs; `GGML_SYCL_MKL_FA_DEBUG=1` prints `[FA-DISP]` lines naming the kernel each case ran on. The first build of the scratch clone takes ~10 min at `-j 8`; another worker may be building at the same time, so expect it to be slower. Whole-suite FA runs are not feasible on the CPU (two cases took 50 min before per-kernel images); always filter.

## Design (manager decided, do not change)
1. **Geometry (port of 0005)** in `dequantize_block_nc_sycl` and `convert_unary_nc_sycl`: for rows shorter than 512 values (dequantize) / 256 values (convert), use a 2-D local range `(rows_per_wg, items_per_row)` with `items_per_row` = values-per-row / values-per-item rounded up to a multiple of `WARP_SIZE` (the backend's `GGML_SYCL_WARP_SIZE`, 16 for Intel), and `rows_per_wg = 256 / items_per_row`; the row index comes from group and local id. Longer rows, and the contiguous `to_fp16` paths, launch exactly as today (byte-identical launch parameters). The kernel bodies' arithmetic is unchanged. Keep the existing `downsample_sycl_global_range` guard for > INT_MAX global sizes.
2. **Vectorised q4_0/q8_0 (port of 0006)**: a dedicated non-contiguous kernel for q4_0 and q8_0 that converts 8 values per work-item with 2-byte aligned loads (ggml blocks are only 2-byte aligned) and 8/16-byte stores; same arithmetic as `dequantize_q4_0`/`dequantize_q8_0`, so output is bit-identical. Fall back to the generic kernel when a row is not a whole number of blocks or the work-item count exceeds INT_MAX. Both `sycl::half` and `float` destinations (whatever `ggml_get_to_fp16_nc_sycl` needs).
3. **Microbenchmark** `bench/micro/dequant_nc_sycl.cpp`: copy the upstream kernel bodies byte for byte (as volta-fa's `dequant_nc.cu` does), run upstream geometry, design 1 and design 2 on the FA K/V view layout `ne = [D, n_kv, n_head_kv]`, row stride `n_head_kv*D`, for D 64/128/256, n_kv 4096/16384, n_head_kv 4/8, types q8_0 and q4_0; **compare outputs bit for bit** against upstream and exit non-zero on any mismatch; print per-variant time (median of N reps, `--reps`) plus a device-to-device copy of the output size as a bandwidth reference. Build line in the file header; one TU, icpx `-fsycl`, JIT so it runs on the CPU device here and on the A770 later.

## Files you may touch
- `third_party/llama.cpp/ggml/src/ggml-sycl/convert.cpp` (and `convert.hpp` only if a declaration must change; say why)
- `third_party/patches/0003-SYCL-fewer-idle-work-items-converting-short-non-contiguous-rows.patch`, `third_party/patches/0004-SYCL-vectorized-dequantization-of-non-contiguous-q4_0-q8_0.patch` (new; `git -C third_party/llama.cpp diff` split in two, each with a subject line, a why paragraph, and "Tested on: Intel OpenCL CPU device only")
- `bench/micro/dequant_nc_sycl.cpp` (new), `bench/results/p8-cpu-bitexact-20261010/README.txt` (new)
- `docs/design/P8-sycl-kv-conversion.md` (new, ≤ 80 lines: what, why, what was verified where, what waits for the A770)
Nothing else. (Patch numbers 0001/0002 belong to P-6.)

## Steps
1. Microbenchmark first, against unmodified upstream bodies: build, run on the CPU device, all variants bit-exact.
2. `xe-report` the microbenchmark for acm-g10 (static): SIMD, GRF, spill for each kernel variant; spill must be 0.
3. Patch `convert.cpp` (design 1, then 2). Build llama.cpp via `bin/sycl-cpu-tbo.sh` and run `-o FLASH_ATTN_EXT -p` cases with q8_0/q4_0 K/V and 3–16 query tokens, plus `-o CPY` and `-o GET_ROWS` (they share conversion helpers); record before/after counts (before = unpatched build, run first).
4. Negative control: break one index in the vectorised kernel → microbenchmark reports a mismatch AND an FA q8_0 case fails; revert; both pass.

## Must pass before you finish
- microbenchmark: bit-exact for every variant/shape (paste the table); exits 0; negative control exits non-zero
- `bin/sycl-cpu-tbo.sh` runs from step 3: before/after summary lines, no new failures
- xe-report table for the microbenchmark kernels (spill 0)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, repo `ctest` summary line

## Done means
Two patches, a bit-exact microbenchmark ready to time on the A770, CPU-device correctness evidence, a static acm-g10 report, and a short design doc. Final report: files changed; commands with result lines; tables; what was NOT verified (A770 timing, end-to-end decode). Do not decide whether to ship.
