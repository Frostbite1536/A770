# Ticket p11b-sycl-fa-fp16-a770-geometry: rerun the biased-V FA cases on the CPU SYCL device with A770 launch geometry

Your current directory is a git worktree of A770 (branch `p11b-sycl-fa-fp16-a770-geometry`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10). This ticket measures numerics; it does not fix any kernel. Do not use SIMD16 DPAS, 2D block loads, TF32, FP8, FP64, or Xe2/PVC code. If you think you need to change a kernel, STOP and explain.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path; `A770_TOOLS=/home/p51/Github/A770/tools`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write in this worktree. Inside scratch clones, `git apply` / `git diff` are fine.
- Do not loosen tolerances. Do not edit or delete existing test cases.
- Never invent numbers. CPU-device results are numerics evidence for the same SYCL kernel source, not A770 results. Timing: not run.
- Never kill a process you did not start. Builds are long (about 12 min each at `-j 8`): rerun after a timeout. Another project may be using this machine.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Read `bench/results/p11-sycl-fa-fp16-20261011/README.txt` in full, especially "A770 parallel_blocks", "CPU launch geometry" and "Manager verification". p11 found biased-V cases pass 16/16 against the FP32 CPU reference with `GGML_SYCL_F16` ON and OFF. But the CPU device reports max_work_group_size 8192 over 8 compute units. So `max_wg_per_cu` (`ggml-sycl.cpp:248-249`) is 1024, and `launch_fattn` (`fattn-common.hpp:1061-1104`) splits every TILE case into 64-key partial sums. On an A770 (nsm=32, max_wg_per_cu=2, assuming max_work_group_size 1024), the same cases sum 1024-16384 keys per accumulator. p11 did not test the long sums. This ticket does, by making the CPU device use A770 launch geometry.

## Design (manager decided, do not change)
1. **Test-only geometry override.** Extend `third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch`, inside the existing `GGML_SYCL_TEST_CPU=1 && device.is_cpu()` block in `ggml_sycl_init`. If env `GGML_SYCL_TEST_NSM` is set to a positive integer, use it as `nsm`. If env `GGML_SYCL_TEST_MAX_WG_PER_CU` is set to a positive integer, use it as `max_wg_per_cu`. Set `max_wg_per_cu` after its normal assignment, so the override wins. Print one stderr line per CPU device with the effective `nsm` and `max_wg_per_cu` (for example `GGML_SYCL_TEST_CPU: launch geometry nsm=32 max_wg_per_cu=2`). Without the env vars, behaviour is exactly as today. Nothing changes for GPU devices.
2. **Harness F16 switch.** `bin/sycl-cpu-tbo.sh`: configure with `-DGGML_SYCL_F16=${SYCL_CPU_F16:-ON}` (default unchanged). Allow only ON or OFF; anything else exits 2. If the existing build dir was configured with a different value, reconfiguring must apply the new value (cmake -D does). Document both env vars in `docs/TOOLCHAIN.md` "SYCL on the CPU" (≤ 6 lines).
3. **Scratch clones** under `third_party/p11b/` (gitignored), each = pin + the updated test-only patch + 0005 (`third_party/patches/0005-*.patch`):
   - **A**: `GGML_SYCL_F16=ON` (as shipped).
   - **B**: `GGML_SYCL_F16=OFF` (VEC FP32; TILE still half2).
   Add one more scratch-only diagnostic to both clones, archived as a diff but never part of any shipped patch. In `launch_fattn` (`fattn-common.hpp`), print `[FA-PB] parallel_blocks=<p> ntiles_total=<T> nbatch_fa=<n> ne11=<kv>` to stderr once per launch when `GGML_SYCL_MKL_FA_DEBUG` is set. In `tests/test-backend-ops.cpp`, print the NMSE `err` for passing cases too, using the same format as the failure line but ending ` (pass)`. These two prints are the only scratch source changes beyond the patches.
4. **Runs** (use `bin/sycl-cpu-tbo.sh` with `LLAMA=<clone>`, `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11b-sycl-cache`, `GGML_SYCL_MKL_FA_DEBUG=1`). For each clone A and B, run the 16 `v_bias=2` cases under three geometries:
   - **g-cpu**: no overrides (repeat of p11; must still pass 16/16 for A, which also checks that the prints changed nothing);
   - **g-a770**: `GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2`;
   - **g-min**: `GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=1` (the longest sums launch_fattn can choose at nsm=32; brackets the unverified max_work_group_size assumption).
   Then run the p6a subset `P` (exact filter in `bench/results/sycl-cpu-fa-baseline-20261010/README.txt`) on clone A under g-a770.
   For every biased-V run, record a table: D, kv, nb, route (`[FA-DISP]`), parallel_blocks (`[FA-PB]`), keys per accumulator (kv/parallel_blocks for TILE), pass/fail, NMSE.
5. **Override check.** The g-a770 `[FA-PB]` values must match the "Numeric A770 table" rows for biased-V in the p11 README (TILE p=16 for nb 4/16 and p=2 for nb 512; VEC p=16). If they differ, explain from source; do not change the formula.

## Files you may touch
- `third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch` (geometry override only; regenerate it with `git diff` from a clone so it still applies to the pin)
- `bin/sycl-cpu-tbo.sh` (SYCL_CPU_F16 only)
- `docs/TOOLCHAIN.md` ("SYCL on the CPU" section only)
- `bench/results/p11b-sycl-fa-fp16-geometry-20261011/` (new): `README.txt` with commands, per-run tables, wall times, the two scratch diagnostic diffs; gzipped logs
- `docs/research/llama-cpp-arc-fa-dispatch.md` §7 only: append a "With A770 launch geometry" paragraph (≤ 10 lines, facts and pointers)
Nothing else.

## Must pass before you finish
- Updated test-only patch applies to a fresh pin clone alone and together with 0001-0005 (`git apply --check`)
- `bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P"` on a clone with the updated test-only patch and no geometry env: 85/85, as before (harness default path still works)
- `P` on clone A under g-a770: report the result line (whatever it is; a failure here is a finding, not something to fix)
- all six biased-V tables complete (A/B × g-cpu/g-a770/g-min)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, repo `ctest` summary line

## Done means
We know whether the shipped SYCL FA kernels (VEC and TILE separately) pass the biased-V cases against an FP32 reference when the KV range is split the way an A770 would split it, with the NMSE margin. Final report: files changed; commands with result lines; the tables; what was NOT verified (A770 numerics, A770 max_work_group_size, real-model V statistics, any fix). Do not propose or decide the fix.
