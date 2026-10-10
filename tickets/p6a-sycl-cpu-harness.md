# Ticket p6a-sycl-cpu-harness: run llama.cpp's SYCL FlashAttention tests on this laptop's CPU

Your current directory is a git worktree of A770 (branch `p6a-sycl-cpu-harness`, based on `main` @ `095ddfc`). Work only here. A clone of llama.cpp at the pinned commit (`bench/LLAMA_COMMIT`, 79e2e74) is already checked out at `third_party/llama.cpp` inside this worktree (gitignored).

## Rules (mandatory, repeated in every ticket)
- Target hardware is Intel Arc A770 (Xe-HPG, acm-g10). This ticket writes no kernels; do not change any llama.cpp kernel code except where the negative control below says so, and revert that.
- The toolchain lives in the MAIN checkout `/home/p51/Github/A770`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. icpx/icx, oneMKL SYCL and the OpenCL CPU runtime are in `/home/p51/Github/A770/tools/oneapi-2026.1` (use by absolute path; read-only). Never install anything; there is no network.
- Do NOT commit, push, or run git commands that write (in this worktree or in the llama.cpp clone). Read-only git (`git diff`, `git status`, `git apply --check`) is fine. `git apply` of your own patch inside `third_party/llama.cpp` is fine (it is a gitignored scratch clone).
- Do not loosen tolerances or edit existing tests to make them pass; report failures as they are.
- Never invent numbers. Anything that needs an A770 is "not run". A CPU result is a CPU result: never describe it as GPU behaviour.
- Do not read .env, tokens, or model weights.
- Never kill a process you did not start. Long builds: if a command is cut off by a timeout, rerun it; ninja resumes.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
There is no Arc GPU here, but llama.cpp's SYCL kernels can run on Intel's OpenCL **CPU** device (`sycl-ls` with the env below shows "Intel(R) OpenCL ... Xeon E3-1505M v6", aspects include fp16, sub-group sizes 4 8 16 32 64). The next ticket (P-6) changes the SYCL TILE FlashAttention kernel (`ggml/src/ggml-sycl/fattn-tile.hpp`), so we need `test-backend-ops -o FLASH_ATTN_EXT -b SYCL0` comparing the SYCL backend against the CPU backend, running here. See `docs/research/llama-cpp-arc-fa-dispatch.md` for why.

Known obstacle: ggml-sycl only accepts GPU platforms. `ggml/src/ggml-sycl/dpct/helper.hpp:911-978` (`get_preferred_gpu_platform_name`) throws "can not find preferred GPU platform" when no GPU platform exists. `dev_mgr()` (`:1033`) always adds the default device first. There may be further GPU-only assumptions (arch detection, Level Zero, device info); find them by running.

Environment for running SYCL on the CPU:
```
O=/home/p51/Github/A770/tools/oneapi-2026.1
export OCL_ICD_VENDORS=$O/etc/OpenCL/vendors LD_LIBRARY_PATH=$O/lib ONEAPI_DEVICE_SELECTOR=opencl:cpu
```

## Design (manager decided, do not change)
- **Test-only patch, env-gated.** All llama.cpp changes needed to run on the CPU device live in ONE patch, active only when `GGML_SYCL_TEST_CPU=1` is set. With the variable unset, behaviour must be byte-identical to upstream. Keep it minimal; it is test infrastructure and is never sent upstream.
- **Build**: JIT (no `GGML_SYCL_DEVICE_ARCH`, so the default spir64 target runs on the CPU device), `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$O/bin/icx -DCMAKE_CXX_COMPILER=$O/bin/icpx -DCMAKE_PREFIX_PATH=$O`, target `test-backend-ops` only, build dir `third_party/llama.cpp/build-sycl-cpu`. If a configure/link dependency is missing from `$O`, STOP and name it.
- **Script** `bin/sycl-cpu-tbo.sh [test-backend-ops args...]`: sets the env above plus `GGML_SYCL_TEST_CPU=1`; checks the patch is applied to `${LLAMA:-third_party/llama.cpp}` (`git apply --check --reverse`) and applies it if not; configures/builds if needed; runs `test-backend-ops <args>` with `-b SYCL0` defaulted when no `-b` is given. It must fail loudly (non-zero) if SYCL0 is not the OpenCL CPU device or if the build fails. shellcheck-clean, `set -euo pipefail`, STOP on a failed `cd`.

## Files you may touch
- `third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch` (new; produce with `git -C third_party/llama.cpp diff`, header comment saying what it is for and that it is not for upstream)
- `bin/sycl-cpu-tbo.sh` (new)
- `docs/TOOLCHAIN.md`: one short section "SYCL on the CPU (correctness only)" after "What works here"
- `bench/results/sycl-cpu-fa-baseline-20261010/` (new): `README.txt` (env, commands, counts, wall time, list of failing/unsupported case groups) and `fa-test.txt.gz` (full log)
- `third_party/llama.cpp/**` in this worktree: only via applying/reverting the patch and the temporary negative control
Nothing else.

## Steps
1. Build (expect a long first build). Make the minimal env-gated patch so `test-backend-ops -o FLASH_ATTN_EXT -b SYCL0` starts on the CPU device.
2. Write `bin/sycl-cpu-tbo.sh`. Run `bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT` (whole FA suite). If the full suite would take more than ~3 hours, run a documented subset with `-p` that still covers head dims 64/128/256, GQA ratios 1/2/4/8 (and any non-power-of-two the suite has), F16 and q8_0/q4_0 K/V, and 1..32 query tokens; record the exact filter.
3. **Negative control (required)**: temporarily break the TILE kernel (e.g. drop the `* scale` on Q at `fattn-tile.hpp:833`), rebuild, run a TILE-routed subset, confirm FAILs appear; revert exactly (show `git -C third_party/llama.cpp diff --stat` equals the test patch only), rebuild, confirm passes return. Report counts for each run.
4. Record which FA cases the SYCL backend reports as unsupported on this device, and which kernel (TILE/VEC/MKL/oneDNN) the suite exercised, as far as you can tell (oneDNN is off by build flag).
5. Write the results README and the TOOLCHAIN section.

## Must pass before you finish
- `bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT` (or the documented subset) completes; summary line pasted
- the negative control fails, then passes after revert
- `shellcheck -x bin/*.sh` and `bin/check-forbidden.sh`
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build` in this worktree (summary line)
- With `GGML_SYCL_TEST_CPU` unset, the patched helper behaves as upstream: state how you verified this (reading the diff is acceptable if explained)

## Done means
The script and patch exist, the FA suite (or documented subset) has run on the CPU SYCL device with counts recorded, and the negative control proved the harness catches a broken TILE kernel. Final report: files changed; every command with its result line; pass/fail/unsupported counts; wall times; the negative-control counts; what was NOT verified. Do not decide whether to ship.

## Manager decision (round 2, after your TBB stop)
Your stop was correct. `tbb-devel` (TBB 2023.1 CMake config) is now installed in `/home/p51/Github/A770/tools/oneapi-2026.1`. The manager verified that the configure command from "Design" now completes (`Build files have been written`). Continue from where you stopped: your test-only patch is already applied in `third_party/llama.cpp`. This machine has 8 cores: build with `-j 8`. The first full ggml-sycl build is long; if a command times out, rerun it (ninja resumes). Everything else in the ticket is unchanged.

## Manager decision (round 3, after your `pown` stop)
Diagnosis: `_Z4pownDv32_fDv32_iS_` comes from `get_alibi_slope` (`ggml/src/ggml-sycl/common.hpp:765-777`, `dpct::pow(float, int)` → `sycl::pown`), compiled into TILE and VEC. The CPU JIT vectorises at sub-group 32 and its builtin library has no 32-wide `pown`. The function returns 1.0f before reaching `pown` whenever `max_bias <= 0`.
1. **Allowed (one compile-time exception to "byte-identical when unset")**: in the test-only patch, replace `return dpct::pow(base, exph);` with a scalar loop (`float r = 1.0f; for (int i = 0; i < exph; ++i) r *= base; return r;`). Comment it in the code and in the patch header: CPU-JIT workaround, ALiBi slope only, P-6 does not pad under ALiBi, the rounding difference is within test tolerance. Nothing else in kernel code may change.
2. **Do not pursue the `-cl-opt-disable` numerical failures.** Optimisation stays ON.
3. **Gate before the full suite**: after the rebuild, run one VEC f16 case and one TILE case (D 128, GQA 4, a few tokens, mask on). If both pass, continue with the original Steps 2–5. If either fails numerically with optimisation on, STOP: report the exact case string, the error value the test prints, and which kernel ran (`GGML_SYCL_MKL_FA_DEBUG=1`), and do not try further workarounds.
