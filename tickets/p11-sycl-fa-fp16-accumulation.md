# Ticket p11-sycl-fa-fp16-accumulation: how much does FP16 output accumulation cost the SYCL FA kernels? (CPU SYCL device)

Your current directory is a git worktree of A770 (branch `p11-sycl-fa-fp16-accumulation`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10). This ticket measures numerics; it does not fix any kernel. Do not use SIMD16 DPAS, 2D block loads, TF32, FP8, FP64, or Xe2/PVC code. If you think you need to change a kernel beyond the diagnostic variants below, STOP and explain.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path; `A770_TOOLS=/home/p51/Github/A770/tools`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write in this worktree. Inside scratch clones, `git apply` / `git diff` are fine.
- Do not loosen tolerances. Do not edit or delete existing test cases. The only test change is the port of P100's patch below.
- Never invent numbers. CPU-device results are numerics evidence for the same SYCL kernel source, not A770 results. Timing: not run.
- Never kill a process you did not start. Builds are long (about 10 min each at `-j 8`): rerun after a timeout. Another project may be using this machine.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Read `docs/research/llama-cpp-arc-fa-dispatch.md` §1 and §7 first. In short: SYCL's TILE kernel always sums P·V in `sycl::half2` (`SYCL_FAST_FP16` is defined unconditionally in `ggml-sycl/common.hpp:48`). VEC does the same when `GGML_SYCL_F16` is ON (`fattn-vec.hpp:142-143`). llama.cpp's CPU reference also sums F16 V in FP16 (`ggml-cpu/ops.cpp`, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `VKQ16`), so it cannot judge this.

The P100 sibling project (read-only: `/home/p51/Github/P100/docs/research/pre-hardware-findings-2-2026-10-10.md` §1–2, `/home/p51/Github/P100/third_party/patches/0003-tests-biased-V-flash-attention-cases.patch`) added 16 biased-V FA cases (V + 2 on channels 0..7; hs 128/256; kv 16K/32K; nb 1/4/16/512; 4 Q heads per KV head; no mask). It also made the CPU reference accumulate in FP32 only when `params->use_ref`.

Harness: `bin/sycl-cpu-tbo.sh [test-backend-ops args]` (Intel OpenCL CPU device; `LLAMA=<clone>` selects a scratch clone; see `docs/TOOLCHAIN.md` "SYCL on the CPU"). Use `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11-sycl-cache` and `GGML_SYCL_MKL_FA_DEBUG=1` (prints `[FA-DISP]` lines naming the kernel per case).

## Design (manager decided, do not change)
1. **Patch 0005 = port of P100 0003**, unchanged in substance: the `v_bias` test member, the 16 cases and the `!params->use_ref` gating in `ops.cpp`. It must apply to the pin both alone and on top of 0001–0004 (`third_party/patches/`). Adapt only line offsets.
2. **Scratch clones** under `third_party/p11/` (gitignored via `third_party/`), each = pin + test-only CPU patch (`third_party/patches/test-only/*`) + the listed changes:
   - **R** (reference check): 0005, but with its `ops.cpp` hunk reverted (FP16 reference, new cases).
   - **A** (as shipped on the box): 0005. `GGML_SYCL_F16=ON`.
   - **B** (VEC FP32): 0005, configured with `-DGGML_SYCL_F16=OFF`.
   - **C** (TILE FP32): 0005 plus removing the `#define SYCL_FAST_FP16` line from `common.hpp`, `GGML_SYCL_F16=ON`. If C does not build, record the first compiler error and continue without C. Do not modify fattn-tile.hpp to make it build.
3. **Runs** for each clone: the 16 biased-V cases (filter on `v_bias=2`) and the p6a subset `P` (exact one-line filter in `bench/results/sycl-cpu-fa-baseline-20261010/README.txt`). For every biased-V case record:
   - the kernel route (`[FA-DISP]`);
   - pass/fail;
   - the `ERR = x > y` value when it fails.
   If test-backend-ops prints no error for passing cases, say so; do not add printing to the tests.
4. **parallel_blocks on the A770** (code reading only, no run): from `launch_fattn` (`fattn-common.hpp`) and `ggml_sycl_init` (nsm = max_compute_units/16 = 32 on an A770 with 512 XVEs), compute `parallel_blocks` and keys per accumulator for TILE decode (nb 1) and nb 16. Do it for D=128, 8 KV heads × G 4 (Llama-3.1-8B) at 4K/16K/32K, and for the biased-V shapes. Show the formula with file:line references.

## Files you may touch
- `third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch` (new; header: subject, why, credit "ported from the P100 project, patch 0003", "Tested on: Intel OpenCL CPU device only")
- `bench/results/p11-sycl-fa-fp16-20261011/` (new): `README.txt` with commands, per-clone tables (route, pass/fail, ERR), the parallel_blocks table, wall times; gzipped logs
- `docs/research/llama-cpp-arc-fa-dispatch.md` §7 only: append a "Measured on the CPU SYCL device" paragraph (≤ 12 lines, facts and pointers to the results README)
Nothing else.

## Must pass before you finish
- 0005 applies to a fresh clone at 79e2e74 alone, and on top of 0001–0004 (`git apply --check`; a fresh clone under `/tmp` is fine)
- p6a subset `P` on clone A: 85/85, as upstream (0005 must not change existing cases)
- per-clone biased-V tables complete (C may be "did not build", with the error)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, repo `ctest` summary line

## Done means
We know, on the same SYCL kernel source, whether biased-V long-context cases fail with an FP32 reference. If they do, we know whether VEC, TILE or both are responsible, and whether TILE's FP32 path even builds. Final report: files changed; commands with result lines; the tables; what was NOT verified (A770 numerics, real-model V statistics, any fix). Do not propose or decide the fix.
