# Ticket p14-range-tests-local: activation-range MUL_MAT tests as a bench-only patch, run on the SYCL CPU device

Your current directory is a git worktree of A770 (branch `p14-range-tests-local`, based on `main` @ `5613b45`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. XMX only through DPAS at sub-group size 8: SYCL `joint_matrix` 8x8x16 (f16/bf16 in, f32 accumulate) or the OpenCL SIMD8 builtins with an `int8` A operand. Do NOT use SIMD16 DPAS forms or shapes, 2D block loads/stores, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC (sycl-tla, XeTLA). If you think you need one, STOP and explain.
- Every kernel declares its sub-group size. QK^T accumulates in FP32.
- The toolchain lives in the MAIN checkout. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Use `/home/p51/Github/A770/tools/` by absolute path. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/xe-report.py <file> [--expect-dpas REGEX]` and paste its table. Spill must be 0; XMX kernels must show DPAS > 0 at SIMD 8.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/a770/attention.h`.
- Do NOT commit, push, or run git commands that write in the A770 repo or this worktree. Leave changes uncommitted; the manager commits. Exception: inside your scratch llama.cpp clone under `build/` (see Steps 0) you may `git apply`, `git commit` and `git format-patch` to produce the patch.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no A770 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §4, at the pin (`79e2e74`):
- **SYCL's q8_1:** `ggml-sycl/quantize.hpp` writes `ds = half2(half(d), half(sum))`, so the block sum of raw activations overflows at activations around 1e5, and d underflows around 1e-5.
- **Vulkan:** `block_q8_1` also has an FP16 `ds` (`ggml-vulkan/vulkan-shaders/types.glsl:242`).
- **The fused XMX GEMM** (`ggml-sycl/fused-gemm.cpp`) converts src1 to f16, bf16 or tf32 (`GGML_SYCL_DYNAMIC_PRECISION`). f16 without scaling would overflow above 65504.

volta-fa's local test is `~/Github/volta-fa/third_party/patches-local/0009-tests-MUL_MAT-with-activations-beyond-FP16-s-range-l.patch` (`test_mul_mat_range`). It applies on top of volta-fa's 0008, not on the pin. On a V100: CUDA current path 18/35, volta-fa 0008 35/35.

**CPU reference limits** (`ggml/src/ggml-cpu/ggml-cpu.c`, `type_traits_cpu`): Q4_0/Q8_0 use `vec_dot_type` Q8_0 (FP16 d, underflows at 1e-5, so the reference is invalid there). The K-quants use Q8_K (FP32 d).

These tests fail on unmodified upstream by design, so they must never enter the upstream series.

The SYCL backend runs on the Intel OpenCL CPU device here (`bin/sycl-cpu-tbo.sh`), so the SYCL kernels' numerics can be checked without an A770. CPU-device results are not A770 results; label them.

## Design (manager decided, do not change)
1. **`third_party/patches/bench-only/0002-tests-mul_mat-activation-range.patch`** (against the pin). Create `third_party/patches/bench-only/` and its README as ticket p12 describes if they do not exist yet; this patch goes into every build.
   - Port `test_mul_mat_range` (same class, same `,src1_scale=` suffix, same initialisation).
   - Cases: type_a ∈ {Q4_0, Q8_0, Q4_K, Q5_K, Q6_K}; n ∈ {1, 2, 4, 8, 16, 32} (32 is inside the fused GEMM's N ≤ 64); m ∈ {128, 4096}; K = 1024; scale ∈ {1e5, 1e-5}.
   - Skip scale 1e-5 for Q4_0 and Q8_0, with the reason as a comment.
   - Eval list only.
2. **`bench/phase0.sh`** (manager note: `bench-only/` and its README already exist from p12; add `0002` to the explicit every-build `bench_patches` list in `phase0.sh` and a row to the README, which must say these tests fail on q8_1 paths by design and never go upstream or into an upstream test list). **`test` step:** per backend, one more run, `test-backend-ops test -o MUL_MAT -b $D -p src1_scale`, into `range-$be.txt`.
   - On SYCL0, also run with `GGML_SYCL_XMX_GATHER_TYPES=0` and with `GGML_SYCL_ENABLE_OPT=0`, so a failure can be attributed to a path.
   - Print the pass counts in the run log under a heading saying these fail on q8_1 paths by design.

## Files you may touch
- `third_party/patches/bench-only/0002-tests-mul_mat-activation-range.patch` (new), `third_party/patches/bench-only/README.md`
- `bench/phase0.sh`, `bin/sycl-cpu-tbo.sh` (only if it needs an option to apply bench-only patches; default behaviour unchanged)
- `tests/CMakeLists.txt` (a ctest that runs the range filter on the CPU device only when the oneAPI CPU runtime is installed; otherwise skip with a message)
Nothing else.

## Steps
0. **Harness (manager note).** This worktree has no `third_party/llama.cpp` (gitignored). Make a scratch clone with `git clone --no-hardlinks /home/p51/Github/A770/third_party/llama.cpp build/llama` (it is at the pin `79e2e74`); never write to the main checkout's tree. `LLAMA=$PWD/build/llama bin/sycl-cpu-tbo.sh [test-backend-ops args]` builds and runs on the Intel OpenCL CPU device (see `docs/TOOLCHAIN.md` "SYCL on the CPU"); use `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p14-sycl-cache`. Label every CPU-device result "SYCL on the Intel OpenCL CPU device".
   Check `0002` applies (a) to the pin alone, (b) on top of bench-only `0001`, (c) on top of `third_party/patches/0001`–`0005`, and (d) on top of the upstream variant `bench/patches/run.sh` builds (pin + `0005` + the `0002` cases it injects; see `prepare()` there). Report each.
1. Port the test in a scratch worktree at the pin and export it. The patch must apply alone and on top of 0001–0005.
2. Run the range filter on the SYCL CPU device in three arms: defaults, `GGML_SYCL_XMX_GATHER_TYPES=0`, and `GGML_SYCL_ENABLE_OPT=0`.
   - Report pass/fail per case for each arm, labelled "SYCL on the Intel OpenCL CPU device".
   - The CPU device has no XMX, so the fused GEMM probably never runs there; say what the start-up log reports.
3. Implement the phase0 change and the ctest.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all; the new test may skip without the CPU runtime)
- `shellcheck -x bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`

## Done means
The range cases run per backend and arm in Phase 0, and the CPU-device results are recorded per case.

Final report:
- files changed;
- each command with its result line;
- the per-case CPU-device table;
- what was NOT verified (the A770, Vulkan, the fused XMX GEMM path).

Do not interpret.
