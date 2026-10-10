# Ticket @NAME@: @ONE-LINE GOAL@

Your current directory is a git worktree of A770 (branch `@BRANCH@`, based on `main` @ `@SHA@`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. XMX only through DPAS at sub-group size 8: SYCL `joint_matrix` 8x8x16 (f16/bf16 in, f32 accumulate) or the OpenCL SIMD8 builtins with an `int8` A operand. Do NOT use SIMD16 DPAS forms or shapes, 2D block loads/stores, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC (sycl-tla, XeTLA). If you think you need one, STOP and explain.
- Every kernel declares its sub-group size. QK^T accumulates in FP32.
- The toolchain lives in the MAIN checkout. Start every shell command with `export PATH=@REPO@/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Use `@REPO@/tools/` by absolute path. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/xe-report.py <file> [--expect-dpas REGEX]` and paste its table. Spill must be 0; XMX kernels must show DPAS > 0 at SIMD 8.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/a770/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no A770 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
@WHY. Link the invariant IDs. Give file:line anchors. For optimisation tickets paste the profile summary (kernel, global/local range, sub-group size, GRF mode, spill, SLM, threads occupancy, XVE active/stalled %, memory bandwidth, DPAS count).@

## Design (manager decided, do not change)
@tile sizes, sub-group size, work-group shape, SLM layout/padding, GRF mode, register/SLM budget, target occupancy@

## Files you may touch
- @path@: @what may change@
Nothing else.

## Steps
1. @...@

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build`
- `ctest --test-dir build -R @focused tests@ --output-on-failure`
- `bin/check-forbidden.sh`
- `python bin/xe-report.py @kernel files@` (kernel tickets only)

## Done means
@exact state@. Final report: files changed; each command with its result line; xe-report table; which negative controls failed as expected; what was NOT verified. Do not decide whether to ship.
