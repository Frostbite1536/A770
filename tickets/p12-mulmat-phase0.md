# Ticket p12-mulmat-phase0: MUL_MAT correctness and 1–16-token timing in the Phase 0 kit (SYCL0 paths and Vulkan0)

Your current directory is a git worktree of A770 (branch `p12-mulmat-phase0`, based on `main` @ `8ab9ea4`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. XMX only through DPAS at sub-group size 8: SYCL `joint_matrix` 8x8x16 (f16/bf16 in, f32 accumulate) or the OpenCL SIMD8 builtins with an `int8` A operand. Do NOT use SIMD16 DPAS forms or shapes, 2D block loads/stores, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC (sycl-tla, XeTLA). If you think you need one, STOP and explain.
- Every kernel declares its sub-group size. QK^T accumulates in FP32.
- The toolchain lives in the MAIN checkout. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Use `/home/p51/Github/A770/tools/` by absolute path. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/xe-report.py <file> [--expect-dpas REGEX]` and paste its table. Spill must be 0; XMX kernels must show DPAS > 0 at SIMD 8.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/a770/attention.h`.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted; the manager commits.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no A770 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §1–§3:
- On the V100, about 95% of the small-batch gap was quantised matmul, not attention.
- `bench/phase0.sh` runs no MUL_MAT at all today (FA support/test/perf and llama-bench only).

At the pin (`79e2e74`), SYCL dense MUL_MAT can take three kinds of path:
- MMVQ/DMMV on reordered weights: `opt_feature.reorder` is set for Intel GPUs (`ggml-sycl.cpp` ≈ line 240) for Q4_0/Q8_0/Q2_K–Q6_K (`ggml_sycl_supports_reorder_*`, ≈ line 4201). `GGML_SYCL_ENABLE_OPT=0` turns it off (≈ line 418).
- MMQ.
- `ggml_sycl_op_mul_mat_sycl`, which for quantised weights first tries the fused XMX dequant-GEMM (`ggml-sycl/fused-gemm.cpp`, called at ≈ line 3134):
  - N ≤ 64 (`GGML_SYCL_FG_MAX_N`), with Arc-A SIMD8 f16/bf16 8×8×16 combinations;
  - type mask `GGML_SYCL_XMX_GATHER_TYPES`, combination mask `GGML_SYCL_XMX_GATHER_SHAPES`;
  - the device's combinations are logged once as `XMX dequant-GEMM combinations available 0x…`.

Upstream's perf list (`tests/test-backend-ops.cpp`, `make_test_cases_perf`) only has MUL_MAT at n ∈ {1, 2, 3, 4, 5, 8, 512} (4096×14336) and Q4_K at n ∈ {1, 2, 3, 4, 8}.

## Design (manager decided, do not change)
1. **New directory `third_party/patches/bench-only/`** for patches the kits apply identically to **every** backend build on the A770 (test cases, development switches); never part of the upstream series.
   - Do not put them in `test-only/`: `bench/patches/run.sh` applies `test-only/*` only for its CPU smoke test, and `test-only/0001` must never be applied on the A770.
   - Add `third_party/patches/bench-only/README.md` listing each patch with provenance, purpose, "never upstream", and whether it goes into every build or only one.
2. **`third_party/patches/bench-only/0001-tests-mul_mat-perf-cases-1-16.patch`** (against the pin). In `make_test_cases_perf`, after the existing `bs` loop, add `test_mul_mat(type_a, GGML_TYPE_F32, m, n, k, {1, 1}, {1, 1})` with:
   - type_a ∈ {Q4_0, Q8_0, Q4_K, Q5_K, Q6_K};
   - (m, k) ∈ {(4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336)};
   - n = 1..16;
   - skip combinations already in the list (no duplicate case names).

   These are the same cases as the P100 repo's ticket p13, so results compare across the sibling repos.
3. **`bench/phase0.sh`:**
   - Apply the "every build" `bench-only/*.patch` files to each backend tree before building, and log each.
   - Add MUL_MAT to the `test` step (`test-backend-ops test -o MUL_MAT -b $D`, per backend).
   - Add a new step `mmperf` to the default STEPS, after `perf`.
4. **`mmperf`:** `ROUNDS` (default 3) interleaved rounds. Each round runs these arms one after another, all with `test-backend-ops perf -o MUL_MAT --output sql -p <filter for the cases in 2>`:

   | arm | backend | environment |
   | --- | --- | --- |
   | `sycl` | SYCL0 | defaults |
   | `sycl-noopt` | SYCL0 | `GGML_SYCL_ENABLE_OPT=0` (no reorder) |
   | `sycl-nofg` | SYCL0 | `GGML_SYCL_XMX_GATHER_TYPES=0` (no fused XMX GEMM) |
   | `vulkan` | Vulkan0 | defaults |

   Also capture the SYCL backend's start-up log once (it prints its `GGML_SYCL_*` settings and the `XMX dequant-GEMM combinations` line) into `sycl-env.txt`.
5. **GB/s summary:** `bin/mm-gbps.py`, with the interface the P100 repo's ticket p13 specifies:
   - median µs and GB/s per (type, m, k) and n, per arm;
   - bytes from the `BLOCKS` table in `~/Github/volta-fa/bin/qmatmul-table.py`;
   - `--peak-gbps` default 560;
   - standard library only, reusing `bench/analyze.py`'s SQL parsing.

   If `~/Github/P100/bin/mm-gbps.py` already exists, copy it verbatim and change only the default peak. The kit writes the table to `$OUT/mmperf.md`.
6. Header comment: the arms, what each environment variable disables (with file anchors at the pin), and the research doc pointer.

## Files you may touch
- `third_party/patches/bench-only/0001-tests-mul_mat-perf-cases-1-16.patch`, `third_party/patches/bench-only/README.md` (new)
- `bench/phase0.sh`, `bin/mm-gbps.py` (new)
- `tests/fixtures/mm-gbps/*` (new: hand-written perf SQL fixture), `tests/CMakeLists.txt` (an `mm_gbps` test; plus a phase0 dry-run check if the repo has one, otherwise a `bash -n` test of the script)
Nothing else.

## Steps
1. Write the patch in a scratch worktree of `third_party/llama.cpp` at the pin under `build/`.
   - Build `test-backend-ops` with `bin/sycl-cpu-tbo.sh`'s CPU-device setup, or CPU-only if that is simpler.
   - Run one new case (q4_K, m=4096, n=16, k=4096) in perf mode to show it exists and runs, and check for duplicate case names.
2. On the SYCL CPU device, confirm that the three SYCL environment variables are read: the start-up log lists them. Report the `XMX dequant-GEMM combinations` line if the CPU device prints one (it will likely report 0x0; say so).
3. Implement the phase0 changes and `bin/mm-gbps.py` with its fixture test.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all)
- `shellcheck -x bench/phase0.sh bench/patches/run.sh bin/*.sh`, `bin/check-forbidden.sh`

## Done means
Phase 0 on the A770 produces:
- MUL_MAT correctness on both backends;
- 1–16-token timings for four arms;
- a GB/s table against 560 GB/s.

Final report:
- files changed;
- each command with its result line;
- the number of new perf cases and the duplicate check;
- the CPU run line;
- the start-up log lines for the three variables;
- what was NOT verified (everything on an A770).

Do not interpret.
