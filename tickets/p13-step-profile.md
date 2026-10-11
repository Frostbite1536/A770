# Ticket p13-step-profile: per-kernel time breakdown of an 8- and 16-token step in the Phase 0 kit

Your current directory is a git worktree of A770 (branch `p13-step-profile`, based on `main` @ `8ab9ea4`). Work only here.

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
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §1, §5. volta-fa chose attention work before profiling a whole small-batch step; the real gap was matmul. The per-kernel breakdown (nsys on CUDA, grouped by kernel and grid shape) is what showed it.

This repo's roadmap ranks attention first without such a breakdown. `docs/PROFILING.md` already shows `unitrace --device-timing --kernel-submission` on llama-bench, but no kit step runs it.

## Design (manager decided, do not change)
1. **New opt-in step `prof` in `bench/phase0.sh`.** It is not in the default STEPS and needs `bench/models.txt` and `unitrace` on PATH; otherwise it logs one line and skips.
   - For the first model in `bench/models.txt` and each backend, profile one small-batch step at depth 0 and at 16384:
     ```
     unitrace --device-timing --kernel-submission <llama-bench> -m M -ngl 99 -fa 1 -p 8 -n 0 -d <depth> -r 1
     ```
     and the same with `-p 16`.
   - Save the text output under `$OUT/prof/`. Raw binary result directories are never committed.
   - Flags differ between unitrace releases, so first run `unitrace --help` into `$OUT/prof/unitrace-help.txt`, and fail with a clear message if `--device-timing` is missing.
2. **New `bin/step-breakdown.py`** (standard library only). It parses unitrace's device-timing summary into a table per (backend, p, depth):
   - kernel name, calls, total µs, % of the step;
   - a category column: `matmul` (names matching `mul_mat`, `mmvq`, `dmmv`, `mmq`, `fused`/`dequant_gemm`, oneMKL/oneDNN gemm kernels), `attention` (`flash_attn`, `fattn`, `sdpa`), `conversion` (`dequantize`, `convert`, `cpy`), `norm/elementwise`, `other`.

   The category patterns live in one dictionary at the top of the script, with a comment that they must be checked against real kernel names on the box. Output: `$OUT/prof/breakdown.md`.
3. **Fixture test:** a unitrace summary excerpt in the format of the installed unitrace release. If no real sample can be produced here, write it from the unitrace documentation and label the fixture as synthetic in its first line. Assert the categories and percentages.

## Files you may touch
- `bench/phase0.sh`, `bin/step-breakdown.py` (new), `tests/fixtures/step-breakdown/*` (new), `tests/CMakeLists.txt` (`step_breakdown` test), `docs/PROFILING.md` (a short "Step breakdown" paragraph pointing at the step)
Nothing else.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all)
- `shellcheck -x bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`

## Done means
`STEPS=prof bench/phase0.sh` on the A770 gives the matmul vs attention share of an 8- and 16-token step per backend.

Final report:
- files changed;
- each command with its result line;
- whether the fixture is real or synthetic;
- what was NOT verified (unitrace output format on the box, everything on an A770).

Do not interpret.
