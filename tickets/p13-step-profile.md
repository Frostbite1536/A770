# Ticket p13-step-profile: per-kernel time breakdown of an 8- and 16-token step in the Phase 0 kit

Your current directory is a git worktree of A770 (branch `p13-step-profile`, based on `main` @ `d588d56`). Work only here.

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
3. **Fixture test** (manager note: `unitrace` is not installed on this machine and there is no network, so the fixture will be synthetic; its first line must say so): a unitrace summary excerpt in the format of the installed unitrace release. If no real sample can be produced here, write it from the unitrace documentation and label the fixture as synthetic in its first line. Assert the categories and percentages.

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

## Rework round 2 (manager decision; supersedes Design item 1's single run and the parser's "one file per profile")

The first round is already in this worktree (uncommitted): `bench/phase0.sh` `prof` step, `bin/step-breakdown.py`, `tests/fixtures/step-breakdown/*`, the ctest and the PROFILING.md paragraph. Keep what is there and change only what follows. Everything else in the ticket still holds.

**Problem.** One `llama-bench … -r 1` capture includes the warm-up prompt run and, at depth 16384, a 16384-token depth prefill (`tools/llama-bench/llama-bench.cpp` at the pin, lines 2415–2475: warm-up unless `--no-warmup`; the depth prefill runs once, then later repetitions restore a cached state with `llama_state_seq_set_data`). The summed kernel times would describe the prefill, not the 8/16-token step.

**Design (differential).**
1. For each (backend, p, depth), run unitrace twice with otherwise identical arguments: `-r 1` → `$OUT/prof/<be>-p<p>-d<depth>-r1.txt`, and `-r $((1 + PROF_REPS))` → `$OUT/prof/<be>-p<p>-d<depth>-rN.txt`. `PROF_REPS` defaults to 4; document it in the header comment. Keep the default warm-up in both (it cancels).
2. `bin/step-breakdown.py` pairs the two files and computes, per kernel name, step calls = (calls_N − calls_1) / PROF_REPS and step time = (time_N − time_1) / PROF_REPS. It takes `--reps` (default 4); `phase0.sh` passes `--reps "$PROF_REPS"`. A pair with a missing partner is an error naming the file.
   - Include a kernel when its call delta is > 0. Show calls per step with up to 2 decimals when not integral. A kernel with a call delta of 0 is omitted. A kernel with a negative call delta is an error. A row whose time delta is ≤ 0 stays in the table, marked `(noise)`, and counts as 0 toward the totals and shares.
   - New category `transfer` for memory operations unitrace lists among device timings (pattern: `MemoryCopy|MemoryFill|memcpy|memset|Memcpy|Memset`, checked before every other category). At depth > 0 this includes llama-bench's per-repetition KV state restore, which is not part of a real decode step. Report `transfer` rows and their subtotal, but EXCLUDE them from the step total and from every percentage. Say why under the table, in one line.
   - Categories: add `soft_max` to `norm/elementwise` (llama.cpp's op name), and `quantize` (activation quantisation to q8_1, e.g. `quantize_q8_1`) to `conversion`. The order must keep `dequantize`/`dequant_gemm`/`fused` behaviour as it is now. Keep the comment that the patterns must be checked against real kernel names on the box, and add that SYCL's oneMKL FA path (≥ 32 tokens) would show up as `gemm` → matmul, which does not arise at p = 8/16.
   - Replace the report's preamble with the differential method in two sentences.
3. **Fixture.** Two synthetic files (`r1` and `rN`, `# SYNTHETIC` first line in both) whose difference over reps = 4 gives hand-checkable per-step values. They must include a large one-off "prefill" contribution present equally in both files (it must vanish from the result), one `(noise)` row, and one memory-copy row in `transfer`. Assert the per-step calls/µs, category shares, the transfer exclusion, the noise row, and the missing-partner error. Update the Phase0Test fakes for the 16 runs (2 per combination) and check that both `-r` values are passed.
4. **PROFILING.md:** rewrite the paragraph for the differential method (≤ 10 lines), including the transfer exclusion and that the fixture is synthetic.

Must pass: the same commands as above. Final report: files changed, each command with its result line, the hand calculation for the fixture's per-step values, and what was NOT verified.
