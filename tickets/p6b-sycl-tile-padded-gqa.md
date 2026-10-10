# Ticket p6b-sycl-tile-padded-gqa: phantom-head support + padded GQA dispatch in the SYCL TILE FA kernel

Your current directory is a git worktree of A770 (branch `p6b-sycl-tile-padded-gqa`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone), with the test-only CPU patch from p6a applied.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. This ticket changes a non-XMX SYCL kernel; do not add XMX code. Do NOT use SIMD16 DPAS, 2D block loads, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC. Do not change any kernel's sub-group size. If you think you need one of these, STOP and explain.
- The toolchain lives in the MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/oneapi-2026.1`, `tools/neo`), read-only, by absolute path. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; there is no network.
- Do NOT commit, push, or run git commands that write. Read-only git is fine; `git apply` / `git diff` inside the scratch clone `third_party/llama.cpp` are fine.
- Do not loosen tolerances and do not edit or delete existing test cases in `tests/test-backend-ops.cpp`; only ADD the block specified below. Report failures as they are.
- Never invent numbers. There is no A770 here: timing is "not run". CPU-device results are correctness evidence only.
- Never kill a process you did not start. If a long build/test command is cut off by a timeout, rerun it.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Read `docs/design/P6-sycl-padded-gqa.md` (the design, decided) and `docs/research/llama-cpp-arc-fa-dispatch.md` §1–2. In short: the SYCL TILE kernel packs `ncols2` Q heads of one KV head per work-group but derives the KV head as `head0 / gqa_ratio` (`ggml/src/ggml-sycl/fattn-tile.hpp:725-732`), so ncols2 must divide G; dispatch therefore uses ncols2 = 1 or 2 for G = 3, 5, 6, 7 and streams K/V up to G times. `launch_fattn` already sizes the grid as `ceil(G/ncols2) * n_kv_heads * ne03` (`fattn-common.hpp:1018-1021, 1109`). CUDA's MMA kernel shows the indexing to copy (`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1892-1917`, phantom masks at `:1300` and `:1742`).

Harness: `bin/sycl-cpu-tbo.sh [test-backend-ops args]` builds and runs llama.cpp's SYCL backend on the Intel OpenCL CPU device (see `docs/TOOLCHAIN.md` "SYCL on the CPU"). Use it for every correctness run.

## Design (manager decided, do not change)
In `ggml/src/ggml-sycl/fattn-tile.hpp`:
1. **Indexing** (replace `:725-732`): `ntiles_z_gqa = (gqa_ratio + ncols2 - 1)/ncols2; sequence = z/(ntiles_z_gqa*ne12); kv_head = (z/ntiles_z_gqa) % ne12; zt_gqa = z % ntiles_z_gqa; head0 = kv_head*gqa_ratio + zt_gqa*ncols2;` K/V base pointers use `kv_head`. Everything else that uses `head0`/`sequence` keeps its meaning. When ncols2 divides G this must be the identical mapping to today (state the algebra in a comment, one line).
2. **Phantom heads** (`zt_gqa*ncols2 + c >= gqa_ratio`; only possible when ncols2 > 1): Q load writes zeros to `Q_tmp` and does not read global Q; the sink read/adjustment is skipped; the output and `dst_meta` writes are skipped with `continue` (not `return`: a later column in the same loop can be a real head).
3. **Dispatch** (`launch_fattn_tile_switch_ncols2`, DV ≤ 256 branch only, before the `% 8` ladder): `if (use_gqa_opt && G is not a power of two && G < 8 && Q->ne[1]*next_pow2(G) <= GGML_SYCL_FATTN_TILE_PAD_MAX_COLS)` launch with ncols2 = next_pow2(G) (3→4; 5, 6, 7→8). Define `GGML_SYCL_FATTN_TILE_PAD_MAX_COLS 32` as a named constant with a comment: tune from A770 measurements. DV = 512 branch unchanged. No new template instances.
4. No other kernel, no `fattn.cpp` selection change, no `launch_fattn` change.

In `tests/test-backend-ops.cpp`, after the block ending "KV not a multiple of the compaction workgroup size." (around `:11696`), ADD exactly volta-fa's block plus a quantised line:
```
    // Non-power-of-two GQA ratios at small batch sizes: backends may pad the Q heads per tile
    // to the next power of two and mask the extra heads (including their attention sinks).
    for (int hs : { 128, 256 }) {
        for (int gqa : { 3, 5, 6, 7 }) {
            for (int nb : { 1, 2, 3, 4, 8, 16 }) {
                for (bool sinks : { false, true }) {
                    test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 2, {gqa, 1}, 4096, nb, true, sinks, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
                }
            }
            for (int nb : { 3, 8 }) {
                test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 2, {gqa, 1}, 4096, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
            }
        }
    }
```
(Check the constructor's parameter order against `:8369` and adapt only argument spelling if the signature differs; say so.)

## Files you may touch
- `third_party/llama.cpp/ggml/src/ggml-sycl/fattn-tile.hpp`, `third_party/llama.cpp/tests/test-backend-ops.cpp` (scratch clone)
- `third_party/patches/0001-SYCL-FA-TILE-phantom-heads-and-padded-GQA.patch` (new: `git -C third_party/llama.cpp diff -- ggml/` with a header: subject line, one paragraph why, "Tested on: Intel OpenCL CPU device only")
- `third_party/patches/0002-tests-FA-non-power-of-two-GQA-small-batch.patch` (new: the test-backend-ops hunk only)
- `bench/results/p6b-cpu-correctness-20261010/` (new): `README.txt` with commands, counts, wall times; gzipped logs
Nothing else. The test-only CPU patch stays out of 0001/0002.

## Steps
1. Before changing anything: run `bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p '<filter for the new block>'` with ONLY the test block added (kernel unpatched). Record counts (expected: pass, since dispatch never pads today). This is the "before" baseline.
2. Implement design items 1–2 (kernel). Run the full FA suite subset p6a used plus the new block. Existing cases must be unchanged (pass). Record counts.
3. Implement item 3 (dispatch). Rerun the same. Record counts.
4. **Negative controls (each must FAIL, then pass after revert; record counts):**
   a. phantom output write not skipped (remove the `continue` guard);
   b. phantom sink read not guarded;
   c. K/V base using `head0 / gqa_ratio` instead of `kv_head` with padding on.
5. `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, and this repo's `ctest`.
6. Write the two patch files and the results README.

## Must pass before you finish
- New block + existing FA cases (p6a's set) pass on the CPU SYCL device with the patch; summary lines pasted for steps 1–3
- All three negative controls fail as expected and recover
- 0001 and 0002 apply cleanly to the pinned commit without the test-only patch: `git clone -q --no-checkout third_party/llama.cpp /tmp/p6b-check && git -C /tmp/p6b-check checkout -q 79e2e74 && git -C /tmp/p6b-check apply --check <0001> <0002>` (a fresh clone in /tmp is allowed; do not touch the scratch clone's history)
- `git -C third_party/llama.cpp diff --stat` lists exactly the files of the test-only patch, 0001 and 0002
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, repo `ctest` summary line

## Done means
Patches 0001/0002 exist; correctness evidence on the CPU SYCL device is recorded with before/after counts and three failing negative controls. Final report: files changed; every command with its result line; counts; wall times; what was NOT verified (A770 timing, AOT/register report for acm-g10, real-GPU numerics). Do not decide whether to ship.
