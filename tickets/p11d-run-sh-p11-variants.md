# Ticket p11d-run-sh-p11-variants: add the P-11 candidates and biased-V checks to the day-one A770 kit

Your current directory is a git worktree of A770 (branch `p11d-run-sh-p11-variants`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10). This ticket changes the benchmark kit only, not any kernel or patch.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path; `A770_TOOLS=/home/p51/Github/A770/tools`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write in this worktree.
- Do not loosen tolerances. Do not edit or delete existing test cases.
- Never invent numbers. There is no A770 on this machine: CPU smoke proves plumbing only; A770 results are "not run".
- Never kill a process you did not start. Builds are long (about 12 min each): rerun after a timeout. Another project may be using this machine.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Read first: `docs/design/P11-sycl-fa-fp16-accumulation.md` (including "Accept if"), `bench/patches/run.sh` in full, `docs/A770_BOX.md`, `prompts/a770-session-1.md`, and `bench/results/patches-p9-cpu-smoke-final/` (how the previous kit was smoke-tested). Patch 0005 adds 16 biased-V FA cases (`v_bias=2` in their names) and an FP32 CPU reference. The candidates are `third_party/patches/p11c/F-*.patch` and `S-*.patch`, each applying on top of 0001-0005.

## Design (manager decided, do not change)
1. **Shared test corpus.** Every variant's scratch tree gets patch 0005, including `upstream` and `vulkan`: Vulkan's tests use the same CPU reference. Apply it in `prepare` like the other shared test additions, unless the variant's own patch list already contains it. It changes only tests and the CPU reference under `use_ref`; say so in the script comment. Bump the signature string to `shared-cases-v3`.
2. **New variants.** `fp32f` = 0001-0005 + `p11c/F-*.patch`; `cap` = 0001-0005 + `p11c/S-*.patch`. New default: `VARIANTS="upstream p6 conv conv8 fp32f cap"`. Update the header comment. `grf` stays opt-in and unchanged.
3. **Biased-V correctness row.** The correctness table gets a separate row per SYCL variant for the 16 `v_bias=2` cases (op label `FA-BIASV`). Those cases count only there, never in the main FLASH_ATTN_EXT row, however the run is filtered (for example, run the main FA pass and classify by case name, or run the slice separately; pick one and document it). Failures in FA-BIASV set `failed=1` only for `fp32f` and `cap`; for other variants they are recorded and listed but expected (P-11 Accept-if 2). The summary says this in one line.
4. **Geometry record.** The env step records, for the SYCL0 device, `max_work_group_size` and `max_compute_units`, plus their quotient (`max_wg_per_cu`, the input to `launch_fattn`). Use `sycl-ls --verbose` output if it contains both; otherwise a few-line SYCL or OpenCL query compiled or run from the scratch area. Print them in `summary.md`.
5. **Perf corpus.** Add one perf block next to the existing P9 block: hs {128, 256}, G {4}, kv {32768}, nb {1, 2, 4, 16}, types F16 and Q8_0, 4 KV heads, mask on. Extend the default `PERF_FILTER` kv alternation to `(4096|16384|32768)`. Keep the existing P9 block byte-identical.
6. **llama-bench.** Add `fp32f` and `cap` to the lb variant lists, in run order and in both analysis orders. Add a third tag `tg` with args `-p 0 -n 32 -d '4096,16384,32768'`. Both `small` and `kvq8` stay unchanged.
7. **Docs.** `docs/A770_BOX.md` and `prompts/a770-session-1.md`: add the new variants, the FA-BIASV row, the geometry record and the `tg` tag, plus a pointer to P11's "Accept if" (quote it verbatim in the prompt's acceptance section, as is done for P6/P8). Keep the edits minimal.

## Files you may touch
- `bench/patches/run.sh`
- `docs/A770_BOX.md`, `prompts/a770-session-1.md` (only the additions in 7)
- `bench/results/p11d-run-sh-cpu-smoke-20261011/` (new): README with commands and result lines, summary.md and logs from the smoke run (gzipped where large)
Nothing else.

## Must pass before you finish
- `shellcheck -x bin/*.sh bench/patches/run.sh` with no diagnostics; `bin/check-forbidden.sh`
- CPU plumbing smoke:
  ```
  CPU_SMOKE=1 FORCE=1 STEPS="env build test" VARIANTS="upstream fp32f cap" \
  TEST_FILTER=<a filter covering the 16 v_bias=2 cases plus a few existing cases> \
  BUILD_FLAGS_EXTRA="-DGGML_SYCL_DEVICE_ARCH= -DGGML_SYCL_DNN=OFF -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF" \
  WORK=<fresh scratch dir> bench/patches/run.sh
  ```
  It must complete, and the correctness table must show FA-BIASV rows for all three variants and main FA rows that contain no v_bias cases. Use the CPU environment from `docs/TOOLCHAIN.md` "SYCL on the CPU"; with the CPU's own launch geometry all three are expected to pass. Then run a second smoke with `GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2` exported, VARIANTS="upstream cap", STEPS="test" on the same WORK (it reuses the builds). The upstream FA-BIASV row should then show the p11b failures without setting the exit status, and cap's row should pass. Report both exit codes. If either run differs from this expectation, report it; do not change tests.
  If `CPU_SMOKE`/`FORCE` gating blocks the geometry env vars or the second run, explain what you changed in run.sh to allow it. Only test-only plumbing may change.
- A dry check that `STEPS=env` prints the geometry lines on the CPU device.
- Repo `ctest` summary line.

## Done means
`bench/patches/run.sh` with defaults on the A770 box builds and evaluates `fp32f` and `cap` next to the P-6/P-8 variants. It reports biased-V correctness separately, records the A770's launch geometry, and has decode perf and llama-bench coverage at 32K. Final report: files changed; commands with result lines; the smoke correctness tables; what was NOT verified (anything on an A770).
