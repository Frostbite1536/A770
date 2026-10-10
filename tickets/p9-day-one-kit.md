# Ticket p9-day-one-kit: A/B harness for the SYCL patch series + A770 box runbook

Your current directory is a git worktree of A770 (branch `p9-day-one-kit`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10). This ticket writes scripts and docs, no kernels.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write in this worktree. The script you write will create its own scratch clones under its WORK dir (gitignored); that is allowed when you smoke-test it.
- Never invent numbers. There is no A770 here: every timing in your report is "not run". The CPU-device smoke test proves plumbing only.
- Never kill a process you did not start. Builds are long: rerun after a timeout.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
On the A770 box (arriving in about a week) day one must be measuring, not scripting. volta-fa's equivalent is `/home/p51/Github/volta-fa/bench/patches/run.sh`, `docs/CLOUD_V100.md` §0 and `prompts/v100-session-2.md`: read them (read-only) and mirror their structure and safeguards (cumulative variants on one pinned commit, ccache, interleaved rounds, GPU state logged per round, refusal on wrong hardware/toolchain unless `FORCE=1`, `bench/analyze.py` tables). Our patches: `third_party/patches/0001`–`0002` (P-6, SYCL TILE phantom heads + padded GQA, with tests), `0003`–`0004` (P-8, SYCL conversion geometry + vectorised q4_0/q8_0). `third_party/patches/test-only/` is NEVER applied by this script. Microbenchmark: `bench/micro/dequant_nc_sycl.cpp`. Phase 0 kit: `bench/phase0.sh` (reuse its env capture; do not duplicate it).

## Design (manager decided, do not change)
`bench/patches/run.sh`:
- Variants (cumulative, SYCL backend): `upstream` (none), `p6` (0001-0002), `conv` (0001-0003), `conv8` (0001-0004). Plus one non-cumulative, opt-in variant `grf` = upstream + `third_party/patches/p10-draft-SYCL-FA-TILE-grf-size-automatic.patch` (P-10: does halving threads/XVE to cut spills help at runtime?), included in `perf` only when listed in `VARIANTS`. Plus a `vulkan` reference build of upstream (INV-PERF-002: the faster backend is the baseline), used only in perf/llama-bench steps.
- SYCL build per variant: `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx`, targets `test-backend-ops llama-bench`; oneDNN left at its default (ON) on the box. `BUILD_FLAGS_EXTRA` env appended (the CPU smoke test uses it to drop `DEVICE_ARCH`, oneDNN and Level Zero).
- Steps (`STEPS`, default `env micro build test perf lb`): `env` = `bench/phase0.sh` with `STEPS=env` into the same OUT; `micro` = build + run `dequant_nc_sycl` (bit-exact check must pass, timings recorded); `test` = `test-backend-ops test -o FLASH_ATTN_EXT` on SYCL0 for every variant (+ `-o CPY -o GET_ROWS` for conv/conv8) with pass/fail counts per variant; `perf` = `test-backend-ops perf -o FLASH_ATTN_EXT -p <P-6/P-8 case filter>` interleaved over variants for `ROUNDS` (default 3), GPU state (`intel_gpu_top -J -s` one sample, or "n/a") logged per round; `lb` = for each model in `MODELS`, `llama-bench -fa on -p 1,2,4,8,16 -n 0 -d 4096,16384` (P-6) and `-ctk q8_0 -ctv q8_0 -p 4,8 -n 32 -d 16384` (P-8), interleaved over `upstream p6 conv8 vulkan`, `LB_ROUNDS` (default 2).
- `summary.md` via `bench/analyze.py` (tbo and lb modes) plus the micro table and per-variant correctness counts.
- Guard: refuses unless `sycl-ls` shows an A770 and `icpx` is 2026.1, or `FORCE=1`.
- `ONEAPI_DEVICE_SELECTOR`, `GGML_SYCL_*` env vars the run depends on are written to `OUT/env-vars.txt`.

`docs/A770_BOX.md` (new runbook, ≤ 120 lines): §0 quick path (exact command order for the first session: setup checklist from `docs/TOOLCHAIN.md` "A770 box", `bench/phase0.sh`, `bench/patches/run.sh`, `bin/llama-sycl-fa-static.sh` cross-check, what to copy home, what never to commit), §1 BIOS/driver checklist (ReBAR, kernel driver, versions to record), §2 troubleshooting stubs (device not found, DEVICE_LOST/watchdog per llama.cpp SYCL.md `GGML_SYCL_FA_ONEDNN_MAX_KV`, Level Zero vs OpenCL). Models: the four in `bench/models.example.txt`.

`prompts/a770-session-1.md` (new): a guided-session prompt modelled on volta-fa's `prompts/v100-session-2.md`: what to read first, the checklist, the acceptance criteria from `docs/design/P6-sycl-padded-gqa.md` and `docs/design/P8-sycl-kv-conversion.md` (quote them; if a design has no explicit "Accept if", write "manager to define before the session" rather than inventing one), and the rules (INV-PERF-001/002, not-run labelling, never commit raw logs or models).

## Files you may touch
- `bench/patches/run.sh` (new), `docs/A770_BOX.md` (new), `prompts/a770-session-1.md` (new), `.gitignore` (add the script's WORK dir if not covered)
Nothing else.

## Must pass before you finish
- `shellcheck -x bin/*.sh bench/*.sh bench/patches/*.sh`, `bash -n bench/patches/run.sh`, `bin/check-forbidden.sh`
- **CPU smoke test of the plumbing**: `FORCE=1 VARIANTS="upstream p6" STEPS="build test" BUILD_FLAGS_EXTRA="<drop DEVICE_ARCH/oneDNN/L0 as p6a's harness does>" ` with the env and test-only CPU patch from `bin/sycl-cpu-tbo.sh` applied to the scratch clones (add an explicit `CPU_SMOKE=1` switch that applies `third_party/patches/test-only/*` on top; default off, and say in the header it is for this smoke test only), filtered to a handful of FA cases with `TEST_FILTER`; paste the per-variant counts. If the build time makes even this impossible within the session, STOP and report how far it got.
- repo `ctest` summary line

## Done means
The A/B harness, runbook and session prompt exist; the harness's build/test plumbing ran end to end on the CPU device for two variants. Final report: files changed; commands with result lines; what was NOT verified (everything on the A770). Do not decide whether to ship.
