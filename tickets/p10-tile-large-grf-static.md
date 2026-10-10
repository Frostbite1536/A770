# Ticket p10-tile-large-grf-static: do the SYCL TILE FA kernels stop spilling with the large register file? (static)

Your current directory is a git worktree of A770 (branch `p10-tile-large-grf-static`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. Do NOT use SIMD16 DPAS, 2D block loads, TF32, FP8, FP64, or Xe2/PVC code. Do not change any kernel's sub-group size or arithmetic. If you think you need to, STOP and explain.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path; export `A770_TOOLS=/home/p51/Github/A770/tools`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write. `git diff`/`git apply` in the scratch clone are fine.
- Report exactly what IGC reports; nothing here is a runtime measurement. "Fewer spills" is not "faster": say so.
- Never kill a process you did not start. Another worker is building on this machine: keep AOT compiles to at most 2 TUs in parallel.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/llama-cpp-arc-fa-dispatch.md` §6 and `bench/results/llama-sycl-fa-static-20261010/`: every upstream TILE instance at D 128/256 compiles to SIMD32, 128 GRF, 8 threads/XVE, and spills 7.4–14.8 KB. `launch_fattn` has a `use_large_grf` template parameter that attaches `grf_size_automatic` to the kernel (`ggml/src/ggml-sycl/fattn-common.hpp:835-907`); VEC uses it for D ≥ 512 (`fattn-vec.hpp:594`); TILE never does (its six `launch_fattn<...>` calls in `launch_fattn_tile_switch_ncols1`, `fattn-tile.hpp:1081-1147`, take the default `false`). Tool: `bin/llama-sycl-fa-static.sh [OUTDIR]` (reports TILE D 64/128/256 and VEC f16/q8_0 from compile_commands).

## Design (manager decided, do not change)
Three static variants of the TILE TUs (D 128 and D 256 only; VEC unchanged):
- **A, upstream**: already measured (`bench/results/llama-sycl-fa-static-20261010/`); do not rerun.
- **B, `grf_size_automatic`**: pass `true` as `use_large_grf` in all six TILE `launch_fattn` calls (one template argument each, nothing else). Lets IGC choose 128 or 256 per kernel.
- **C, forced 256 GRF for the TU**: unpatched source, compiled with xe-report's `--large-grf` (adds `-ze-opt-large-register-file`). Upper bound on what 256 GRF can do.
For B and C report, per instance `flash_attn_tile<D, D, ncols1, ncols2, softcap, 32>`: SIMD, GRF, threads/XVE, spill, SLM; and a comparison table A/B/C with columns: instance, spill A, spill B, spill C, GRF B, threads/XVE B. Highlight the instances P-6 uses (ncols2 = 4 and 8 at cols_per_block 8/16/32, i.e. ncols1 x ncols2 in {8,16,32}).
If `bin/llama-sycl-fa-static.sh` cannot take a variant's extra flag (C), add an `XE_REPORT_ARGS` env passthrough to it (that is in scope).

## Files you may touch
- `bin/llama-sycl-fa-static.sh` (only the `XE_REPORT_ARGS` passthrough, if needed)
- `third_party/patches/p10-draft-SYCL-FA-TILE-grf-size-automatic.patch` (new: variant B as a diff against upstream, header: subject, why, "static evidence only; runtime effect unmeasured")
- `bench/results/p10-tile-grf-static-20261010/` (new): B and C reports, `comparison.md` (the A/B/C table + a 5-line reading: where B removes spills, what it costs in threads/XVE)
Nothing else.

## Must pass before you finish
- B and C reports complete for D 128 and 256 (paste comparison.md)
- `git -C third_party/llama.cpp diff --stat` shows only the six template-argument changes for B (then revert for C, and say how you verified the revert)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`

## Done means
A/B/C static comparison for the TILE instances exists with the P-6 instances highlighted, and a draft patch for B. Final report: files changed; commands with result lines; comparison.md; compile wall times; what was NOT verified (runtime cost of spills vs halved threads: an A770 A/B). Do not decide whether to ship.
