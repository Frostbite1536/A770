# Ticket p6c-llama-sycl-fa-static: static acm-g10 report for llama.cpp's SYCL FA kernels

Your current directory is a git worktree of A770 (branch `p6c-llama-sycl-fa-static`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. This ticket writes tooling and reports; it changes no kernel.
- The toolchain lives in the MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/neo`, `tools/oneapi-2026.1`), read-only, by absolute path. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. `bin/xe-report.py` finds tools relative to the repo root: in this worktree point it at the main checkout's tools via a new `A770_TOOLS` env var (default: `<repo>/tools`); do not copy or symlink tools into the worktree. Never install anything; no network.
- Do NOT commit, push, or run git commands that write. Read-only git is fine.
- Never invent numbers; report exactly what IGC reports. Nothing here is a runtime measurement.
- Never kill a process you did not start. AOT compiles of llama.cpp TUs are slow: use at most `-j 4` worth of parallel compiles (another worker is building on this machine).
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`bin/xe-report.py` reports per-kernel SIMD, GRF, threads/XVE, spill, private, SLM, DPAS and FP64 for a standalone `.cl`/`.cpp`. P-6 changes llama.cpp's SYCL TILE kernel (`docs/design/P6-sycl-padded-gqa.md`, "Static check"), so we need the same report for llama.cpp's own translation units, before and after the patch. Those TUs need llama.cpp's include paths and defines: `ggml/src/ggml-sycl/template-instances/fattn-tile-instance-dkq{64,128,256}-dv*.cpp` (TILE) and `fattn-vec-instance-f16-f16.cpp`, `fattn-vec-instance-q8_0-q8_0.cpp` (VEC). A CMake configure of llama.cpp with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` gives the exact flags.

## Design (manager decided, do not change)
- `bin/xe-report.py` gains `--compile-commands PATH`: look up `src` in that `compile_commands.json`, take its `-I`, `-isystem`, `-D` and `-std=` flags, drop everything else (in particular any `-fsycl-targets`, `-o`, `-c`), and compile with the existing AOT path (`-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device acm-g10" -shared -fPIC`). Undefined host symbols in the `.so` are fine. Also `A770_TOOLS` (above). Existing behaviour and all `tests/xe` cases unchanged.
- Kernel names in llama.cpp are long templates: keep the full demangled name available with `--wide` (no truncation); the default view may truncate.
- Configure (do not build) llama.cpp once to get `compile_commands.json`: build dir `third_party/llama.cpp/build-sycl-aot-cfg`, flags `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER=$O/bin/icx -DCMAKE_CXX_COMPILER=$O/bin/icpx -DCMAKE_PREFIX_PATH=$O` with `O=/home/p51/Github/A770/tools/oneapi-2026.1`.
- New script `bin/llama-sycl-fa-static.sh [OUTDIR]`: configures as above if needed, runs xe-report on the five TUs listed in Background with `--compile-commands --wide`, writes one text file per TU plus a `summary.txt` (per TU: kernel count, max GRF, any SPILL, SIMD widths seen, min threads/XVE) into OUTDIR (default `bench/results/llama-sycl-fa-static-<date>/`). Exit non-zero if any compile fails; spills are reported, not fatal (this is upstream code).

## Files you may touch
- `bin/xe-report.py` (the two options above, nothing else)
- `bin/llama-sycl-fa-static.sh` (new)
- `tests/CMakeLists.txt` + `tests/test_xe_report.py`: one new case `compile_commands` using a tiny `compile_commands.json` generated at test time that points at `tests/xe/joint_matrix.cpp` with an extra `-I` and `-D` (prove both are applied, e.g. a guard in a header found only via that `-I` that fails to compile without the define); it must pass, and skip (77) without icpx
- `bench/results/llama-sycl-fa-static-20261010/` (new): the upstream baseline report
- `docs/TOOLCHAIN.md`: replace the "*Not done yet* (roadmap P-3b)" sentence with how to run the new script
Nothing else.

## Must pass before you finish
- `bin/llama-sycl-fa-static.sh` completes for all five TUs; paste `summary.txt`
- `ctest --test-dir build --output-on-failure` in this worktree, with `A770_TOOLS=/home/p51/Github/A770/tools` exported so xe-report tests run rather than skip (summary line)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`

## Done means
xe-report can report llama.cpp TUs from compile_commands, the upstream baseline report for TILE (D 64/128/256) and VEC (f16, q8_0) is committed-ready under bench/results, and the script reruns it in one command (it will be rerun on the p6b-patched tree). Final report: files changed; commands with result lines; summary.txt; wall time per TU; anything IGC warned about (SIMD retries, spills). Do not decide whether to ship.
