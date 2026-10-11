# Ticket p11c-sycl-fa-tile-fp16-fix-candidates: two candidate fixes for SYCL TILE's long FP16 output sums

Your current directory is a git worktree of A770 (branch `p11c-sycl-fa-tile-fp16-fix-candidates`, based on `main` @ `@SHA@`). Work only here. A clone of llama.cpp at the pinned commit (79e2e74) is checked out at `third_party/llama.cpp` (gitignored scratch clone, unpatched).

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. Do NOT use SIMD16 DPAS forms, 2D block loads/stores, TF32, FP8, FP64, or Xe2/PVC code (sycl-tla, XeTLA). If you think you need one, STOP and explain.
- Toolchain: MAIN checkout `/home/p51/Github/A770` (`.venv`, `tools/`), read-only, by absolute path; `A770_TOOLS=/home/p51/Github/A770/tools`. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;`. Never install anything; no network.
- Do NOT commit, push, or run git commands that write in this worktree. Inside scratch clones, `git apply` / `git diff` are fine.
- Do not loosen tolerances. Do not edit or delete existing test cases.
- Never invent numbers. CPU-device results are numerics evidence for the same SYCL kernel source, not A770 results. Timing: not run.
- Never kill a process you did not start. Builds are long (about 12 min each at `-j 8`; static reports about 10 min): rerun after a timeout. Another project may be using this machine.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
Read first: `docs/research/llama-cpp-arc-fa-dispatch.md` §6 and §7, `bench/results/p11b-sycl-fa-fp16-geometry-20261011/README.txt` (including "Manager verification"), and `docs/TOOLCHAIN.md` "SYCL on the CPU".

SYCL's TILE kernel (`ggml/src/ggml-sycl/fattn-tile.hpp`) keeps its P·V output accumulator `VKQ` in `sycl::half2` for its whole KV range, because `SYCL_FAST_FP16` is always defined. The running-max rescale is also applied in half2. With A770 launch geometry (CPU device, `GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2`), biased-V cases give these errors against the FP32 reference (tolerance 5e-4):
- 1024 keys per accumulator: NMSE about 8e-6;
- 2048 keys: about 1.5e-4;
- 4096 keys (`MAX_WG_PER_CU=1`): FAIL;
- 8192 and 16384 keys: 0.44 and 2.3.

VEC passes everywhere. TILE already spills on acm-g10 at every D=128/256 instance (§6), so extra registers have a cost.

The P100 sibling project fixed the same issue in CUDA. Its patch is read-only at `/home/p51/Github/P100/third_party/patches/0004-CUDA-FA-tile-flush-FP16-VKQ-to-FP32-per-KV-tile.patch`. It keeps half2 updates within one KV tile and adds them into an FP32 `VKQ_f32` after each tile. The running-max rescale, the cross-warp combine (through `KV_tmp` reinterpreted as float, with a static_assert on size) and the final write all move to FP32.

## Design (manager decided, do not change)
Two independent candidate patches, each applying on top of `third_party/patches/0001-0005`. Neither enters the numbered series in this ticket: the A770 A/B decides.

**F: FP32 flush per KV tile.** Path `third_party/patches/p11c/F-SYCL-FA-tile-flush-FP16-VKQ-to-FP32-per-KV-tile.patch`. Port P100 0004 to `fattn-tile.hpp`, same structure:
- `VKQ_f32` (float2, same count as VKQ) is declared next to `VKQ` in the `SYCL_FAST_FP16` branch;
- the per-iteration rescale is applied to `VKQ_f32` in FP32;
- at the end of each `flash_attn_tile_iter` call, `VKQ` is converted, added into `VKQ_f32` and zeroed;
- the np>1 cross-warp combine uses float storage in the existing shared buffer, with a `static_assert` that it fits;
- the final rescale and write read `VKQ_f32`.
The non-FAST_FP16 (FP32) branch must stay byte-identical. Only fattn-tile.hpp changes.

**S: cap keys per TILE block.** Path `third_party/patches/p11c/S-SYCL-FA-tile-cap-keys-per-block.patch`. No device code change. In the non-stream-k branch of `launch_fattn` (`fattn-common.hpp`), when the caller asks for it, raise the lower bound on `parallel_blocks` to `ceil(ntiles_KQ * nbatch_fa / max_keys)` (still capped by `ntiles_KQ`), before the existing efficiency search. The search then starts from that bound and can only go up. TILE's launch passes `max_keys = GGML_SYCL_FATTN_TILE_MAX_KEYS_PER_BLOCK` (`#define` in fattn-tile.hpp, value 1024; comment: the measured p11b bound). VEC and every other caller pass 0 (off), so their launches are unchanged. Keep the plumbing minimal (one defaulted parameter is fine). Make sure the existing `dst_tmp_meta`/fixup sizing still covers the larger block count; read it and say how.

**Scratch clones** under `third_party/p11c/`: `base` = pin + 0001-0005 + updated test-only patch; `F` = base + F; `S` = base + S.

**Runs** (via `bin/sycl-cpu-tbo.sh` with `LLAMA=<clone>`, `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11c-sycl-cache`, `GGML_SYCL_MKL_FA_DEBUG=1`). Geometries: g-cpu (none), g-a770 (`NSM=32 MAX_WG_PER_CU=2`), g-min (`NSM=32 MAX_WG_PER_CU=1`).
1. `v_bias=2` (16 cases) on base, F and S under g-a770 and g-min. base under g-a770 should reproduce p11b (12/16); it is the control. You may add the p11b scratch-only NMSE-on-pass print to test-backend-ops.cpp in all three clones (archive the diff; not part of any patch). Tables: D, kv, nb, route, parallel_blocks (from a scratch `[FA-PB]` print as in p11b, archived), keys per accumulator, pass/fail, NMSE.
2. Existing coverage on F and S under g-cpu and g-a770: the p6a subset `P` (exact filter in `bench/results/sycl-cpu-fa-baseline-20261010/README.txt`, 85 cases) and the P-6 cases from patch 0002 (filter `nr23=\[[3567],1\]`; report the count). Report result lines. A failure is a finding; do not fix it by changing tests.
3. Static report for F only, on acm-g10: `LLAMA=<F clone> bin/llama-sycl-fa-static.sh bench/results/p11c-sycl-fa-tile-fp16-fix-20261011/static-F`. Use base for the comparison: run the same script on base into `static-base` (reuse `bench/results/p6b-static-acm-g10-20261010/` only if you show that base's fattn-tile sources equal what that report compiled). Give a per-instance table for TILE dkq64/128/256: GRF, SIMD, spill bytes base→F, delta. Note any instance whose SIMD width or GRF mode changes.

## Files you may touch
- `third_party/patches/p11c/` (new): the two candidate patches, each with a header: subject, why (cite p11b numbers), "Tested on: Intel OpenCL CPU device only", credit "F ported from the P100 project, patch 0004" for F
- `bench/results/p11c-sycl-fa-tile-fp16-fix-20261011/` (new): `README.txt` (commands, tables, static comparison, wall times, scratch diagnostic diffs) and gzipped logs; static report subdirectories
- `docs/research/llama-cpp-arc-fa-dispatch.md` §7 only: append a "Candidate fixes (CPU numerics, static)" paragraph (≤ 10 lines, facts and pointers)
Nothing else.

## Must pass before you finish
- F and S each apply to a fresh pin clone on top of 0001-0005 (`git apply --check`); F's diff does not touch the non-FAST_FP16 branch (show `git diff` hunks)
- base g-a770 biased-V reproduces 12/16 (control)
- F and S biased-V results under g-a770 and g-min, with complete tables
- `P` and the 0002 P-6 cases on F and S under g-cpu and g-a770 (result lines)
- F static report and base comparison table
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`, repo `ctest` summary line

## Done means
For each candidate we know: whether it passes the biased-V cases at A770 geometry and the worst case (g-min), with NMSE; whether existing coverage still passes; and for F, the register/spill cost per TILE instance on acm-g10. Final report: files changed; commands with result lines; tables; what was NOT verified (A770 numerics or speed, real-model V statistics). Do not choose between F and S.
