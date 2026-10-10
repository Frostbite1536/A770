# Ticket p4-xe-planning-models: Xe-HPG tile planner + GQA decode cost model (static arithmetic, no GPU)

Your current directory is a git worktree of A770 (branch `p4-xe-planning-models`, based on `main` @ `c20a5c0`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. This ticket writes Python planning tools, no kernels.
- Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;` (cmake, ninja, ctest, python, shellcheck). Never install anything; no network.
- Do NOT commit, push, or run git commands that write. Leave changes uncommitted.
- Never invent numbers. Every hardware constant comes from the table below, with its source label carried into the code as a comment. Outputs are models and must say so ("model, not measured").
- You may READ `/home/p51/Github/volta-fa/bin/plan-tiles.py` and `/home/p51/Github/volta-fa/bin/smallbatch-model.py` as style references (same CLI style, same honesty about estimates). Do not write outside this worktree.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
volta-fa had `bin/plan-tiles.py` (smem/registers/occupancy/roofline for a tile config) and `bin/smallbatch-model.py` (K/V passes × bandwidth for small batches). A770 needs both, rebuilt for Xe-HPG: the occupancy unit is the hardware thread (one sub-group) on an XVE, not a warp on an SM; the register file is per hardware thread (GRF); XMX is DPAS 8x8x16 at sub-group 8. See `docs/INVARIANTS.md` INV-ARCH-001/002, `docs/research/xe-hpg-static-probes-2026-10-10.md`, `docs/design/P6-sycl-padded-gqa.md`.

## Constants (use exactly these; keep the label in a comment next to each)
| Constant | Value | Label |
| --- | --- | --- |
| Xe-cores | 32 (A770), 28 (A750) via `--xe-cores` | Intel spec |
| XVEs per Xe-core | 16 | Intel Xe-HPG architecture |
| HW threads per XVE | 8 at 128 GRF, 4 at 256 GRF | verified: IGC zeinfo `eu_thread_count` (probe 5) |
| GRF register size | 32 bytes; 128 or 256 registers per HW thread | Intel Xe-HPG architecture (4 KB / 8 KB per thread) |
| Sub-group = one HW thread | SIMD 8, 16 or 32; DPAS needs 8 | INV-ARCH-001 |
| Max work-group size | 1024 work-items | llama.cpp SYCL.md device table for A770 |
| SLM per work-group | 64 KB | INV-ARCH-002 (confirm on the box) |
| SLM per Xe-core (total) | 64 KB default, `--slm-per-xecore` | ASSUMED, unknown: print a warning that SLM-limited occupancy may be pessimistic |
| L2 | 16 MB | Intel spec |
| DRAM bandwidth | 560 GB/s (16 GB), 512 GB/s (8 GB), `--bw` | Intel spec, datasheet; Phase 0 replaces it with measured |
| XMX FP16 dense peak | ~138 TFLOPS at 2.1 GHz (32 Xe-cores x 16 XMX x 64 FMA/clk x 2), `--xmx-tflops` | derived from Intel spec; unreachable ceiling |
| Vector FP16 peak | ~34 TFLOPS at 2.1 GHz (2x FP32: 32 x 16 XVE x 8 lanes x 2 x 2.1 GHz x 2), `--vec-tflops` | derived; TILE/VEC kernels run here |

## Design (manager decided, do not change)
**`bin/plan-tiles.py`** (`--d --dv --br --bc --sg-size --sgs-per-wg --grf {128,256} --kv-bufs --pad --sq --skv --heads --causal --sweep`):
- SLM bytes per work-group: Q tile BR x (D+pad) fp16, K and V tiles `kv-bufs` x BC x (D+pad) fp16, and S/P staging BR x BC (fp32 S + fp16 P) because joint_matrix fragment layout is opaque (row max/sum go through SLM). Flag > 64 KB as INVALID.
- Per-sub-group live-data estimate in GRF registers: O accumulator (BR x DV fp32) and S tile (BR x BC fp32) split across sub-groups, plus a fixed overhead of 24 registers; compare with 128/256 and say "estimate; use bin/xe-report.py for the real count".
- Occupancy per Xe-core: work-groups = min(threads limit = 16 x threads_per_XVE / sgs_per_wg, SLM limit, 1024-work-item limit); report limiter, sub-groups resident, waves over 32 Xe-cores for grid = ceil(sq/BR) x heads.
- Roofline like volta-fa (DRAM best/worst bracket by L2 reuse, XMX floor), using the constants above, labelled model.
- `--sweep` table over BR {8,16,32,64}, BC {32,64,128}, sgs_per_wg {2,4,8,16}, both GRF modes, valid configs only.

**`bin/decode-gqa-model.py`** (`--d --kv-heads --gqa --depth --tokens --kv-type {f16,q8_0,q4_0} --bw --ncols2 {upstream,padded}` plus a default table mode):
- K/V passes per KV head per call: upstream SYCL TILE rule = ceil(G / ncols2) with ncols2 the largest of 8/4/2 dividing G else 1 (`fattn-tile.hpp:1195-1213`); padded rule per `docs/design/P6-sycl-padded-gqa.md` (next pow2 when G in {3,5,6,7} and tokens x next_pow2(G) <= 32).
- Bytes = kv_heads x depth x (Dk+Dv) x bytes/elem x passes (f16 2 B; q8_0 34/32 B; q4_0 18/32 B per element; note that the SYCL TILE path converts quantised K/V to F16 first for tokens >= 3, so add the conversion read+write and the F16 re-read, per `docs/research/llama-cpp-arc-fa-dispatch.md` §3; for tokens <= 2 the quantised cache goes to VEC, which reads K/V once per Q head: show it as a range [1, G] passes).
- L2 bracket: when one KV head's K+V at that depth fits in 16 MB, show the time as a range [1 pass, N passes] (re-reads may hit L2); otherwise N passes.
- Default table: Llama-3.1-8B (D 128, 8 KV heads, G 4), Qwen2.5-7B (D 128, 4 KV, G 7), Qwen2.5-14B (D 128, 8 KV, G 5), depths 4096/16384/32768, tokens 1/4/8, f16; columns: passes upstream, passes padded, µs/layer upstream, µs/layer padded, modelled speed-up. Header line: "MODEL ONLY (datasheet bandwidth); not measured on an A770".

Both scripts: `#!/usr/bin/env python3`, stdlib only, `--help` docstring like volta-fa's, exit 0.

## Files you may touch
- `bin/plan-tiles.py` (new), `bin/decode-gqa-model.py` (new)
- `tests/CMakeLists.txt`: add tests `plan_tiles_smoke` and `decode_gqa_model_smoke` with `PASS_REGULAR_EXPRESSION` pinned to a value you compute by hand in your report (e.g. passes upstream = 7 and padded = 1 for G 7 at 1 token; SLM bytes for one fixed config)
Nothing else.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (summary line; xe_report tests need `/home/p51/Github/A770/tools` and may skip with code 77 in a worktree; that is fine, say so)
- `python bin/plan-tiles.py --d 128 --br 32 --bc 64 --sg-size 8 --sgs-per-wg 4 --grf 128` and `--sweep` (paste output)
- `python bin/decode-gqa-model.py` (paste the default table)
- `bin/check-forbidden.sh`, `shellcheck -x bin/*.sh`

## Done means
Both scripts exist with the specified CLI and outputs, both smoke tests pass, and the report shows the hand calculation behind each pinned value. Final report: files changed; commands with result lines; the pasted outputs; anything you could not model and why. Do not decide whether to ship.
