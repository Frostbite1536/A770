# P-11: FP16 output accumulation in SYCL FA TILE

Status: two candidates CPU-verified (`third_party/patches/p11c/`); A770 A/B pending. Research: `docs/research/llama-cpp-arc-fa-dispatch.md` §7.

## Problem

SYCL's TILE kernel keeps the P·V accumulator `VKQ` in `sycl::half2` over each block's whole KV range (`SYCL_FAST_FP16` is always defined). Each block's range is `kv / parallel_blocks`, and `launch_fattn` picks `parallel_blocks` from `nsm` and `max_wg_per_cu`. On the CPU SYCL device at A770 launch geometry (nsm 32, max_wg_per_cu 2), the biased-V cases of patch 0005 fail at ≥ 4096 keys per accumulator, with NMSE up to 2.3 (`bench/results/p11b-sycl-fa-fp16-geometry-20261011/`). Masked Llama-3.1-8B decode at 32K runs TILE at 4096 keys per accumulator under that geometry. VEC is unaffected (64–128 keys per sub-accumulator).

The A770 geometry is an assumption: `max_wg_per_cu = max_work_group_size / max_compute_units` has not been read from an A770.

## Candidates (`bench/results/p11c-sycl-fa-tile-fp16-fix-20261011/`)

| | F: FP32 flush per KV tile | S: ≤ 1024 keys per block |
|---|---|---|
| Change | fattn-tile.hpp device code (port of P100 0004) | `launch_fattn` lower bound on `parallel_blocks`, TILE only |
| Biased-V at A770 / worst geometry | 16/16, NMSE ≤ 5e-9 | 16/16, NMSE ≤ 9e-6 |
| Error | removed (half2 holds one 64-key tile) | bounded, not removed |
| Cost | spill −576…+2592 B per instance (D128 1×4: +320 B) | more blocks: more partial-output writes and combine work |
| Speed | not run | not run |

## Accept if (A770, `bench/patches/run.sh`; manager-defined 2026-10-11, before any A770 run)

1. **Geometry**: record the A770's `max_work_group_size` and `max_compute_units` (env step). If `max_wg_per_cu` is not 2, recompute keys per accumulator for the decode shapes before reading 2–4.
2. **Correctness**: `fp32f` and `cap` pass the 16 biased-V cases on the A770 and show no new FLASH_ATTN_EXT failures vs `conv8`. `upstream`/`conv8` biased-V results are recorded as evidence (failures expected at ≥ 4096 keys per accumulator), not counted as regressions.
3. **Kernel**: at the decode shapes (nb 1–4, G 4, kv 4096/16384/32768, F16 and q8_0), the candidate's median over ≥ 3 interleaved rounds is no more than 3% slower than `conv8`.
4. **End to end**: Llama-3.1-8B tg32 at depth 16384 and 32768 is no more than 3% slower than `conv8`, and `-p 1..16` at depth 4096/16384 likewise.

Choice: take F if it meets 2–4. Otherwise take S if it meets 2–4. Otherwise ship neither and report the costs. A candidate that is faster than `conv8` is fine. Thresholds are fixed before measurement; the owner may revise them before the session, not after.
