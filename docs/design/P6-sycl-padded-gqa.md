# P-6 design: padded GQA in the SYCL TILE FlashAttention kernel

Status: **designed, not implemented.** Harness for correctness without an Arc GPU: `bin/sycl-cpu-tbo.sh` (p6a, SYCL on the OpenCL CPU device, done). Implementation: ticket `p6b-sycl-tile-padded-gqa`. Owner decision points are marked **Decision**.

Port of volta-fa P-6 (`~/Github/volta-fa/docs/design/P6-small-batch-gqa.md`, patches 0001/0002). The mechanism is the same, but where the fix lands differs. On Volta the MMA kernel already masked phantom heads and only dispatch changed. On Arc the SYCL **TILE** kernel has no phantom-head support (neither does CUDA's TILE kernel), so the kernel changes too.

## Problem (code reading; unmeasured)

`docs/research/llama-cpp-arc-fa-dispatch.md` §1–2: below 32 query tokens the SYCL backend runs FA on TILE or VEC (no XMX). That includes **F16 single-token decode on any GQA model** (`fattn.cpp:262-265`). TILE packs `ncols2` Q heads of one KV head per work-group, with ncols2 = the largest of 8/4/2 that **divides** G (`fattn-tile.hpp:1195-1213`), so each KV head's K/V is streamed ceil(G/ncols2) times:

| G (example model) | ncols2 today | K/V passes | padded ncols2 | passes |
| --- | --- | --- | --- | --- |
| 3 | 1 | 3 | 4 | 1 |
| 5 (Qwen2.5-14B) | 1 | 5 | 8 | 1 |
| 6 | 2 | 3 | 8 | 1 |
| 7 (Qwen2.5-7B) | 1 | 7 | 8 | 1 |

Decode is bandwidth-bound, so this is potentially up to G× on the attention part of decode at depth. L2 (16 MB) will absorb some re-reads at short context, so **the real gain is a Phase 0 measurement, not this table**.

## Why the kernel must change

`flash_attn_tile` (`fattn-tile.hpp:725-732`) derives the work-group's first Q head as `head0 = group(0)*ncols2 - sequence*ne02` and its KV head as `head0 / gqa_ratio`. That is only valid when ncols2 divides G. `launch_fattn` already sizes the grid as `ceil(G/ncols2) * n_kv_heads * ne03` (`fattn-common.hpp:1018-1021, 1109`, ported from CUDA MMA). If ncols2 does not divide G, the two disagree, and the output write (`:1016-1018`) of a phantom head lands on **the next KV head's first real Q head**.

## Design

### Change A: KV-head-major work-group indexing (kernel)

Replace `fattn-tile.hpp:725-732` with the mapping CUDA's MMA kernel uses (`fattn-mma-f16.cuh:1911-1917`):

```
const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
const int z        = group(0);
const int sequence = z / (ntiles_z_gqa * ne12);
const int kv_head  = (z / ntiles_z_gqa) % ne12;
const int zt_gqa   = z % ntiles_z_gqa;                  // which ncols2-slice of this KV head's Q heads
const int head0    = kv_head * gqa_ratio + zt_gqa * ncols2;
K/V base: nb12 * kv_head, nb22 * kv_head
```

**Identity for existing configs:** if ncols2 divides G, then ntiles_z_gqa = G/ncols2 and `head0 = z*ncols2 - sequence*ne02` exactly as today, so every existing launch maps the same work-group to the same heads. The ticket proves it with the existing suite unchanged.

A head `c` of the work-group is **phantom** iff `zt_gqa*ncols2 + c >= gqa_ratio`. For phantom heads:

1. **Q load** (`:812-858`): write zeros to `Q_tmp` and do not read global Q (the read would be another KV head's Q, or out of bounds for the last KV head). Zero Q gives logits equal to the mask: finite, no NaN.
2. **Sinks** (`:977`): do not read `sinks[head0 + c]` (out of bounds or wrong head). Skip the sink adjustment for that column.
3. **Output and `dst_meta`** (`:1004-1056`): skip the write. Use `continue`, not `return`: unlike the existing `ncols1` tail check, a later `jc` in the same loop can be a real head.
4. ALiBi is untouched: the slope is only used with ncols2 = 1, and padding requires `max_bias == 0`.
5. The `np > 1` in-work-group combine and the `parallel_blocks > 1` combine kernel need no change. Phantom columns are combined in SLM like any column but never written, and the combine kernel iterates over real heads (`fattn-common.hpp:1175`).

### Change B: padded ncols2 in TILE dispatch (`fattn-tile.hpp:1195-1213`, DV ≤ 256 only)

Before the exact-divisor ladder:

```
if (use_gqa_opt && !is_pow2(G) && G < 8 && Q->ne[1] * next_pow2(G) <= TILE_PAD_MAX_COLS) -> ncols2 = next_pow2(G)  // 3 -> 4; 5, 6, 7 -> 8
```

`TILE_PAD_MAX_COLS = 32` (the largest TILE column block below 32 tokens, `:1081-1091`): pad only while one work-group still holds every Q head of the KV head for all tokens. Beyond that, padding adds wasted columns to a kernel that may be compute-bound. **Decision (Phase 0):** the threshold is a named constant to tune from A770 measurements, not a claim.

No new template instances: ncols2 = 4 and 8 are already instantiated for DV ≤ 256, so binary size and JIT/AOT time are unchanged. The DV = 512 branch (MLA shapes) is untouched.

### Change C (optional, not in p6b): one GQA predicate

Selection (`fattn.cpp:130, 181-190`) and TILE dispatch (`fattn-tile.hpp:1169`) compute the GQA condition differently (stride check vs batch limit). Unlike Volta, I found no crash path. TILE loads Q with vector copies of up to 16 bytes (`ggml_sycl_memcpy_1`, `:826`) from a base that already includes `head0 * nb02`. So a stride that isn't 16-byte aligned meets the same loads at ncols2 = 1 as at ncols2 > 1, and the mismatched predicate adds no new alignment hazard (whether misaligned vector loads are safe on Xe-HPG at all is unverified and out of scope). **Decision: deferred.** It changes which kernel runs for misaligned strides, which is a performance question for Phase 0.

## Tests

Add volta-fa P-6's test block to `tests/test-backend-ops.cpp` unchanged: hs 128/256 × G 3/5/6/7 × nb 1/2/3/4/8/16 × sinks off/on, 2 KV heads, kv 4096, mask on. Two KV heads catch phantom writes that alias the next KV head; sinks catch the out-of-bounds sink read. Add q8_0 K/V at nb 3 and 8, since quantised caches reach TILE on Arc for B ≥ 3. Run them through `bin/sycl-cpu-tbo.sh` (CPU SYCL device: correctness only).

Required negative controls (each must FAIL, then pass after revert):
- dispatch padding on, phantom output write NOT skipped → aliasing caught by the 2-KV-head cases;
- phantom sink read not guarded → caught by sinks cases, or by an out-of-bounds read showing as wrong output.

## Static check

AOT-compile the patched `fattn-tile.hpp` instances for acm-g10 (`-DGGML_SYCL_DEVICE_ARCH=acm-g10` + IGC dumps) and compare SIMD, GRF and spill against the unpatched build for the ncols2 = 4/8 instances. Expected: unchanged or one or two more GRFs for the extra index math. Any new spill is a failing result (INV-PERF-003).

## Accept if (A770, `bench/patches/run.sh`, variant `p6` vs `upstream`)

1. **Correctness**: `test-backend-ops test -o FLASH_ATTN_EXT` on SYCL0 has no new failures; the new GQA 3/5/6/7 block passes.
2. **Kernel**: on the new block's G 5/6/7 cases at 1–4 tokens (where padding applies: tokens × 8 ≤ 32), and G 3 at 1–8, the median over ≥ 3 interleaved rounds is ≥ 1.3× faster. Cases beyond the threshold must be unchanged (±3%). No existing case (G 1/2/4/8) is more than 3% slower.
3. **End to end**: `llama-bench -fa on -p 1,4 -d 16384` on Qwen2.5-7B (G 7) improves ≥ 10%, and `-p 8` (not padded) is unchanged (±3%). Llama-3.1-8B (G 4, never padded) changes by less than ±3%.
4. **Static**: no new spills in the TILE instances (`bin/llama-sycl-fa-static.sh`, patched vs upstream).

Reject, or retune `GGML_SYCL_FATTN_TILE_PAD_MAX_COLS`, if 2 or 3 fails. If padding loses at 8 tokens but wins at 1–4, lower the threshold rather than dropping the patch. These thresholds are manager decisions made before the measurement, so the result cannot move them.

## Not verified until an A770 run

Timing of everything above; whether L2 already hides the re-reads; the threshold; the effect on end-to-end decode tok/s (`llama-bench -p 1,2,4,8 -d 4096,16384` on a GQA-7 model, SYCL vs Vulkan).
