# FlashAttention dispatch on the A770 in llama.cpp (code study, 2026-10-10)

Code reading only; nothing built or run on an Arc GPU. Anchors are `ggml/src/ggml-<backend>/<file>:<line>` at llama.cpp `79e2e74` (`bench/LLAMA_COMMIT`, the same pin as volta-fa). Notation follows volta-fa's `docs/research/small-batch-volta.md`: B = query tokens per call (`Q->ne[1]`), G = GQA ratio, ncols2 = Q heads packed per work-group, "passes" = how many times each KV head's K/V is streamed from memory per call (ignoring L2).

The A770 has two llama.cpp backends that can run FA: **SYCL** (oneAPI, Level Zero) and **Vulkan**. They take different paths, so the baseline for each shape is whichever is faster (INV-PERF-002).

## 1. SYCL backend: kernel selection

`ggml-sycl/fattn.cpp:112-279`, in order:

1. FA off unless `GGML_SYCL_ENABLE_FLASH_ATTN` (default 1, `ggml-sycl.cpp:451`).
2. `gqa_opt_applies = G >= 2 && mask && max_bias == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0` (`:130`); a stride check later clears it for non-16-byte-aligned strides (`:181-190`).
3. **B ≥ 32 → oneDNN SDPA** if `ggml_sycl_flash_attn_ext_onednn_supported` (`:135-138`, "ONEDNN requires min 32 query tokens"). This is the XMX path for prefill.
4. **MKL path**: B ≥ 32, n_kv ≥ 1024, G ≥ 2, mask, no sinks/ALiBi/softcap, D in 64..512 step 64 (`:158-179`). Converts non-F16 K/V to F16 first, then XMX GEMMs.
5. Head-dim / type filters (`:193-245`).
6. **Small batches** (`:260-278`): F16 K/V with B = 1 → VEC only if `!gqa_opt_applies`, otherwise **TILE**. Quantised K/V with B ≤ 2 → VEC (TILE on Battlemage only, `:269-272`). Everything else → TILE.

So on the A770, for B < 32, every FA call goes to TILE or VEC. Neither uses XMX. That includes **ordinary F16 decode on any GQA model**.

## 2. SYCL TILE: heads per work-group

`ggml-sycl/fattn-tile.hpp:1153-1214` (`launch_fattn_tile_switch_ncols2`) is a port of the CUDA TILE dispatch:

- `use_gqa_opt = mask && max_bias == 0 && B <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0`, with `gqa_limit = 16` if G ≤ 4 and D ≤ 256, else unlimited (`:1168-1169`).
- ncols2 = the largest of 8, 4, 2 that **divides** G, else 1 (`:1195-1213`; 16 is tried first for DV = 512).
- The kernel reads K/V for head `head0 / gqa_ratio` (`:728-732`) and the work-group covers ncols2 consecutive Q heads.

| G | ncols2 | passes over K/V per call |
| --- | --- | --- |
| 4, 8 | 4, 8 | 1 |
| 6 | 2 | **3** |
| 5, 7 | 1 | **5, 7** |
| 3 | 1 | **3** |

This is the same mechanism as volta-fa P-6 (`third_party/patches/0002` there). On Volta it only hit small-batch prefill/verify, because decode went elsewhere. On the A770 it also hits **single-token F16 decode** (§1 step 6), which is bandwidth-bound: GQA 7 (e.g. Qwen2.5-7B: 28 Q / 4 KV heads) streams each KV head 7 times. *(inferred, unmeasured: L2 absorbs some of the re-reads; at long context the K/V of one head exceeds the 16 MB L2.)*

**Two predicates.** Selection uses `gqa_opt_applies` (`fattn.cpp:130`, with a stride check, without the batch limit), while TILE dispatch recomputes `use_gqa_opt` (`fattn-tile.hpp:1169`, with a batch limit, without the stride check). I found no abort path like Volta's NO_DEVICE_CODE crash (volta-fa patch 0001), but the two can disagree about the same call.

## 3. SYCL: quantised K/V conversion

`launch_fattn` (`ggml-sycl/fattn-common.hpp:954-1012`) converts non-F16 K/V to F16 before TILE runs. Non-contiguous views use `dequantize_block_nc_sycl` / `convert_unary_nc_sycl` (`ggml-sycl/convert.cpp:590-650`): one work-group of `SYCL_DEQUANTIZE_BLOCK_SIZE` = 256 work-items (`presets.hpp:41`) per (row, head), 2 values per work-item for dequantize. For a 128-value KV row, 192 of 256 work-items exit immediately. This is the launch geometry volta-fa patch 0005 fixed in CUDA (there it was block-count-bound and cost more than the attention at depth). Decode with B ≤ 2 avoids it (VEC), so it applies to B = 3..31 with a quantised cache.

## 4. Vulkan backend on the A770

- Intel Xe1 is detected by a minimum sub-group size of 8 plus packed int8 dot products (`ggml-vulkan.cpp:85-97`).
- **GQA packing for small batches**: if N ≤ 8 and 1 < G ≤ `max_gqa` (block rows, capped at 32), N is set to G and the work-groups along y are divided by G (`:8126-8134`). Any G is accepted, not only powers of two, so Vulkan does not have the §2 re-read problem for N ≤ 8.
- **The Intel decode fast path `xe_fa_decode` is off for the A770**: it requires Xe2, or Xe1 with coopmat **and `uma`** (integrated) (`:3184`, `:8211-8213`).
- The f16 dequant scratch for quantised K/V is disabled on Xe1 ("Intel Xe1 regresses, see PR 25494", `:8115-8117`).
- Which FA variant the A770 gets (scalar vs coopmat1) depends on whether the driver exposes `VK_KHR_cooperative_matrix`. Record `vulkaninfo` on the box.

## 5. Ranked candidates (unmeasured; Phase 0 decides)

1. **Padded GQA in the SYCL TILE dispatch** (port of volta-fa P-6 change 2): pad G = 3, 5, 6, 7 to the next power of two with masked phantom heads, so K/V streams once. It covers F16 decode and B < 32. It needs a check that the SYCL TILE kernel masks heads ≥ G the way CUDA's MMA kernel does (CUDA TILE did not need it upstream; confirm before porting).
2. **One GQA predicate for SYCL selection and dispatch** (port of volta-fa P-6 change 1).
3. **Conversion launch geometry** for short non-contiguous rows (port of volta-fa 0005/0006 to `convert.cpp`).
4. **An XMX small-batch kernel** (B 1–31). Today nothing below 32 tokens uses XMX on the A770. Only worth it if Phase 0 shows TILE compute-bound rather than bandwidth-bound after (1).
5. Multi-GPU: `GGML_SYCL_DEV2DEV_MEMCPY` modes and peer access on consumer Arc (README notes crashes on the L0 path).
6. **TILE register pressure** (§6): opt TILE into `grf_size_automatic` (256 GRF: half the threads, maybe no spills), or a SIMD16 TILE variant. Static experiment first (P-10).

## 6. Static: the SYCL TILE kernels spill on acm-g10 (2026-10-10)

`bin/llama-sycl-fa-static.sh` (IGC 2.41.5, icpx 2026.1, upstream 79e2e74; report in `bench/results/llama-sycl-fa-static-20261010/`): **every** TILE instance at D = 128 and D = 256 compiles to SIMD32 with 128 GRF (8 threads/XVE) and spills **7.4–14.8 KB** per hardware thread. That includes the ncols2 = 4/8 instances P-6 relies on. The cause: the kernel requires sub-group 32 (`fattn-tile.hpp:1077`, "can't support WARP_16_SIZE"), and at SIMD32, 128 GRF leaves 128 bytes (32 floats) per work-item, too few for the VKQ accumulators. VEC f16 has 14 spilling kernels (max 4.4 KB); VEC already opts into `grf_size_automatic` for D ≥ 512 only (`fattn-vec.hpp:594`), and TILE never does (`launch_fattn<..., use_large_grf = false>`, `fattn-common.hpp:907`).

**P-10 (static, `bench/results/p10-tile-grf-static-20261010/comparison.md`)**: 256 GRF (`grf_size_automatic` or forced) reduces most spills (e.g. D 128 `1x8` 7.6 → 1.2 KB) but removes none, makes several D = 256 configs worse (`2x8` 10.2 → 17.8 KB), and halves threads/XVE to 4. Large GRF is not a fix; the SIMD32 requirement is the structural cause, so a SIMD16 TILE variant (twice the registers per work-item) is the design option, after Phase 0 shows the spills cost time.

Static only: how much these spills cost at runtime (spill traffic goes through L1/L3) is a Phase 0 question. It is a new candidate below (§5, item 6).

Before porting anything, re-check upstream for the same fix (the SYCL backend moves fast: oneDNN/MKL FA paths and Xe2 tuning landed in 2026).
