# P8: non-contiguous SYCL K/V conversion

Target: Intel Arc A770 (Xe-HPG, acm-g10), non-XMX kernels.
llama.cpp pin: 79e2e74. Manager design ports volta-fa changes 2 and 3.

Short quantised rows previously used one 256-work-item group per row,
with two values per item. A 128-value row left 192 items idle.
Patch 0003 rounds the required items per row to WARP_SIZE (16 on Intel)
and puts 256/items_per_row rows in a group. Unary conversion does the
same for rows below 256 values; dequantisation packs rows below 512.
The last group checks its row bound. Arithmetic stays unchanged.
For widths that do not divide 256, a group contains fewer than 256 items.
Long-row launch parameters stay the same. The contiguous unary wrapper
explicitly disables packing; contiguous quantised dispatch is unchanged.
The existing unary downsample_sycl_global_range guard stays in place.

Patch 0004 gives non-contiguous q4_0/q8_0 eight values per work-item.
Loads assume only two-byte alignment, as required by 18/34-byte ggml
blocks. Builtin memcpy emits the aligned 8/16-byte output copies.
The vector kernel keeps dfloat2 operations and ggml_sycl_cast, including
half intermediate rounding with GGML_SYCL_F16 enabled. It supports half
and float destinations. Only the non-contiguous q4_0/q8_0 dispatch changes.
Partial quant blocks or a rounded plane launch above INT_MAX use the
generic kernel. The bound is checked before multiplying row counts.
Every changed/new kernel launch declares reqd_sub_group_size(WARP_SIZE).
convert.hpp needs no change.

The standalone JIT benchmark copies upstream q4_0/q8_0 dequantisers,
non-contiguous generic and unary bodies byte for byte. Its row variants
change only row indexing and the bounds check. The vector body is shared
by exact textual copy with the patch. The main matrix uses FA cache views:
ne=[D,n_kv,n_head_kv], token stride=n_head_kv*D, head stride=D.
D=64/128/256, n_kv=4096/16384, heads=4/8, q4_0/q8_0, half/float output.
It compares every byte against upstream, including the D2D reference,
and exits nonzero on a mismatch. Timings are median event durations after
warm-up; --reps controls the sample count. --edges checks ragged groups,
non-power-of-two widths, long rows, and padded four-dimensional unary views.

Verified on the Intel OpenCL CPU device, mechanism check:
- Main matrix: 48/48 combinations, 144/144 comparisons in each arithmetic mode.
- Edges: 20/20 quantised combinations and 7/7 unary shapes in each arithmetic mode.
- Backend before/after: p6a FA 85/85, supplemental FA 24/24,
  CPY 52/52, GET_ROWS 33/33; identical case lists, no new failures.
- Padded K/V views exercise nc conversion in the p6a corpus. Supplemental
  FA cases cover 3/4/8/16 queries with contiguously allocated token-major views.
- Negative source-head index: benchmark exits 1 with a byte mismatch;
  padded q8_0 FA exits 1, 0/1 passed. Byte-for-byte restoration gives
  benchmark 48/48 plus all edges, and the same FA case 1/1, both exit 0.
- acm-g10 static compile: 14 kernels in each arithmetic mode; SIMD16,
  128-GRF mode, 8 threads/XVE; zero spill/private/SLM/DPAS/FP64.
- Repository ctest: 28/28 passed; forbidden-feature scan and shell lint passed.

See bench/results/p8-cpu-bitexact-20261010/README.txt for exact commands,
filters, output tables, and final result lines.
The p6a CPU-only harness patch is excluded from both conversion patches.
No existing test or tolerance changes; no commit, push, install, or network.

Not verified: A770 execution or timing, hardware bandwidth/occupancy,
end-to-end model decode, and the INT_MAX fallback at runtime.
A770 work: run the same JIT benchmark with the GPU selector, record its
median conversion and D2D times, run backend correctness, then measure
end-to-end decode with quantised caches. This ticket makes no ship decision.

## Accept if (A770, `bench/patches/run.sh`; manager-defined 2026-10-10, before any A770 run)

1. **Correctness**: `conv` and `conv8` show no new FLASH_ATTN_EXT, CPY or GET_ROWS failures vs `upstream`. The f16→q4_0 CPY flake is rerun on both before it is attributed. The microbenchmark is bit-exact on the A770 (`micro` step).
2. **Microbenchmark** (A770, median of `--reps`): at D=128, vec8 is ≥ 1.5× upstream on every shape, and neither variant is more than 3% slower than upstream on any shape. On the V100 the same port gave 3.1–3.5× at D=128 and 1.7–1.9× at D=256.
3. **Kernel**: with q8_0 and q4_0 K/V at 4–16 tokens and kv 16384, the `conv8` median over ≥ 3 interleaved rounds is ≥ 1.2× `p6` at D=128. No F16 case moves by more than ±3%. 1–2-token cases run VEC (no conversion; research §3) and must be unchanged (±3%).
4. **End to end**: Llama-3.1-8B with `-ctk q8_0 -ctv q8_0 -p 4,8 -d 16384` is ≥ 5% faster with `conv8` than with `p6`, and tg32 is unchanged (±3%).

Reject 0004 if 2 fails for vec8. Reject 0003 if its rows variant is slower than upstream. Thresholds are fixed before measurement; the owner may revise them before the session, not after.
