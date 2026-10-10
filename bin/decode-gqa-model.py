#!/usr/bin/env python3
"""Static SYCL GQA decode bandwidth model for Intel Arc A770. No GPU needed.

Compare upstream TILE's exact-divisor ncols2 with the proposed padded dispatch.
Times are DRAM floors per layer, model, not measured. Dk = Dv = D. The one-head
L2 fit test is optimistic: other heads and traffic compete for the same cache.
Quantised caches use VEC for <=2 tokens; >=3 tokens add conversion to F16 and
the TILE F16 re-reads. Assumes the GQA optimisation predicates hold (mask,
no ALiBi, supported dimensions/strides). Models small batches below 32 tokens.

  python bin/decode-gqa-model.py             # default f16 comparison table
  python bin/decode-gqa-model.py --d 128 --kv-heads 4 --gqa 7 --depth 32768 --tokens 1
  python bin/decode-gqa-model.py --kv-type q8_0 --tokens 4 --ncols2 padded

Excludes Q/O, launch, softmax, combine, dequantisation compute and wasted padding
compute. Replace datasheet bandwidth with Phase 0 measurements via --bw.
"""
import argparse
import math
import sys


L2_BYTES = 16 * 1024 * 1024  # Intel spec: 16 MB L2 (binary capacity units).
BW_GBS = 560.0  # Intel spec, datasheet: 16 GB 560 GB/s; 8 GB 512 via --bw.
ELEMENT_BYTES = {"f16": 2, "q8_0": 34 / 32, "q4_0": 18 / 32}  # Ticket: block storage incl. scales.
PAD_MAX_COLS = 32  # docs/design/P6-sycl-padded-gqa.md, Change B; unmeasured threshold.


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def positive_float(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return n


def ncols2(gqa, tokens, padded):
    # Upstream fattn-tile.hpp:1195-1213: largest 8/4/2 dividing G, otherwise 1.
    upstream = next((c for c in (8, 4, 2) if gqa % c == 0), 1)
    pow2 = 1 << (gqa - 1).bit_length()
    # docs/design/P6-sycl-padded-gqa.md, Change B (DV <= 256 checked by caller).
    if padded and gqa in (3, 5, 6, 7) and tokens * pow2 <= PAD_MAX_COLS:
        return pow2
    return upstream


def cost(d, kv_heads, gqa, depth, tokens, kv_type, bw, padded):
    packed = ncols2(gqa, tokens, padded and d <= 256)
    passes = (gqa + packed - 1) // packed
    elements = kv_heads * depth * (d + d)
    cache = elements * ELEMENT_BYTES[kv_type]
    f16 = elements * ELEMENT_BYTES['f16']
    # docs/research/llama-cpp-arc-fa-dispatch.md §3: quantised <=2 uses VEC;
    # >=3 converts once (compressed read + F16 write), then TILE reads F16.
    vec = kv_type != 'f16' and tokens <= 2
    conversion = cache + f16 if kv_type != 'f16' and not vec else 0
    streamed = cache if vec or kv_type == 'f16' else f16
    upper_passes = gqa if vec else passes
    fits = streamed / kv_heads <= L2_BYTES
    # VEC explicitly reports [1,G] potential DRAM passes, even without L2 fit.
    lower_passes = 1 if fits or vec else upper_passes
    low = conversion + streamed * lower_passes
    high = conversion + streamed * upper_passes
    return dict(packed=packed, passes=(1, gqa) if vec else (passes, passes),
                cache=cache, f16=f16, conversion=conversion, vec=vec, fits=fits,
                bytes=(low, high), us=(low / (bw * 1e9) * 1e6, high / (bw * 1e9) * 1e6))


def interval(values, digits=2):
    lo, hi = values
    return f"{lo:.{digits}f}" if lo == hi else f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def pass_text(c):
    lo, hi = c['passes']
    return str(lo) if lo == hi else f"[{lo}, {hi}]"


def speedup(up, pad):
    # Compare matching ideal-reuse and no-reuse endpoints, not independent extremes.
    ratios = [u / p for u, p in zip(up['us'], pad['us'])]
    return interval((min(ratios), max(ratios))) + "x"


def custom(a):
    costs = {mode: cost(a.d, a.kv_heads, a.gqa, a.depth, a.tokens, a.kv_type, a.bw,
                        mode == 'padded') for mode in ('upstream', 'padded')}
    print(f"custom Dk=Dv={a.d}, KV heads={a.kv_heads}, G={a.gqa}, depth={a.depth}, "
          f"tokens={a.tokens}, cache={a.kv_type}, bandwidth={a.bw:g} GB/s")
    print(f"passes upstream = {pass_text(costs['upstream'])}; "
          f"passes padded = {pass_text(costs['padded'])}")
    for mode, c in costs.items():
        print(f"{mode:8} ncols2={c['packed'] if not c['vec'] else 'n/a (VEC)'}; "
              f"bytes/layer={interval(c['bytes'], 0)}; µs/layer={interval(c['us'])}; "
              f"one KV head {'cache' if c['vec'] else 'F16 K+V'} fits L2={c['fits']}")
    c = costs[a.ncols2]
    print(f"selected {a.ncols2}: {interval(c['us'])} µs/layer; "
          f"modelled speed-up upstream/padded={speedup(costs['upstream'], costs['padded'])}")
    if c['vec']:
        print("VEC: quantised cache, no conversion; [1, G] DRAM passes brackets reuse; "
              "logical reads are once per Q head; padding does not change this path")
    elif c['conversion']:
        print(f"conversion: compressed read {c['cache']:.0f} B + F16 write {c['f16']:.0f} B "
              "+ F16 re-read per TILE pass; conversion traffic always included in DRAM model")


def table(a):
    print(f"f16; bandwidth={a.bw:g} GB/s; Dk=Dv=D; µs per attention layer")
    print("model          depth tok | passes upstream padded | µs/layer upstream | µs/layer padded | modelled speed-up")
    for name, d, kv_heads, gqa in (("Llama-3.1-8B", 128, 8, 4),
                                  ("Qwen2.5-7B", 128, 4, 7),
                                  ("Qwen2.5-14B", 128, 8, 5)):
        for depth in (4096, 16384, 32768):
            for tokens in (1, 4, 8):
                up = cost(d, kv_heads, gqa, depth, tokens, 'f16', a.bw, False)
                pad = cost(d, kv_heads, gqa, depth, tokens, 'f16', a.bw, True)
                print(f"{name:13} {depth:5} {tokens:3} | {pass_text(up):>15} {pass_text(pad):>6} | "
                      f"{interval(up['us']):>17} | {interval(pad['us']):>15} | {speedup(up, pad)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=positive_int, default=128)
    ap.add_argument("--kv-heads", type=positive_int, default=4)
    ap.add_argument("--gqa", type=positive_int, default=7)
    ap.add_argument("--depth", type=positive_int, default=32768)
    ap.add_argument("--tokens", type=positive_int, default=1)
    ap.add_argument("--kv-type", choices=tuple(ELEMENT_BYTES), default='f16')
    ap.add_argument("--bw", type=positive_float, default=BW_GBS, help="GB/s; 560 (16 GB), 512 (8 GB)")
    ap.add_argument("--ncols2", choices=('upstream', 'padded'), default='upstream')
    a = ap.parse_args()
    custom_options = ('--d', '--kv-heads', '--gqa', '--depth', '--tokens', '--kv-type', '--ncols2')
    is_custom = any(arg.split('=')[0] in custom_options for arg in sys.argv[1:])
    if is_custom and a.tokens >= 32:
        ap.error("tokens must be <32: oneDNN/MKL dispatch is outside this decode model")
    if is_custom and a.kv_type != 'f16' and a.d % 32:
        ap.error("quantised dimensions must be multiples of the 32-element storage block")
    print("MODEL ONLY (datasheet bandwidth); not measured on an A770" if a.bw == BW_GBS
          else "MODEL ONLY (user-supplied bandwidth); not measured on an A770")
    if is_custom:
        custom(a)
    else:
        table(a)
    print("model, not measured; time ranges = ideal L2 reuse .. no reuse when one head fits 16 MB")
    print("speed-up ranges compare matching reuse endpoints; shared L2 capacity/reuse is not guaranteed")
    print("assumes GQA predicates hold; excludes Q/O, launch/softmax/combine and conversion/padding compute")


if __name__ == "__main__":
    main()
