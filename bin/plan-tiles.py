#!/usr/bin/env python3
"""Static tile planner for Intel Arc A770 (Xe-HPG, acm-g10). No GPU needed.

Reports SLM, estimated per-hardware-thread GRF usage, resident work-groups and
attention roofline floors. A sub-group is one hardware thread, not one work-item.
All output is a model, not measured. Real GRF counts come from bin/xe-report.py.
The specified staging model uses D+pad for both K and V; DV affects O and FLOPs.

  python bin/plan-tiles.py --d 128 --br 32 --bc 64 --sg-size 8 --sgs-per-wg 4 --grf 128
  python bin/plan-tiles.py --d 128 --sweep

Bandwidth/peaks are datasheet ceilings; Phase 0 must replace them with measurements.
"""
import argparse
import math


XE_CORES = 32  # Intel spec: A770 32, A750 28 via --xe-cores.
XVES = 16  # Intel Xe-HPG architecture: XVEs per Xe-core.
THREADS = {128: 8, 256: 4}  # verified: IGC zeinfo eu_thread_count (probe 5).
GRF_BYTES = 32  # Intel Xe-HPG architecture: 128/256 GRFs = 4/8 KB per HW thread.
SG_SIZES = (8, 16, 32)  # INV-ARCH-001: sub-group = one HW thread; DPAS needs 8.
MAX_WG_ITEMS = 1024  # llama.cpp SYCL.md device table for A770.
SLM_WG = 64 * 1024  # INV-ARCH-002 (confirm on the box).
SLM_CORE = 64 * 1024  # ASSUMED, unknown: total SLM per Xe-core, override below.
L2_BYTES = 16 * 1024 * 1024  # Intel spec: 16 MB L2 (binary capacity units).
BW_GBS = 560.0  # Intel spec, datasheet: 16 GB 560 GB/s; 8 GB 512 via --bw.
XMX_TFLOPS = 138.0  # derived from Intel spec: 32*16 XMX*64 FMA/clk*2*2.1 GHz; unreachable ceiling.
VEC_TFLOPS = 34.0  # derived: 32*16 XVE*8 lanes*2*2.1 GHz*2; TILE/VEC kernels run here.
OVERHEAD_GRF = 24  # Ticket's fixed live-data estimate, not a hardware constant.
F16_BYTES, F32_BYTES = 2, 4  # Storage sizes of fp16/fp32.


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def nonnegative_int(value):
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return n


def positive_float(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return n


def ceildiv(n, d):
    return (n + d - 1) // d


def tile(a, br, bc, sgs, grf):
    slm = dict(q=br * (a.d + a.pad) * F16_BYTES,
               k=a.kv_bufs * bc * (a.d + a.pad) * F16_BYTES,
               v=a.kv_bufs * bc * (a.d + a.pad) * F16_BYTES,
               s=br * bc * F32_BYTES, p=br * bc * F16_BYTES)
    total = sum(slm.values())
    # Split live O/S bytes across HW threads, then convert to 32-byte GRFs.
    regs = ceildiv(br * (a.dv + bc) * F32_BYTES, sgs * GRF_BYTES) + OVERHEAD_GRF
    items = a.sg_size * sgs
    limits = {"threads": XVES * THREADS[grf] // sgs,
              "SLM": a.slm_per_xecore // total,
              "work-items": MAX_WG_ITEMS // items}
    # The ticket applies the 1024-work-item limit as a resident planning cap.
    # Only the per-WG maximum is sourced; no per-core item limit is verified.
    groups = min(limits.values())
    limiter = "+".join(k for k, v in limits.items() if v == groups)
    reasons = []
    if total > SLM_WG:
        reasons.append("SLM > 64 KB/work-group")
    if regs > grf:
        reasons.append("estimated GRF usage exceeds mode")
    if items > MAX_WG_ITEMS:
        reasons.append("work-group > 1024 work-items")
    if groups == 0:
        reasons.append("no resident work-group in model")
    grid = ceildiv(a.sq, br) * a.heads
    return dict(slm=slm, total=total, regs=regs, groups=groups, limiter=limiter,
                limits=limits, resident=groups * sgs, items=items, grid=grid,
                waves=grid / (groups * a.xe_cores) if groups else math.inf,
                occupancy=groups * sgs / (XVES * THREADS[grf]), reasons=reasons)


def roofline(a, br):
    if a.causal:
        # INV-NUM-004: bottom-right mask. Count visible pairs exactly, even sq > skv.
        first = max(1, a.skv - a.sq + 1)
        count = min(a.sq, a.skv)
        pairs = count * (first + a.skv) // 2
        # One prefix of K/V per Q tile, up to its last row's visible key.
        streamed_keys = sum(min(a.skv, max(0, end + a.skv - a.sq))
                            for end in range(br, a.sq, br)) + a.skv
    else:
        pairs = a.sq * a.skv
        streamed_keys = ceildiv(a.sq, br) * a.skv
    flops = 2 * pairs * (a.d + a.dv) * a.heads
    qo = a.sq * (a.d + a.dv) * F16_BYTES * a.heads
    kv_per_key = (a.d + a.dv) * F16_BYTES
    best = qo + a.skv * kv_per_key * a.heads
    worst = qo + streamed_keys * kv_per_key * a.heads
    return dict(flops=flops, best=best, worst=worst,
                fits=a.skv * kv_per_key <= L2_BYTES,
                xmx=flops / (a.xmx_tflops * 1e12) * 1e3,
                vec=flops / (a.vec_tflops * 1e12) * 1e3,
                dram_best=best / (a.bw * 1e9) * 1e3,
                dram_worst=worst / (a.bw * 1e9) * 1e3)


def report(a):
    t = tile(a, a.br, a.bc, a.sgs_per_wg, a.grf)
    r = roofline(a, a.br)
    print(f"config  BR={a.br} BC={a.bc} D={a.d} DV={a.dv} sg-size={a.sg_size} "
          f"sgs/WG={a.sgs_per_wg} GRF={a.grf} kv-bufs={a.kv_bufs} pad={a.pad}")
    print("SLM     " + "  ".join(f"{k.upper()} {v}" for k, v in t['slm'].items())
          + f"  = {t['total']} B ({t['total'] / 1024:.1f} KB); K/V staging uses D")
    print(f"GRF     {t['regs']}/{a.grf} registers/HW thread "
          "(estimate; use bin/xe-report.py for the real count)")
    print(f"occ     {t['groups']} work-groups/Xe-core, {t['resident']}/"
          f"{XVES * THREADS[a.grf]} sub-groups resident = {t['occupancy']:.0%} "
          f"(limiter: {t['limiter']}; limits {t['limits']})")
    print(f"        {t['items']} work-items/WG; 1024-work-item resident cap is a planning assumption")
    print(f"grid    {t['grid']} work-groups over {a.xe_cores} Xe-cores = {t['waves']:.2f} waves")
    print(f"roof    sq={a.sq} skv={a.skv} heads={a.heads} causal={a.causal}: "
          f"{r['flops'] / 1e9:.3f} GFLOP")
    print(f"        XMX floor {r['xmx']:.3f} ms at {a.xmx_tflops:g} TFLOPS "
          + ("(unreachable ceiling; DPAS 8x8x16)" if a.sg_size == 8
             else "(unavailable at this sg-size; DPAS needs 8)"))
    print(f"        vector floor {r['vec']:.3f} ms at {a.vec_tflops:g} TFLOPS (TILE/VEC ceiling)")
    print(f"        DRAM floor [{r['dram_best']:.3f}, {r['dram_worst']:.3f}] ms "
          f"at {a.bw:g} GB/s; bytes [{r['best']}, {r['worst']}]")
    print(f"        ideal L2 reuse .. no reuse; one head K+V fits 16 MB L2: {r['fits']}; "
          "ideal reuse is not guaranteed")
    compute = r['xmx'] if a.sg_size == 8 else r['vec']
    print(f"        combined floor [{max(compute, r['dram_best']):.3f}, "
          f"{max(compute, r['dram_worst']):.3f}] ms; excludes softmax/launch/sync/spills")
    print("INVALID: " + "; ".join(t['reasons']) if t['reasons'] else "valid (model)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=positive_int, default=128)
    ap.add_argument("--dv", type=positive_int, help="O dimension; default D")
    ap.add_argument("--br", type=positive_int, default=32)
    ap.add_argument("--bc", type=positive_int, default=64)
    ap.add_argument("--sg-size", type=int, choices=SG_SIZES, default=8)
    ap.add_argument("--sgs-per-wg", type=positive_int, default=4)
    ap.add_argument("--grf", type=int, choices=tuple(THREADS), default=128)
    ap.add_argument("--kv-bufs", type=positive_int, default=1)
    ap.add_argument("--pad", type=nonnegative_int, default=0, help="row padding in fp16 elements")
    ap.add_argument("--sq", type=positive_int, default=8192)
    ap.add_argument("--skv", type=positive_int, default=8192)
    ap.add_argument("--heads", type=positive_int, default=32, help="Q heads times batch")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--xe-cores", type=int, choices=(32, 28), default=XE_CORES)
    ap.add_argument("--slm-per-xecore", type=positive_int, default=SLM_CORE, help="assumed total SLM, bytes")
    ap.add_argument("--bw", type=positive_float, default=BW_GBS, help="GB/s; 560 (16 GB), 512 (8 GB)")
    ap.add_argument("--xmx-tflops", type=positive_float, default=XMX_TFLOPS)
    ap.add_argument("--vec-tflops", type=positive_float, default=VEC_TFLOPS)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    a.dv = a.dv if a.dv is not None else a.d
    print("MODEL ONLY; model, not measured on an A770")
    print(f"WARNING: SLM/Xe-core={a.slm_per_xecore} B is ASSUMED, unknown; "
          "SLM-limited occupancy may be pessimistic")
    if not a.sweep:
        report(a)
        return
    print(f"sweep D={a.d} DV={a.dv} sg-size={a.sg_size} kv-bufs={a.kv_bufs} pad={a.pad}; "
          "valid estimates only; use bin/xe-report.py for real GRF counts")
    print(" BR  BC sgs GRF   SLM_B regs WG/core SG/core   occ limiter       waves")
    for br in (8, 16, 32, 64):
        for bc in (32, 64, 128):
            for sgs in (2, 4, 8, 16):
                for grf in THREADS:
                    t = tile(a, br, bc, sgs, grf)
                    if not t['reasons']:
                        print(f"{br:3} {bc:3} {sgs:3} {grf:3} {t['total']:7} {t['regs']:4} "
                              f"{t['groups']:7} {t['resident']:7} {t['occupancy']:5.0%} "
                              f"{t['limiter']:<13} {t['waves']:5.2f}")


if __name__ == "__main__":
    main()
