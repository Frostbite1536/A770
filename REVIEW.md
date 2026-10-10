# Review rules

Severity: **Critical** (blocks merge), **Nit** (fix if easy), **Pre-existing** (file separately). Review changed files in full, not just the diff.

## Checklist

- [ ] No device feature outside Xe-HPG acm-g10 (INV-ARCH-001): no SIMD16 DPAS forms or shapes, 2D block messages, TF32, FP8, FP64; nothing copied from Xe2/PVC kernels. (SIMD8 DPAS, `joint_matrix` 8×8×16 f16/bf16 are fine.)
- [ ] `bin/xe-report.py` output attached: spill = 0, SLM ≤ 64 KB incl. dynamic, **DPAS > 0 on every XMX kernel, at SIMD 8**, GRF mode and threads/XVE stated (INV-ARCH-002, INV-PERF-003).
- [ ] Every kernel declares its sub-group size (INV-ARCH-004).
- [ ] Parity tests cover tail tiles (N not a multiple of tile), fully masked rows, large logits, GQA ratios incl. 3/5/6/7, `seq_q < seq_kv` causal (INV-NUM-002/003/004).
- [ ] QK accumulates in FP32 (no `half8` DPAS accumulator for S) (INV-NUM-005).
- [ ] Tolerances unchanged, or the change is a recorded manager decision.
- [ ] No N×N intermediate in global memory (INV-NUM-001).
- [ ] Out-of-bounds: every SLM/global access guarded for partial tiles; barriers not inside divergent control flow; sub-group operations called by the whole sub-group.
- [ ] Every fallible SYCL/Level Zero call is checked; an async exception handler is installed.
- [ ] Multi-GPU: capability check + fallback, event-based sync, buffer lifetime (INV-MGPU-*).
- [ ] Any performance claim links to an artifact with A770 variant, driver/IGC, oneAPI, kernel driver, ReBAR, commit, baseline backend + commit (INV-PERF-001/002).
- [ ] Edits to existing tests are justified.
- [ ] Docs updated when behaviour changes.
