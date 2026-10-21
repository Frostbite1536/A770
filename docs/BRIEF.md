# A770 — attention and multi-GPU work for the Intel Arc A770

*Share brief for Intel-Arc-owning communities; sibling of the
[volta-fa](https://github.com/Frostbite1536/volta-fa/blob/main/docs/BRIEF.md) and
[P100](https://github.com/Frostbite1536/P100/blob/main/docs/BRIEF.md) briefs. Honest
status: **nothing has run on an A770 yet** — the day-one measurement kit is ready.
What exists is a source-traced code study, a static compiler-probe suite with real
findings, five ready SYCL patches correctness-checked on the OpenCL CPU device, and
the tooling to keep kernel claims honest.*

**Repo:** https://github.com/Frostbite1536/A770 (AGPL-3.0)

## What it is

The same method as our V100 and P100 projects, applied to the Intel Arc A770
(Xe-HPG, `acm-g10`) in SYCL and OpenCL C: find where llama.cpp loses time or accuracy
on this specific silicon, fix it as a small patch series against a pinned commit
(`79e2e74`), and make no performance claim that wasn't measured. The A770 is a cheap
16 GB card with XMX matrix engines, but most Intel GPU kernel work now targets
Battlemage (Xe2) and Data Center GPU Max — Alchemist gets the leftovers.

## The gap that motivated it

From `docs/research/llama-cpp-arc-fa-dispatch.md` (code-read at the pin; llama.cpp
runs FA on the A770 through two backends, SYCL and Vulkan — the baseline per shape is
whichever is faster):

1. **Below 32 query tokens, no XMX anywhere.** The SYCL backend's XMX paths (oneDNN
   SDPA, MKL) require ≥ 32 query tokens. Every smaller batch — including **ordinary
   F16 decode on any GQA model** — goes to TILE/VEC kernels with no matrix engines.
   The card pays for XMX silicon and doesn't use it in the hot loop.
2. **Non-power-of-two GQA re-reads K/V many times.** TILE packs only power-of-two
   divisors of the GQA ratio: GQA 5 → 5 passes over each KV head per call, GQA 6 → 3,
   GQA 7 → 7 (e.g. Qwen2.5-7B). On the A770 this hits single-token decode, which is
   bandwidth-bound. The Vulkan backend already packs any G at N ≤ 8 — so the fix has
   a precedent inside llama.cpp itself.
3. **Quantised-KV conversion wastes 3 of 4 work-items.** A 128-value KV row launches
   256 work-items; 192 exit immediately — the same launch-geometry bug we measured and
   fixed on NVIDIA (V100: conversion 1.7–6.9× faster, bit-identical). Applies to
   batches 3–31 with a quantised cache.
4. **The XMX silent-fallback trap.** Our static probes found that the Battlemage/Xe-HPC
   *SIMD16* forms of the XMX builtins **compile successfully on acm-g10 with zero DPAS
   instructions and no warning** — the store writes registers that were never written.
   Code copied from Xe2/PVC examples can look correct, build clean, and silently not
   use XMX at all. This matters to anyone writing Arc kernels, beyond llama.cpp.

## What exists today

- **Five llama.cpp SYCL patches** (plus a draft and two candidate fixes): padded GQA
  in TILE (0001, the §2 fix, ported from volta-fa P-6), its tests (0002), the
  conversion geometry + vectorised dequant ports (0003/0004, from volta-fa's measured
  pair), and biased-V FA tests with an FP32 CPU reference (0005, from P100 — the
  FP16-accumulation accuracy bug also lives in TILE here; two candidate fixes sit in
  `p11c/`). Correctness-checked on the Intel OpenCL CPU device; **not timed** — no
  Arc GPU has run them.
- **A no-sudo static toolchain with a per-kernel `acm-g10` report**: SIMD width, GRF,
  threads/XVE, spill, SLM, DPAS and FP64 counts — plus the probe fixtures that pinned
  down the legal XMX forms (SIMD8 `joint_matrix` 8×8×16 f16/bf16→f32 works; the SIMD16
  forms silently emit no DPAS; IGC's spill retry silently changes SIMD width unless
  pinned; SLM limits are not enforced at compile time).
- **A Phase 0 day-one kit** (`bench/phase0.sh`, `bench/patches/run.sh`) with acceptance
  criteria fixed in advance in the design docs.

## Why you'd want to try it — or help

Same deal as the P100 repo: whoever owns the card produces the first real numbers.
The kit builds every patch variant, runs correctness suites and interleaved A/B timing
in one session, and records `vulkaninfo` + driver so the two-backend baseline (SYCL vs
Vulkan per shape) is captured properly. If you own an A770, Phase 0 answers: how much
bandwidth the GQA re-reads actually cost at depth, whether the padded-GQA patch
transfers, and whether TILE's FP16 output sums need the accuracy fix here too.

## How to try it

```bash
git clone https://github.com/Frostbite1536/A770
# llama.cpp at bench/LLAMA_COMMIT (79e2e74), then apply
# third_party/patches/000{1..5}-*.patch in order
```

Build with oneAPI 2026.1 (`icpx`, Level Zero). Patches are not rebased past the pinned
commit. Runbook for the first session: `docs/A770_BOX.md`, `prompts/a770-session-1.md`.

## Future work

Phase 0 first (`docs/ROADMAP.md`); after that, in expected order of value: dispatch
fixes above (cheapest), then the first XMX small-batch attention kernel on Alchemist —
SIMD8 `joint_matrix`, DPAS-count-checked by `bin/xe-report.py` so it can't silently
fall off the matrix engines — then quantised KV without the conversion tax, and
dual-card. Sibling projects: [volta-fa](https://github.com/Frostbite1536/volta-fa)
(V100), [P100](https://github.com/Frostbite1536/P100) (Pascal).
