# A770

Attention and multi-GPU work for the Intel Arc A770 (Xe-HPG, `acm-g10`), in SYCL and OpenCL C, meant to plug into llama.cpp. Sibling project of [volta-fa](../volta-fa) (NVIDIA V100): same process and numerics contract, different hardware.

**Status: pre-hardware, day-one kit ready.** Nothing has run on an A770. What exists:
- a CPU attention reference and numerics model with measured tolerances (carried over from volta-fa);
- a no-sudo static toolchain (ocloc + IGC + icpx 2026.1) with a per-kernel acm-g10 report: SIMD, GRF, threads/XVE, spill, SLM, DPAS and FP64 counts;
- compile probes that pinned down which XMX forms are legal on the A770;
- a code study of how llama.cpp's SYCL and Vulkan backends run FlashAttention on it;
- a Phase 0 measurement kit (not run);
- five llama.cpp SYCL patches (0001-0004 ported from volta-fa, 0005 from P100), checked for correctness on the Intel OpenCL CPU device (`bin/sycl-cpu-tbo.sh`), not timed:
  - 0001/0002: padded GQA in the TILE FA kernel (P-6);
  - 0003/0004: faster K/V conversion (P-8);
  - 0005: biased-V FA test cases with an FP32 CPU reference (P-11, test-only); two candidate fixes for TILE's FP16 output sums in `third_party/patches/p11c/`;
- an A/B harness and runbook for the first A770 session: `bench/patches/run.sh`, [docs/A770_BOX.md](docs/A770_BOX.md), `prompts/a770-session-1.md`. Acceptance criteria are fixed in advance in the P-6/P-8/P-11 design docs.

Every performance claim in this repo must link to a file in `profiles/` or `bench/results/`. Until it does, treat it as a hypothesis.

## Why

The A770 16 GB is a cheap card with XMX matrix engines, but most Intel GPU kernel work now targets Battlemage (Xe2) and Data Center GPU Max. llama.cpp's SYCL backend runs FA below 32 query tokens on kernels that don't use XMX, and its dispatch copies CUDA rules that waste bandwidth for GQA ratios that aren't powers of two (`docs/research/llama-cpp-arc-fa-dispatch.md`). Goals, in order: faster decode and small-batch attention at depth, a quantised KV cache that doesn't pay a conversion tax, then dual-card setups.

## Read this before trusting the plan

- llama.cpp already runs FA on the A770 through **two** backends (SYCL and Vulkan). The baseline per shape is the faster of the two. The first job is **Phase 0: measure and find the gap**.
- The cheapest wins we expect are **dispatch fixes ported from volta-fa** (padded GQA, one GQA predicate, K/V conversion geometry), not new kernels.
- XMX on the A770 is **SIMD8 DPAS only**. Code written for Battlemage (SIMD16) can compile with no error and no XMX instruction. See `docs/research/xe-hpg-static-probes-2026-10-10.md`.

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Ground rules

- Target: `acm-g10` only, oneAPI DPC++ 2026.1, IGC 2.41.5 for static reports. XMX via `joint_matrix` 8×8×16 (f16/bf16 → f32) at sub-group 8. No SIMD16 DPAS, 2D block loads, TF32, FP8 or FP64. Details in [docs/INVARIANTS.md](docs/INVARIANTS.md).
- Correctness first: every kernel is checked against an FP32 reference before it is timed.
- Profile-driven: changes are justified by VTune/unitrace output stored in `profiles/`.

## Layout

```
src/attention/   attention kernels (SYCL / OpenCL C), when Phase 0 justifies one
include/         shared contract (layout, mask, causal) + tolerances
tests/           reference, numerics model, parity tests; tests/xe/ = compile-probe fixtures for xe-report
bench/           Phase 0 kit, analysis script, results/
profiles/        committed VTune/unitrace text summaries (raw results are gitignored)
third_party/     llama.cpp clone (gitignored) + patches/ (committed)
tickets/         agent work tickets
bin/             toolchain setup, xe-report, forbidden-feature check, worker orchestration
prompts/         reusable prompts
docs/            ARCHITECTURE, INVARIANTS, ROADMAP, WORKFLOW, PROFILING, TOOLCHAIN, research/
```

## Build and test

Without an Arc GPU or oneAPI (see [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md)):

```bash
bin/setup-dev.sh                   # WITH_DPCPP=1 also installs icpx, oneMKL SYCL and the OpenCL CPU runtime (~4.4 GB)
export PATH=$PWD/.venv/bin:$PATH   # once per shell
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python bin/xe-report.py tests/xe/dpas_simd8.cl
```

On the A770 box (oneAPI 2026.1 sourced):

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx && cmake --build build && ctest --test-dir build --output-on-failure
bench/phase0.sh
```

## How we build it

An Opus manager plus worker models via Codex, with the same loop as volta-fa (manager writes narrow tickets, workers implement, the profiler is the judge). See [docs/WORKFLOW.md](docs/WORKFLOW.md). Agent rules are in [CLAUDE.md](CLAUDE.md); review rules in [REVIEW.md](REVIEW.md).

## License

AGPL-3.0 ([LICENSE](LICENSE)). The patch series in `third_party/patches/` is additionally
available under llama.cpp's MIT license so it can be merged upstream without relicensing
this repo; everything else is AGPL-3.0 only.
