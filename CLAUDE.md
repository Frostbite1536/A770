# A770 agent instructions

Attention and multi-GPU work for the Intel Arc A770 (Xe-HPG, die `acm-g10`). SYCL (oneAPI DPC++) and OpenCL C kernels, CMake. Baseline and integration target: llama.cpp's SYCL and Vulkan backends (`third_party/llama.cpp`, gitignored clone; our changes live as patches in `third_party/patches/`). Sibling of `~/Github/volta-fa` (same process, different hardware).

## Hard rules (every agent, every ticket)

- **Xe-HPG `acm-g10` ONLY.** XMX is reached through DPAS at **sub-group size 8**: SYCL `joint_matrix` 8×8×16 (f16 or bf16 in, f32 accumulate), or the OpenCL SIMD8 builtins (`intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc)`). Do NOT use SIMD16 DPAS forms or shapes (Xe2/Xe-HPC: on acm-g10 the OpenCL builtin **compiles to no DPAS without an error**, and `joint_matrix` 8×16×16 crashes IGC), 2D block loads/stores/prefetch, TF32, FP8 or FP64 (no FP64 hardware). Do not copy kernels written for Battlemage/PVC (sycl-tla/CUTLASS-SYCL, XeTLA Xe2 configs). If you think you need one of these, STOP and say so.
- **Pinned toolchain**: oneAPI DPC++ 2026.1.x, compute-runtime 26.35.39758.10 + IGC 2.41.5 (`docs/TOOLCHAIN.md`). Changing a pin needs a manager decision.
- Every kernel change reports SIMD width, GRF mode, threads/XVE, spill, private, SLM and DPAS count (`bin/xe-report.py`). Spill > 0 is a failing result. An XMX kernel with DPAS = 0 is a failing result.
- Every kernel declares its sub-group size (`[[sycl::reqd_sub_group_size(N)]]` / `intel_reqd_sub_group_size`); never let IGC pick.
- Never write the N×N score matrix to global memory in the fused path.
- Correctness before speed: compare against the reference in `tests/`. Do not loosen tolerances or edit existing tests to pass; report the real error.
- Do not invent benchmark numbers. Anything not run on an A770 is "not run". Numbers from Battlemage, Lunar Lake or another Arc are not A770 numbers.
- Read `docs/INVARIANTS.md` before changing kernel behaviour; changing an invariant needs a manager decision.
- Never read or commit `.env`, tokens, model weights, or raw VTune/unitrace result directories.
- Workers: no git commands that write; leave changes uncommitted. The manager commits.
- Stay in the ticket's file scope. Out-of-scope need → STOP and explain.

## Commands

```bash
# dev machine (no Arc GPU, no system oneAPI). Setup: docs/TOOLCHAIN.md
bin/setup-dev.sh                   # .venv + tools/neo (ocloc + IGC); WITH_DPCPP=1 adds icpx, oneMKL SYCL, OpenCL CPU runtime 2026.1 (~4.4 GB)
export PATH=$PWD/.venv/bin:$PATH   # once per shell
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
python bin/xe-report.py src/attention/<file>.{cpp,cl} [--large-grf] [--expect-dpas REGEX]   # static acm-g10 report
bin/check-forbidden.sh && shellcheck -x bin/*.sh

# A770 box (oneAPI 2026.1 sourced)
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx && cmake --build build && ctest --test-dir build --output-on-failure
bench/phase0.sh                    # Phase 0 baseline kit: llama.cpp SYCL + Vulkan FA support/test/perf + llama-bench grid
```

## Conventions

- Files < ~1,500 lines; one kernel family per file.
- Tile sizes, sub-group size and work-group shape are template parameters; no magic numbers inline.
- Tensor layout, mask and causal semantics are defined in `include/a770/attention.h`; do not reinterpret them per kernel.
- Every SYCL/Level Zero call that can fail is checked; asynchronous SYCL errors go through an async handler, never ignored.
- Comments explain *why* (layout, padding, GRF/occupancy trade-off), not what.
- Commits say why each test exists. Worker-written code: "Implemented by <worker> from ticket <name>; verified by <manager>".

## Docs map

[HANDOFF](docs/HANDOFF.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [INVARIANTS](docs/INVARIANTS.md) · [ROADMAP](docs/ROADMAP.md) · [WORKFLOW](docs/WORKFLOW.md) · [PROFILING](docs/PROFILING.md) · [TOOLCHAIN](docs/TOOLCHAIN.md) · [research/](docs/research/) · [REVIEW.md](REVIEW.md) · [prompts/](prompts/)
