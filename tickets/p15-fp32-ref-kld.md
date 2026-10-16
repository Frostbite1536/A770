# Ticket p15-fp32-ref-kld: FP32-reference KL divergence for SYCL matmul paths (development switch + Phase 0 step)

Your current directory is a git worktree of A770 (branch `p15-fp32-ref-kld`, based on `main` @ `32fcda0`). Work only here.

## Rules (mandatory, repeated in every ticket)
- Target is Intel Arc A770 (Xe-HPG, acm-g10) ONLY. XMX only through DPAS at sub-group size 8: SYCL `joint_matrix` 8x8x16 (f16/bf16 in, f32 accumulate) or the OpenCL SIMD8 builtins with an `int8` A operand. Do NOT use SIMD16 DPAS forms or shapes, 2D block loads/stores, TF32, FP8, FP64, or code written for Battlemage/Xe2/PVC (sycl-tla, XeTLA). If you think you need one, STOP and explain.
- Every kernel declares its sub-group size. QK^T accumulates in FP32.
- The toolchain lives in the MAIN checkout. Start every shell command with `export PATH=/home/p51/Github/A770/.venv/bin:$PATH;` so `cmake`, `ninja`, `ctest` and `python` resolve there. Use `/home/p51/Github/A770/tools/` by absolute path. Never install anything. Build into `build/` inside this worktree.
- For every kernel you touch run `python bin/xe-report.py <file> [--expect-dpas REGEX]` and paste its table. Spill must be 0; XMX kernels must show DPAS > 0 at SIMD 8.
- Never write the NxN score matrix to global memory (INV-NUM-001). Layout/mask/causal semantics are fixed by `include/a770/attention.h`.
- Do NOT commit, push, or run git commands that write in the A770 repo or this worktree. Leave changes uncommitted; the manager commits. Exception: inside your scratch llama.cpp clone under `build/` (see Steps 0) you may `git apply`, `git commit` and `git format-patch` to produce the patch.
- Do not loosen tolerances and do not edit existing tests to make them pass; report the failure instead.
- Never invent benchmark numbers. There is no A770 on this machine: anything needing one is "not run".
- Do not read .env, tokens, or model weights. No network.
- Stay inside "Files you may touch". If the task needs more, STOP and explain.

## Background
`docs/research/volta-fa-p9-lessons-2026-10-11.md` §4, "Accuracy method". volta-fa measured accuracy on real models with a development switch that routes quantised matmuls to dequantise + FP32 GEMM, and llama-perplexity KL divergence of each path against that reference. On Qwen3.8-27B it showed mean KLD 0.0017 for q8_1 activations vs 0.0001 for FP16 activations, which perplexity against upstream cannot show.

volta-fa's CUDA switch is in `~/Github/volta-fa/bench/results/p9-scale2-20261011-195231/p9-ref.diff` (`GGML_CUDA_F32_REF`). On SYCL:
- the fused XMX GEMM already honours an FP32 request through `GGML_SYCL_DYNAMIC_PRECISION` (tf32 or f32 operand; check `ggml-sycl/fused-gemm.cpp` and `ggml_sycl_src1_prec` at the pin);
- but a full-FP32 reference needs every quantised matmul, including MMVQ/DMMV and MMQ, to go through dequantise + oneMKL SGEMM.

## Design (manager decided, do not change)
1. **`third_party/patches/bench-only/0003-sycl-f32-reference-matmul-dev-switch.patch`** (against the pin). Create `bench-only/` and its README as ticket p12 describes if needed. This patch goes **only into the reference build**; say so in the README.
   - With `GGML_SYCL_F32_REF` set, quantised dense MUL_MAT is routed past the MMVQ/DMMV/MMQ and fused-GEMM paths. The weights are dequantised to an FP32 temporary (the existing `to_fp32` converters) and multiplied with oneMKL SGEMM (the existing `dpct::gemm` / oneMKL call path used elsewhere in `ggml-sycl.cpp`; find it and reuse it).
   - The first time it takes effect, it prints one stderr line: `GGML_SYCL_F32_REF: quantised MUL_MAT via dequantize + SGEMM`.
   - MUL_MAT_ID is not covered; say so in the patch comment.
2. **New opt-in step `kld` in `bench/phase0.sh`** (SYCL only). (Manager notes: `bench-only/` already holds `0001` and `0002`, which `phase0.sh` applies from its explicit every-build `bench_patches` list. `0003` must NOT join that list and must never be applied to `$LLAMA`. Build the reference tree in a separate local clone of `$LLAMA` (e.g. `$LLAMA-f32ref`, `git clone --no-hardlinks`), at the same commit, with the same every-build patches plus `0003`, and log its HEAD and applied patches. Add a README row saying `0003` goes only into the reference build.) It requires `bench/models.txt` and `PPL_TEXT`; otherwise it logs one line and skips.
   - **Builds:** `llama-perplexity` in the normal SYCL tree, and in a reference tree = the SYCL tree + this patch.
   - **Reference per model:**
     ```
     GGML_SYCL_F32_REF=1 llama-perplexity -m M -f PPL_TEXT -c 512 -b $KLD_UB -ub $KLD_UB -ngl 99 -fa on --chunks $KLD_CHUNKS --kl-divergence-base $OUT/<model>.ref.kld
     ```
     with `KLD_UB` default 16 and `KLD_CHUNKS` default 8. Check that the stderr line appeared.
   - **Arms** (normal tree): `sycl` (defaults), `sycl-nofg` (`GGML_SYCL_XMX_GATHER_TYPES=0`) and `sycl-noopt` (`GGML_SYCL_ENABLE_OPT=0`), each with `--kl-divergence-base … --kl-divergence`.
   - Delete the base file afterwards.
   - **Summary:** the `Mean KLD`, `Maximum KLD`, `RMS Δp` and `Same top p` lines per arm.

## Files you may touch
- `third_party/patches/bench-only/0003-sycl-f32-reference-matmul-dev-switch.patch` (new), `third_party/patches/bench-only/README.md`
- `bench/phase0.sh`
- `tests/CMakeLists.txt` (a CPU-device ctest of the switch, skipped without the oneAPI CPU runtime); follow the pattern of the existing `sycl_cpu_mul_mat_range` test, with `LABELS sycl_cpu`, and do not edit that test
Nothing else.

## Steps
0. **Harness (manager note).** This worktree has no `third_party/llama.cpp` (gitignored). Make a scratch clone with `git clone --no-hardlinks /home/p51/Github/A770/third_party/llama.cpp build/llama` (it is at the pin `79e2e74`); never write to the main checkout's tree. `LLAMA=$PWD/build/llama bin/sycl-cpu-tbo.sh [test-backend-ops args]` builds and runs on the Intel OpenCL CPU device (see `docs/TOOLCHAIN.md` "SYCL on the CPU"); use `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p15-sycl-cache`. Label every CPU-device result "SYCL on the Intel OpenCL CPU device".
   Check `0003` applies (a) to the pin alone, (b) on top of bench-only `0001` and `0002`, and (c) on top of `third_party/patches/0001`–`0005`. Report each.
1. Write the switch in a scratch worktree at the pin, and build it with `bin/sycl-cpu-tbo.sh`'s CPU-device setup.
2. **Positive control on the SYCL CPU device** (manager addition: also run `-p src1_scale` with `SYCL_CPU_BENCH_PATCHES=1`, i.e. bench-only `0002`'s range cases, with and without `GGML_SYCL_F32_REF=1`, and report pass counts; the clone then carries `0001`+`0002`+`0003`): with `GGML_SYCL_F32_REF=1`, run `test-backend-ops test -b SYCL0 -o MUL_MAT -p 'type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32'` and report the pass count and the stderr line. Then run without the variable for comparison. Label both as CPU device.
3. Implement the `kld` step and the ctest.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure` (all; the new test may skip)
- `shellcheck -x bench/phase0.sh bin/*.sh`, `bin/check-forbidden.sh`

## Done means
On the A770, `STEPS=kld` gives each SYCL path's KL divergence against an FP32 reference.

Final report:
- files changed;
- each command with its result line;
- the CPU-device MUL_MAT counts with and without the switch;
- what was NOT verified (the A770; no model was run).

Do not interpret.
