# Bench-only patches

These local benchmark patches are **never upstream** and are separate from the
kernel series and CPU-only `test-only/` patches. Phase 0 applies only its explicit
every-build list, once to the shared llama.cpp tree before building either backend.

| Patch | Provenance | Purpose | Builds |
|---|---|---|---|
| `0001-tests-mul_mat-perf-cases-1-16.patch` | A770 ticket p12, against llama.cpp `79e2e74eb`; same cases as P100 ticket p13 | Complete contiguous F32-RHS MUL_MAT perf coverage for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K at four Llama shapes and n=1–16; skip existing cases (280 additions, 320 total). Eval cases and tolerances unchanged. **Never upstream.** | Every build: SYCL and Vulkan, all timing arms |
| `0002-tests-mul_mat-activation-range.patch` | A770 ticket p14, against llama.cpp `79e2e74eb`; ports `test_mul_mat_range` from volta-fa local `0009` | 96 eval-only MUL_MAT cases: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K; n=1,2,4,8,16,32; m=128,4096; K=1024; activation scales 1e5 and 1e-5. Skip Q4_0/Q8_0 at 1e-5 because the CPU reference's Q8_0 FP16 d underflows. These tests **fail on q8_1 paths by design** and **never go upstream or into an upstream test list**. Existing tolerances unchanged. | Every build: SYCL and Vulkan; Phase 0 test filter `src1_scale`, plus SYCL no-gather and no-opt arms |

For correctness on the CPU only, opt in with
`SYCL_CPU_BENCH_PATCHES=1 LLAMA=$PWD/build/llama bin/sycl-cpu-tbo.sh test -o MUL_MAT -p src1_scale`.
The harness default is unchanged. Label these results **SYCL on the Intel OpenCL
CPU device**; they do not verify the A770 or the fused XMX GEMM path.

The local `sycl_cpu_mul_mat_range` ctest runs the default range arm when the
oneAPI CPU runtime and compiler are installed; otherwise it skips with a message.
It checks that all 96 cases execute and that the per-case counts, summary, and
exit status agree. It clones and builds SYCL llama.cpp on first run; `ctest -LE sycl_cpu` skips it. Numerical failures are expected outcomes, preserved in
`build/p14-range/ctest-range-sycl.txt`; a passing ctest means the corpus ran
completely, not that its numerical cases all passed. Harness failures and
incomplete runs fail ctest.

**2026-10-16 correctness record — SYCL on the Intel OpenCL CPU device.**

Scratch clone: `git clone --no-hardlinks /home/p51/Github/A770/third_party/llama.cpp build/llama`;
base `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`. Exported patch via a scratch-only
commit and `git format-patch -1 --stdout`. No A770 repository commit was made.

`git apply --check`, application, and reverse-check all passed for `0002` on:

| Starting tree | Result |
|---|---|
| Pin alone | PASS |
| Pin + bench-only `0001` | PASS |
| Pin + patches `0001`–`0005` | PASS |
| `bench/patches/run.sh` upstream: pin + `0005` + the exact `prepare()` eval/perf injection | PASS |
| Pin + patches `0001`–`0005` + bench-only `0001` | PASS |

The three runs used the command below, clearing the three path switches first;
only the named arm override was set. `SYCL_CPU_BENCH_PATCHES=1` applies both
explicit every-build bench patches; the CPU harness also applies its existing
test-only CPU patch. F16 remained ON.

```sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
env -u GGML_SYCL_ENABLE_OPT -u GGML_SYCL_XMX_GATHER_TYPES -u GGML_SYCL_XMX_GATHER_SHAPES \
  SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p14-sycl-cache \
  SYCL_CPU_BENCH_PATCHES=1 LLAMA="$PWD/build/llama" \
  bin/sycl-cpu-tbo.sh test -o MUL_MAT -b SYCL0 -p src1_scale
# nofg: same command, adding GGML_SYCL_XMX_GATHER_TYPES=0 before bin/sycl-cpu-tbo.sh
# noopt: same command, adding GGML_SYCL_ENABLE_OPT=0 before bin/sycl-cpu-tbo.sh
```

| Arm | Startup ENABLE_OPT / XMX_GATHER_TYPES | Result | Exit |
|---|---|---|---:|
| Defaults | 1 / -1 | 48/96 passed; 48 failed | 1 |
| `GGML_SYCL_XMX_GATHER_TYPES=0` | 1 / 0 | 48/96 passed; 48 failed | 1 |
| `GGML_SYCL_ENABLE_OPT=0` | 0 / -1 | 48/96 passed; 48 failed | 1 |

Startup identifies `SYCL0` as Intel OpenCL CPU, Intel Xeon E3-1505M v6 @ 3.00GHz,
driver `2026.21.7.0.24_160000`; F16=yes, Reorder=N, XMX_GATHER_SHAPES=255,
DYNAMIC_PRECISION=F16, DYNAMIC_REQUIRED_PRECISION=F32. No fused-XMX availability
message appeared. This CPU device has no XMX. The fused XMX GEMM path was not verified.

Per-case results below are **SYCL on the Intel OpenCL CPU device**. Each cell is
**defaults / nofg / noopt**; P=PASS, F=FAIL. All cases use F32 src1 and K=1024.
The recorder checked the exact 96-case grid in each arm, with no duplicate or
missing case, and checked the per-case counts against each runner summary.
Q4_0/Q8_0 at 1e-5 are omitted because the CPU reference's Q8_0 FP16 d underflows.

| type_a | m | scale | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 |
|---|---:|---:|---|---|---|---|---|---|
| q4_0 | 128 | 100000 | F/F/F | F/F/F | F/F/F | F/F/F | F/F/F | F/F/F |
| q4_0 | 4096 | 100000 | F/F/F | F/F/F | F/F/F | F/F/F | F/F/F | F/F/F |
| q8_0 | 128 | 100000 | F/F/F | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q8_0 | 4096 | 100000 | F/F/F | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q4_K | 128 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q4_K | 128 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |
| q4_K | 4096 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q4_K | 4096 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |
| q5_K | 128 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q5_K | 128 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |
| q5_K | 4096 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q5_K | 4096 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |
| q6_K | 128 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q6_K | 128 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |
| q6_K | 4096 | 100000 | P/P/P | P/P/P | P/P/P | P/P/P | F/F/F | F/F/F |
| q6_K | 4096 | 1e-05 | P/P/P | F/F/F | F/F/F | F/F/F | P/P/P | P/P/P |

A770 and Vulkan execution: **not run**. Fused XMX GEMM path: **not verified**.
No kernel source was changed by this ticket; no new xe-report table applies.
