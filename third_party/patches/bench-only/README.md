# Bench-only patches

These local benchmark patches are **never upstream** and are separate from the
kernel series and CPU-only `test-only/` patches. Phase 0 applies only its explicit
every-build list, once to the shared llama.cpp tree before building either backend.

| Patch | Provenance | Purpose | Builds |
|---|---|---|---|
| `0001-tests-mul_mat-perf-cases-1-16.patch` | A770 ticket p12, against llama.cpp `79e2e74eb`; same cases as P100 ticket p13 | Complete contiguous F32-RHS MUL_MAT perf coverage for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K at four Llama shapes and n=1–16; skip existing cases (280 additions, 320 total). Eval cases and tolerances unchanged. **Never upstream.** | Every build: SYCL and Vulkan, all timing arms |
| `0002-tests-mul_mat-activation-range.patch` | A770 ticket p14, against llama.cpp `79e2e74eb`; ports `test_mul_mat_range` from volta-fa local `0009` | 96 eval-only MUL_MAT cases: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K; n=1,2,4,8,16,32; m=128,4096; K=1024; activation scales 1e5 and 1e-5. Skip Q4_0/Q8_0 at 1e-5 because the CPU reference's Q8_0 FP16 d underflows. These tests **fail on q8_1 paths by design** and **never go upstream or into an upstream test list**. Existing tolerances unchanged. | Every build: SYCL and Vulkan; Phase 0 test filter `src1_scale`, plus SYCL no-gather and no-opt arms |
| `0003-sycl-f32-reference-matmul-dev-switch.patch` | A770 ticket p15, against llama.cpp `79e2e74eb` | Development switch `GGML_SYCL_F32_REF` routes quantised dense MUL_MAT through existing FP32 dequantisation and oneMKL SGEMM, bypassing DMMV/MMVQ/MMQ, dense FFN fusion, fused GEMM, FP16 conversion and oneDNN. Logs once on first use. MUL_MAT_ID is not covered. **Never upstream.** | **Reference build only:** `STEPS=kld` creates a separate local clone at the normal tree's HEAD and replays its tracked patches plus `0003`. Never applied to `$LLAMA` or included in the every-build list. |

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

`PPL_TEXT=/path/to/text STEPS=kld BACKENDS=sycl bench/phase0.sh` builds
`llama-perplexity` in the normal SYCL tree and a separate clone under `build/`.
It requires `bench/models.txt` and `PPL_TEXT`; otherwise it logs one line and
skips. `KLD_UB=16` and `KLD_CHUNKS=8` are the defaults for all runs (context 512,
GPU layers 99, FA on). The reference alone gets `GGML_SYCL_F32_REF=1`; its stderr
must contain the switch announcement. Each normal-tree arm (`sycl`, `sycl-nofg`,
`sycl-noopt`) compares against that model's reference logits. The base file is
removed after the arms, including on failure. `kld-summary.txt` records Mean KLD,
Maximum KLD, RMS Δp and Same top p per arm. `kld-patches.txt` records both HEADs,
the replayed source delta and the explicit patches. Set the switch before process
startup; its presence enables it (including a value of `0`).

The `sycl_cpu_mul_mat_f32_ref` ctest uses its own scratch clone, `<build>/llama-f32ref`, so `0003` never enters the clone the range test uses.
It requires all 96 activation-range cases to pass and exactly one reference
announcement, with `LABELS sycl_cpu`; it skips without the oneAPI CPU runtime.
It does not verify the A770. No kernel is added or modified by `0003`: its changes
are host dispatch only, reusing existing converters and SGEMM.

**2026-10-16 p15 correctness record — SYCL on the Intel OpenCL CPU device.**

Scratch clone: `git clone --no-hardlinks /home/p51/Github/A770/third_party/llama.cpp build/llama`,
base `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6`. Exported `0003` using a
scratch-only commit and `git format-patch -1 --stdout`. A770 repository changes
remain uncommitted. `git apply --check`, application, and reverse-check of `0003`
all passed on each required starting tree:

| Starting tree | Result |
|---|---|
| Pin alone | PASS |
| Pin + bench-only `0001` + `0002` | PASS |
| Pin + kernel patches `0001`–`0005` | PASS |

The active clone also passes reverse-check of `0003` with the test-only CPU patch
and both bench corpus patches present. No kernel body or launch was changed;
there is no new xe-report table for this host-dispatch patch.

The commands below used F16 ON and the default path settings. The broad filter
ran before adding the bench corpus patches. The range runs carry bench-only
`0001` + `0002` + `0003` and the harness's test-only CPU patch.

```sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
env -u GGML_SYCL_ENABLE_OPT -u GGML_SYCL_XMX_GATHER_TYPES -u GGML_SYCL_XMX_GATHER_SHAPES \
  GGML_SYCL_F32_REF=1 SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p15-sycl-cache \
  LLAMA="$PWD/build/llama" bin/sycl-cpu-tbo.sh test -b SYCL0 -o MUL_MAT \
  -p 'type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32'
# Unswitched control: same command, replacing GGML_SYCL_F32_REF=1 with env's -u GGML_SYCL_F32_REF.

env -u GGML_SYCL_ENABLE_OPT -u GGML_SYCL_XMX_GATHER_TYPES -u GGML_SYCL_XMX_GATHER_SHAPES \
  GGML_SYCL_F32_REF=1 SYCL_CPU_BENCH_PATCHES=1 \
  SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p15-sycl-cache \
  LLAMA="$PWD/build/llama" bin/sycl-cpu-tbo.sh test -b SYCL0 -o MUL_MAT -p src1_scale
# Unswitched control: same command, replacing GGML_SYCL_F32_REF=1 with env's -u GGML_SYCL_F32_REF.
```

All four results below are **SYCL on the Intel OpenCL CPU device**:

| Filter | `GGML_SYCL_F32_REF` | Result | Exit | Announcement count |
|---|---|---|---:|---:|
| Broad quantised F32-RHS MUL_MAT filter above | `1` | 477/477 passed | 0 | 1 |
| Broad quantised F32-RHS MUL_MAT filter above | unset | 477/477 passed | 0 | 0 |
| `src1_scale` | `1` | 96/96 passed | 0 | 1 |
| `src1_scale` | unset | 48/96 passed; 48 failed | 1 | 0 |

The reference stderr line is exactly:
`GGML_SYCL_F32_REF: quantised MUL_MAT via dequantize + SGEMM`.
Logs are in `build/p15-f32ref/positive-{ref,default}.txt` and
`build/p15-f32ref/range-{ref,default}.{txt,err}`. Per-case counts match the summaries.
No existing test or tolerance was changed.

A local mocked `STEPS=kld` run passed clone/HEAD/patch isolation, the default
arguments and three arm environments, summary collection, base cleanup, the
one-line missing-input skip, and rejection/cleanup when the reference
announcement is absent. Its executable used fixture text only; no model ran.
`build/p15-f32ref/mock-*.txt` contains these harness logs.

`shellcheck -x bench/phase0.sh bin/*.sh`: PASS.
`bin/check-forbidden.sh`: `ok`. `git diff --check`: PASS.

A770 execution, A770 performance, fused XMX execution, and model KL divergence:
**not run**. No model weights were read and no model was run.

Final required suite (all tools read from the main checkout):

```sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
export A770_TOOLS=/home/p51/Github/A770/tools;
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p15-sycl-cache;
cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build --output-on-failure
```

Result: **100% tests passed out of 33**, no skips. Both tests labelled `sycl_cpu`
passed: the existing range ctest recorded 48/96 numerical passes and checked
corpus completeness; the new reference ctest recorded 96/96 passes and exactly
one stderr announcement. These are **SYCL on the Intel OpenCL CPU device**.
The existing range ctest retains its original `/tmp/p14-sycl-cache` setting;
the new ctest and the four manual controls use `/tmp/p15-sycl-cache`.
Full suite log: `build/p15-f32ref/ctest-all.txt`.
