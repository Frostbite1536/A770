# P11d benchmark-kit CPU plumbing smoke, 2026-10-11

Target: Intel Arc A770 (Xe-HPG, acm-g10). **Every A770 result: not run.**
CPU smoke proves plumbing only. llama.cpp pin: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6.
The only tracked code/docs edits are bench/patches/run.sh, docs/A770_BOX.md,
and prompts/a770-session-1.md. Kernels, patch files, existing cases and tolerances
were not edited. Results below come from the Intel OpenCL CPU device.

Commands (run from this worktree; all shell commands begin with the main
checkout's venv PATH, and the main checkout/toolchain is read-only):

```bash
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; export A770_TOOLS=/home/p51/Github/A770/tools
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; export CMAKE_PREFIX_PATH=/home/p51/Github/A770/tools/oneapi-2026.1 GIT_OPTIONAL_LOCKS=0
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; TASK_WORK=/tmp/a770-p11d-20261011-iupP6P TASK_OUT="$PWD/bench/results/p11d-run-sh-cpu-smoke-20261011"
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; TASK_FILTER='v_bias=2|hsk=128,hsv=128,nh=2,nr23=\[7,1\],kv=4096,nb=(1|2|3|4),mask=1,sinks=0,.*type_K=f16,type_V=f16'
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; CPU_SMOKE=1 FORCE=1 STEPS="env build test" VARIANTS="upstream fp32f cap" TEST_FILTER="$TASK_FILTER" BUILD_FLAGS_EXTRA="-DGGML_SYCL_DEVICE_ARCH= -DGGML_SYCL_DNN=OFF -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF" WORK="$TASK_WORK" OUT="$TASK_OUT" bench/patches/run.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2 CPU_SMOKE=1 FORCE=1 STEPS=test VARIANTS="upstream cap" TEST_FILTER="$TASK_FILTER" WORK="$TASK_WORK" OUT="$TASK_OUT/forced-geometry" bench/patches/run.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; CPU_SMOKE=1 FORCE=1 STEPS=env VARIANTS=upstream WORK="$TASK_WORK" OUT="$TASK_OUT/env-only" bench/patches/run.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; shellcheck -x bin/*.sh bench/patches/run.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; bin/check-forbidden.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; cmake -S "$PWD" -B /tmp/a770-p11d-repo-build -G Ninja
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; cmake --build /tmp/a770-p11d-repo-build -j 2
export PATH=/home/p51/Github/A770/.venv/bin:$PATH; ctest --test-dir /tmp/a770-p11d-repo-build --output-on-failure
```

WORK was created fresh with mktemp under /tmp; the second smoke reuses it.
CPU_SMOKE selects the environment from docs/TOOLCHAIN.md "SYCL on the CPU":
main oneAPI 2026.1, opencl:cpu, GGML_SYCL_TEST_CPU=1, optimized JIT spir64,
F16 ON, oneDNN and Level Zero allocation support OFF. CMAKE_PREFIX_PATH
locates the main toolkit's MKL. No installation or network was used.

No gating changes were needed: the existing test-only patch accepts both launch
geometry overrides under CPU_SMOKE=1 FORCE=1. The physical env geometry is
reported separately from these overrides. A scratch copy redirects phase0's
env capture into this OUT so the smoke writes no other repository result path.

The FA pass runs once with TEST_FILTER. Rows are classified by the case name's
v_bias=2 field. Table rc is the slice status; the actual test process rc is in
run.log. Expected upstream biased-V failures remain listed in its failure file
and full log, while only fp32f/cap FA-BIASV failures affect the harness exit.
Missing summaries, crashes and an entirely empty run remain failures.

Verified result lines:

- shellcheck: exit 0, no diagnostics (shellcheck.log).
- check-forbidden: exit 0, `ok` (forbidden.log).
- repo ctest: `100% tests passed out of 28`; exit 0 (repo-checks.log).
- env-only: exit 0; SYCL0 max_work_group_size=8192,
  max_compute_units=8, max_wg_per_cu=1024 (env-only/console.log and summary.md).
- P9 perf block byte-identical to HEAD; P11 acceptance quote verbatim: PASS.

The initial fresh-WORK full build/test attempt passed every selected case, but
exited 2 after duplicate cap rows and a shell syntax error. I edited run.sh while
that long-running shell was reading it, shifting its file offsets. The complete
attempt, including exit-code.txt, is preserved in initial-build-run/. The final
script is rerun unchanged with the same source snapshots and compiled objects;
root build logs describe the incremental relink, and initial-build-run/ retains
the original full build logs. No test, tolerance or candidate patch changed.

First smoke (env/build/test; CPU launch nsm=8, max_wg_per_cu=1024):
`first smoke exit code=0`.

| variant | op | rc | pass | fail | unsupported |
|---|---|---:|---:|---:|---:|
| upstream | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| upstream | FA-BIASV | 0 | 16 | 0 | 0 |
| fp32f | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| fp32f | FA-BIASV | 0 | 16 | 0 | 0 |
| cap | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| cap | FA-BIASV | 0 | 16 | 0 | 0 |

Second smoke (test only, same WORK; nsm=32, max_wg_per_cu=2):
`second smoke exit code=0`. Upstream's underlying test process exits 1; the
harness records its four expected nb=512 biased-V failures and still exits 0.
Both variants' main FA rows exclude every v_bias case. Cap passes all 16 biased
cases. Both final runs match the ticket's expected results.

| variant | op | rc | pass | fail | unsupported |
|---|---|---:|---:|---:|---:|
| upstream | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| upstream | FA-BIASV | 1 | 12 | 4 | 0 |
| cap | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| cap | FA-BIASV | 0 | 16 | 0 | 0 |

The first forced-geometry run also exited 0 with the same counts; it is retained
in forced-geometry-before-format/. A newline-only failure-list fix was followed
by the final test-only rerun above (exit 0); the four failures now occupy four
separate lines in upstream.fail-FA-BIASV.txt. No correctness counts or policy
changed in that formatting fix.

kit-audit.log verifies 20 unique cases per variant (16 biased + 4 existing),
clean main rows, exact biased shape coverage, both effective launch geometries,
expected failures confined to upstream's four nb=512 cases, and exact agreement
between each correctness table and its raw case records. It also verifies source
diff identity between the initial fresh-WORK build and the final rerun, the
byte-identical P9 perf block, and the verbatim P11 acceptance quote. Audit exit 0.


Not verified: A770 device identity/geometry, A770 correctness, performance or
XMX behavior, GPU AOT builds, Vulkan execution, full FLASH_ATTN_EXT suite,
CPY/GET_ROWS, conversion microbenchmark, kernel perf or model/llama-bench runs.
The new 32K perf corpus and tg tag are wired into the kit but have no timing
results here. No shipping/acceptance choice is made.

Artifacts: summary.md/correctness.md, per-variant preflight/test/failure logs,
manifest.txt, env-vars.txt, env.txt, full sycl-ls output, geometry query logs,
build logs, scratch-tree diffs and console/run logs. forced-geometry/ retains
its own table, raw process exit codes and effective geometry in preflight logs;
env-only/ contains the independent env dry check. Large logs/diffs are gzipped.

## Manager verification (Claude Opus 5.5, 2026-10-11)

Read the run.sh diff in full. The correctness parser matches test-backend-ops' own counting: summary totals exclude NOT_SUPPORTED/SKIPPED (`tests/test-backend-ops.cpp:12680`), and it classifies `v_bias=2` case lines into FA-BIASV. The forced-geometry smoke (`upstream` 12/16 FA-BIASV, exit 0; `cap` 16/16) shows the expected-failure gating works. Outside the ticket's list but kept: the env step now runs a scratch copy of `bench/phase0.sh` with OUT redirected into this run's OUT, instead of importing a separate phase0 results tree. Files over 40 KB were gzipped at commit (references above name the uncompressed files).
