# A770 patch-series A/B

Base: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6. CPU_SMOKE=1, FORCE=1.
CPU smoke proves plumbing only. A770 timing: not run.
Read manifest.txt, env.txt (if requested), env-vars.txt and round GPU samples with these results.
INV-PERF-002: use the faster upstream SYCL/Vulkan result for each shape; missing/unsupported rows have no established baseline.

## SYCL0 launch geometry

```text
SYCL0 device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz
SYCL0 max_work_group_size=8192
SYCL0 max_compute_units=8
SYCL0 max_wg_per_cu=1024
GGML_SYCL_TEST_NSM=unset
GGML_SYCL_TEST_MAX_WG_PER_CU=unset
```
max_wg_per_cu is the integer quotient max_work_group_size / max_compute_units used by launch_fattn; CPU test overrides are recorded separately.

## Correctness

not run
FA-BIASV failures are expected evidence for other variants; only fp32f/cap failures affect the exit status (P-11 Accept if 2).

Per-case failures: *.fail-*.txt; full output: *.test-*.txt. Known upstream CPU flake:
`CPY(type_src=f16,type_dst=q4_0,ne_src=[256,4,4,4],...)`; counted, never suppressed.

## Conversion microbenchmark

not run

## Kernel medians and ranges

not run

## llama-bench

not run
