# A770 patch-series A/B

Base: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6. CPU_SMOKE=1, FORCE=1.
CPU smoke proves plumbing only. A770 timing: not run.
Read manifest.txt, env.txt (if requested), env-vars.txt and round GPU samples with these results.
INV-PERF-002: use the faster upstream SYCL/Vulkan result for each shape; missing/unsupported rows have no established baseline.

## SYCL0 launch geometry

not run
max_wg_per_cu is the integer quotient max_work_group_size / max_compute_units used by launch_fattn; CPU test overrides are recorded separately.

## Correctness

| variant | op | rc | pass | fail | unsupported |
|---|---|---:|---:|---:|---:|
| upstream | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| upstream | FA-BIASV | 1 | 12 | 4 | 0 |
| cap | FLASH_ATTN_EXT | 0 | 4 | 0 | 0 |
| cap | FA-BIASV | 0 | 16 | 0 | 0 |
FA-BIASV failures are expected evidence for other variants; only fp32f/cap failures affect the exit status (P-11 Accept if 2).

Per-case failures: *.fail-*.txt; full output: *.test-*.txt. Known upstream CPU flake:
`CPY(type_src=f16,type_dst=q4_0,ne_src=[256,4,4,4],...)`; counted, never suppressed.

## Conversion microbenchmark

not run

## Kernel medians and ranges

not run

## llama-bench

not run
