# Guided A770 session 1: patch-series A/B

Paste this block into a new session in the A770 repo on the box:

```text
Guide the owner through the first Intel Arc A770 (Xe-HPG, acm-g10) measurement
session. Run the prepared kit, preserve results, compare against the predeclared
criteria and report findings. Do not decide whether to ship.

Read first: CLAUDE.md, REVIEW.md, docs/INVARIANTS.md, docs/TOOLCHAIN.md (A770 box),
docs/A770_BOX.md (§0 command order), bench/phase0.sh, bench/patches/run.sh,
bench/models.example.txt, docs/design/P6-sycl-padded-gqa.md,
docs/design/P8-sycl-kv-conversion.md, docs/design/P11-sycl-fa-fp16-accumulation.md
and prompts/a770-session-1.md (this file).
Also read the pinned llama.cpp docs/backend/SYCL.md for selectors and watchdogs.
Then give a short plan and follow the checklist and rules below.
```

## Checklist

1. Complete the toolchain/BIOS/driver checklist in `docs/A770_BOX.md` §0–1.
   Confirm actual A770 identity, board vendor, VRAM, ReBAR, kernel driver,
   oneAPI 2026.1, runtime/IGC and Vulkan versions. Start tmux. No installation or
   network is part of the harness. Check the pinned llama.cpp commit first.
2. Run repo build/ctest, then Phase 0 env/build/support/test/perf for SYCL and
   Vulkan, in the exact command order in the runbook. Save the ctest summary.
3. Run `bench/patches/run.sh` without models: phase0 env capture, bit-exact
   conversion microbenchmark, upstream/p6/conv/conv8/fp32f/cap builds,
   FLASH_ATTN_EXT correctness plus CPY/GET_ROWS for conv/conv8, and a separate
   FA-BIASV row for the 16 biased-V cases (excluded from the main FA counts).
   Only fp32f/cap biased-V failures affect exit status; other variants' failures
   are expected evidence under P-11 criterion 2. Record SYCL0 max_work_group_size,
   max_compute_units and max_wg_per_cu from the env step in summary.md.
   Run three interleaved kernel rounds and Vulkan reference. Inspect `run.log`
   during long builds; after a timeout rerun incremental builds, never kill an unrelated process.
   Inspect counts, unsupported cases, failures and GPU samples before interpreting
   tables. Archive and copy this first result home immediately.
4. Run `STEPS="build lb" MODELS="..." bench/patches/run.sh` with the four paths
   from `bench/models.example.txt` (exact example command in the runbook):
   Llama-3.1-8B Q8_0 (GQA 4 control), Qwen2.5-7B Q8_0 (GQA 7),
   Qwen2.5-14B Q4_K_M (GQA 5), gemma-2-9b Q6_K (D=256, GQA 2, softcap).
   Record model/runtime failures, including VRAM limits, without removing rows.
   Small-batch: pp1/2/4/8/16 at 4K/16K; quantised-KV: q8_0 pp4/8 + tg32 at 16K.
   Decode tag `tg`: `-p 0 -n 32 -d '4096,16384,32768'` (F16 KV).
   Compare upstream/p6/conv8/fp32f/cap/Vulkan over two interleaved rounds.
5. Cross-check upstream vs p6 with `bin/llama-sycl-fa-static.sh` on the scratch
   clones (runbook commands). Separate static resource reports from runtime.
   Optional P-10: select `grf` in VARIANTS for `STEPS="build perf"`; compare the
   upstream-only GRF draft against upstream. It is not cumulative with P-6/P-8.
6. Copy model-run and static results home; verify archive contents. Analyse
   `summary.md` and re-cut tables with `bench/analyze.py tbo` / `lb` as needed.
   Each variant's rounds are one comma-joined input group, with matching labels.
   Report medians, ranges, support gaps and GPU state; evaluate both SYCL and
   Vulkan references and choose the faster baseline for each shape.
7. Give the owner a findings draft: hardware/toolchain, commands and artifacts,
   correctness including per-case failures, each acceptance criterion and evidence,
   unexplained variance, missing measurements and questions requiring decisions.
   Only separately reviewed, curated reports/artifacts belong in git; no raw logs
   or models. No automatic commit, push, upstream message or ship decision.

## Acceptance criteria (quote before measuring)

From `docs/design/P6-sycl-padded-gqa.md`, “Accept if (A770,
`bench/patches/run.sh`, variant `p6` vs `upstream`)”:

> 1. **Correctness**: `test-backend-ops test -o FLASH_ATTN_EXT` on SYCL0 has no new failures; the new GQA 3/5/6/7 block passes.
> 2. **Kernel**: on the new block's G 5/6/7 cases at 1–4 tokens (where padding applies: tokens × 8 ≤ 32), and G 3 at 1–8, the median over ≥ 3 interleaved rounds is ≥ 1.3× faster. Cases beyond the threshold must be unchanged (±3%). No existing case (G 1/2/4/8) is more than 3% slower.
> 3. **End to end**: `llama-bench -fa on -p 1,4 -d 16384` on Qwen2.5-7B (G 7) improves ≥ 10%, and `-p 8` (not padded) is unchanged (±3%). Llama-3.1-8B (G 4, never padded) changes by less than ±3%.
> 4. **Static**: no new spills in the TILE instances (`bin/llama-sycl-fa-static.sh`, patched vs upstream).

The P-6 doc records existing upstream spills and growth in some patched instances.
Report those changes with criterion 2; do not silently reinterpret criterion 4.

From `docs/design/P8-sycl-kv-conversion.md`, “Accept if” (manager-defined
2026-10-10, before any A770 run). Quote that section verbatim to the owner and
confirm it before measuring; the owner may revise it then, never after a run.

From `docs/design/P11-sycl-fa-fp16-accumulation.md`, “Accept if (A770, `bench/patches/run.sh`; manager-defined 2026-10-11, before any A770 run)”:

> 1. **Geometry**: record the A770's `max_work_group_size` and `max_compute_units` (env step). If `max_wg_per_cu` is not 2, recompute keys per accumulator for the decode shapes before reading 2–4.
> 2. **Correctness**: `fp32f` and `cap` pass the 16 biased-V cases on the A770 and show no new FLASH_ATTN_EXT failures vs `conv8`. `upstream`/`conv8` biased-V results are recorded as evidence (failures expected at ≥ 4096 keys per accumulator), not counted as regressions.
> 3. **Kernel**: at the decode shapes (nb 1–4, G 4, kv 4096/16384/32768, F16 and q8_0), the candidate's median over ≥ 3 interleaved rounds is no more than 3% slower than `conv8`.
> 4. **End to end**: Llama-3.1-8B tg32 at depth 16384 and 32768 is no more than 3% slower than `conv8`, and `-p 1..16` at depth 4096/16384 likewise.
>
> Choice: take F if it meets 2–4. Otherwise take S if it meets 2–4. Otherwise ship neither and report the costs. A candidate that is faster than `conv8` is fine. Thresholds are fixed before measurement; the owner may revise them before the session, not after.

Background only, quoted from volta-fa `docs/design/P8-quantised-kv-decode.md`
“Accept if” (different architecture, variants and workload; not A770 acceptance):

> - 0 new failures;
> - `qkv` ≥ 2× `p6cfg` at 1 token, 32K, GQA 4 (q8_0 and q4_0);
> - `conv8` ≥ 1.5× `qkv` at 8 tokens, 32K (quantised);
> - Llama-3.1-8B q8_0 tg64 at 16K ≥ +25% over upstream (the model's pessimistic VEC bound gives +38%);
> - no f16 case outside noise.

## Rules

- INV-PERF-001: every claimed speedup needs a reviewed artifact with board/VRAM,
  driver/IGC, oneAPI, kernel driver, ReBAR, tested commit and baseline commit.
- INV-PERF-002: the baseline per shape is the faster upstream SYCL or Vulkan
  backend at the same pin and default FA dispatch. Beating only the slower one
  is insufficient; missing reference data means the baseline is not established.
- Never invent/extrapolate numbers. Anything unexecuted is **not run**. CPU
  smoke proves plumbing/correctness only; every A770 timing remains **not run**
  until actually measured. Static compile durations are not GPU timings.
- Preserve per-case failures and crashes; stop interpretation and isolate the
  failing case with `test-backend-ops test -b SYCL0 -o <op> -p '<params>'`.
  Rerun on upstream and the affected variant. The pre-existing CPU flake
  `CPY(type_src=f16,type_dst=q4_0,ne_src=[256,4,4,4],...)` fails upstream about
  one in three CPU runs: report every observed failure; do not suppress it.
- Never change tolerances/tests to pass or move acceptance thresholds after a run.
  Treat overlapping ranges and clock/temperature drift as unresolved evidence.
- Keep oneDNN default ON and identical selector/runtime flags for comparisons.
  Label any watchdog cap or OpenCL diagnostic run separately; record dependent
  `ONEAPI_DEVICE_SELECTOR` / `GGML_SYCL_*` variables in the harness env-vars file.
- Test-only patches are applied solely by explicit `CPU_SMOKE=1 FORCE=1`, with
  env/build/test steps; production A770 variants never apply `test-only/*`.
- Never commit raw logs, models, scratch builds or profiler traces. Never read
  or commit credentials. Preserve local archives and curate evidence separately.
- No commit/push or posting to llama.cpp is authorized by this session prompt.
  The owner makes acceptance and shipping decisions.
