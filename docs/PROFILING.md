# Profiling loop

Run on the A770 box. Commit **text summaries** to `profiles/`; raw result directories stay local (`profiles/raw/`, gitignored).

## System view

```bash
# unitrace flags differ between releases; check `unitrace --help` on the box.
unitrace --device-timing --kernel-submission ./third_party/llama.cpp/build-sycl/bin/llama-bench -m model.gguf -fa 1 -p 512 -n 0
intel_gpu_top          # engine busy while a run is going (i915)
```

Look for: time split between FA and matmul kernels, gaps between kernels (host submission, oneDNN primitive creation), K/V conversion kernels before FA.

## Step breakdown

Run `STEPS=prof bench/phase0.sh` on the A770 with `unitrace` on PATH and the first model in `bench/models.txt`.
For each backend, p=8/16 and depth=0/16384, it captures `-r 1` and `-r (1 + PROF_REPS)` (default `PROF_REPS=4`).
Both retain default warm-up; `(rN − r1) / PROF_REPS` cancels shared setup, warm-up and one-off depth prefill.
Help and paired `*-r1.txt`/`*-rN.txt` outputs go in `$OUT/prof/`; `bin/step-breakdown.py --reps "$PROF_REPS"`
generates `breakdown.md` with per-step calls, µs and category shares of summed device time, not wall-clock latency.
Transfers, including per-repetition KV state restore at depth > 0, are reported but excluded from totals and shares.
Nonpositive time deltas are marked `(noise)` and count as 0; negative call deltas and missing partners fail.
It skips if the model list or unitrace is missing and fails if unitrace help lacks `--device-timing`.
Check patterns and summary format on the box; both test captures are synthetic. Keep raw binary results uncommitted.

## Kernel view (VTune)

```bash
vtune -collect gpu-hotspots -r profiles/raw/fa-<shape> -- ./third_party/llama.cpp/build-sycl/bin/test-backend-ops perf -o FLASH_ATTN_EXT -b SYCL0
vtune -report summary -r profiles/raw/fa-<shape> > profiles/fa-<shape>-<date>.txt
```

Metric names differ between VTune versions; check with `vtune -report hw-events` on the box and record the version. What to read:

| Question | Look at |
| --- | --- |
| XMX busy? | XMX/systolic pipe utilisation (if exposed), and the DPAS count from `bin/xe-report.py` to confirm DPAS is there at all |
| Memory-bound? | GPU memory read/write bandwidth vs the measured streaming bandwidth (not the datasheet) |
| Occupancy limiter | XVE threads occupancy; GRF mode and SLM per work-group from `bin/xe-report.py` |
| Stalls | XVE array active / stalled / idle split |
| Spills | `bin/xe-report.py` (static); at runtime, scratch traffic |

## Feeding results back to a model

Paste only: kernel name, global/local range, sub-group size, GRF mode, spill bytes, SLM/work-group, threads occupancy, XVE active/stalled %, memory bandwidth, DPAS count. Use `prompts/profile-feedback.md`. Do not paste whole reports.

## Rules

- Same input, same power limit, warm-up runs discarded. Record GPU frequency during the run (`intel_gpu_top`).
- 3+ interleaved rounds, report medians and spread.
- Record GPU model (8 / 16 GB), driver/IGC, oneAPI, kernel driver, ReBAR, commit and baseline commit with every summary.
