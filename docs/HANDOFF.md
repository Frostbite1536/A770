# Handoff: first session on the A770 box

Written 2026-10-16 at the end of the dev-machine session that merged tickets p12–p15.
The next session runs on a machine with an Arc A770. **Nothing in this repo has run on an A770 yet**; every timing, correctness count and path claim below is either "SYCL on the Intel OpenCL CPU device" or **not run**.

## Where things stand

| Work | State | Where |
|---|---|---|
| P-6 padded GQA, P-8 KV conversion, P-11 FP16 accumulation (patches 0001–0005, fp32f/cap) | Designed, CPU-tested, "Accept if" fixed before any run | `docs/design/`, `bench/patches/run.sh` |
| P-10 large-GRF TILE draft | Static only; opt-in `grf` variant | `third_party/patches/p10-draft-*` |
| p12 MUL_MAT correctness + 1–16-token timing (`mmperf`, default step) | Merged `5613b45` | `bench/phase0.sh`, `bin/mm-gbps.py`, bench-only 0001 |
| p13 per-kernel step breakdown (`prof`, opt-in) | Merged `e663f02` | `bench/phase0.sh`, `bin/step-breakdown.py` |
| p14 activation-range MUL_MAT tests | Merged `32fcda0` | bench-only 0002, phase0 `test` step |
| p15 FP32-reference KL divergence (`kld`, opt-in) | Merged `c410411` | bench-only 0003, phase0 `kld` step |

The runbook is still [A770_BOX.md](A770_BOX.md) §0 and the guided prompt is [prompts/a770-session-1.md](../prompts/a770-session-1.md). This file adds what p12–p15 changed and what to look at first.

## Getting the box ready

The repo, the llama.cpp clone, `.venv/` and `tools/` are not all in git. On the box:

```bash
git clone https://github.com/Frostbite1536/A770.git && cd A770
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout --detach "$(cat bench/LLAMA_COMMIT)"   # 79e2e74eb
bin/setup-dev.sh            # .venv (cmake, ninja, numpy) and tools/neo; no WITH_DPCPP, the box uses system oneAPI 2026.1
```

`A770_BOX.md` hard-codes `/home/p51/Github/A770`; substitute the box's path. Install the drivers and toolchain from [TOOLCHAIN.md, A770 box](TOOLCHAIN.md#a770-box-when-available) first; the harness installs nothing. Copy `bench/models.example.txt` to `bench/models.txt` with the box's model paths (gitignored).

## What p12–p15 changed in Phase 0

- **The `build` step now edits `third_party/llama.cpp`'s working tree.** It applies bench-only 0001 (280 extra MUL_MAT perf cases) and 0002 (96 activation-range cases) on every build, idempotently. `bench/patches/run.sh` clones from the commit, not the working tree, so its variants are unaffected. Do not `git checkout .` that clone between steps without knowing this.
- **`test`** adds MUL_MAT correctness per backend, excluding the range cases (`mm-test-<be>.txt`), and the range cases separately per arm (`range-<arm>.txt`: SYCL0 defaults, nofg, noopt; Vulkan0). The range cases **fail by design** on q8_1 paths (SYCL's q8_1 stores its sum in FP16). Record them; they are evidence, not regressions. CPU-device reference: 48/96 in every SYCL arm; pattern in `third_party/patches/bench-only/README.md`.
- **`mmperf`** (default) runs 320 MUL_MAT perf cases (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K × four matrix shapes × n = 1..16) over `ROUNDS=3` interleaved rounds of four arms: `sycl`, `sycl-noopt` (`GGML_SYCL_ENABLE_OPT=0`, no weight reorder), `sycl-nofg` (`GGML_SYCL_XMX_GATHER_TYPES=0`, no fused XMX dequant-GEMM), `vulkan`. Output: `mm-perf-<arm>.r<k>.sql`, `mmperf.md` (GB/s against a 560 GB/s peak), `sycl-env.txt`.
- **`prof`** (opt-in, needs `unitrace` on PATH and `bench/models.txt`; uses the first model) captures `llama-bench -p 8|16 -n 0 -d 0|16384` at `-r 1` and `-r 1+PROF_REPS` and reports the difference per kernel, so warm-up and the depth prefill cancel. Copies are listed but excluded from the shares.
- **`kld`** (opt-in, needs `PPL_TEXT=/path/to/text` and `bench/models.txt`) builds a second SYCL tree with bench-only 0003 (`GGML_SYCL_F32_REF=1`: quantised MUL_MAT via dequantize + oneMKL SGEMM) as the reference, then measures KL divergence of the sycl / sycl-nofg / sycl-noopt arms at batch 16. One extra full SYCL build. MUL_MAT_ID (MoE) is **not** covered by the reference switch.

Suggested order, after the A770_BOX §0 toolchain checks and repo ctest:

```bash
STEPS="env build support test perf mmperf" bench/phase0.sh    # no models needed
bench/patches/run.sh                                          # FA patch A/B, unchanged
STEPS="build lb" MODELS="..." bench/patches/run.sh            # A770_BOX §0 step 4
STEPS=bench bench/phase0.sh                                   # llama-bench grid (models.txt)
STEPS=prof bench/phase0.sh                                    # if unitrace is installed
PPL_TEXT=/path/to/wiki.test.raw STEPS=kld bench/phase0.sh     # optional
```

## Check these first (they decide how to read everything else)

1. **Is the fused XMX dequant-GEMM active?** Look in `sycl-env.txt` for the `XMX dequant-GEMM combinations` line. On the CPU device it was **not printed at all**. If it is missing or 0x0 on the A770 too, the `sycl-nofg` arm measures nothing different from `sycl`; stop and find out why before reading `mmperf.md`.
2. **Is weight reorder active?** Read the `Reorder` column of the SYCL device table in the start-up log (`sycl-env.txt`). On the CPU device it was `N`, so the CPU runs never exercised the reordered path; if it is also `N` on the A770, `sycl-noopt` measures nothing different from `sycl`.
3. **Range-test pattern on hardware.** CPU device: Q4_0 at 1e5 fails at every n; Q8_0 at 1e5 at n = 1, 16, 32; K-quants at 1e5 at n = 16, 32; K-quants at 1e-5 at n = 2, 4, 8 only (not interpreted). Compare per case; a different pattern on the A770 tells you which kernels run at each n.
4. **Real unitrace output vs the synthetic fixture.** `tests/fixtures/step-breakdown/*` is synthetic (first line says so). If `bin/step-breakdown.py` fails to parse or leaves most time "uncategorised", keep the raw capture local and open a ticket to fix the parser against the real format; don't edit it mid-measurement.
5. **SYCL vs Vulkan at n = 1..16** in `mmperf.md`: per INV-PERF-002 the faster one is the baseline per shape. On the V100 the small-batch gap was ~95% matmul (docs/research/volta-fa-p9-lessons-2026-10-11.md); whether that holds here is the open question p12–p15 were built to answer.

## Not run (anywhere)

All A770 builds, tests and timings; Vulkan0; which MUL_MAT path each case takes; the fused XMX GEMM; unitrace and its real kernel names; any model run; any KL divergence number. The P-6/P-8/P-11 "Accept if" criteria are untested.

## Rules that still apply

- Label every number with where it ran; anything not measured on the A770 is "not run".
- Keep raw logs, unitrace/VTune output, `.kld` files and models out of git. Commit only curated summaries, reviewed separately.
- The owner makes acceptance and shipping decisions. Nothing goes to llama.cpp without the owner.

## Paste-in prompt for the next session

```text
We are on the A770 box. Read CLAUDE.md, docs/HANDOFF.md, docs/A770_BOX.md and
prompts/a770-session-1.md, then follow the session-1 checklist with the Phase 0
additions in HANDOFF.md (mmperf, range tests, prof, kld). Check HANDOFF.md's
"Check these first" items before interpreting any table. Report findings; do
not commit, push or decide acceptance without me.
```
