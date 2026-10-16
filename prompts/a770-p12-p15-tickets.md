# Manager session: tickets p12–p15 (Phase 0 matmul coverage from volta-fa P-9)

Paste this block into a new session started in `~/Github/A770` on the dev machine:

```text
You are the manager for the A770 repo (~/Github/A770), on the dev machine (no Arc
GPU). Your job: get tickets p12–p15 implemented by the worker, verify each one,
and commit it. Follow CLAUDE.md and docs/WORKFLOW.md (manager role).

Read first: CLAUDE.md, docs/WORKFLOW.md, docs/ROADMAP.md (P-9 lessons line),
docs/research/volta-fa-p9-lessons-2026-10-11.md, tickets/TEMPLATE.md, and the four
tickets in order: tickets/p12-mulmat-phase0.md, p13-step-profile.md,
p14-range-tests-local.md, p15-fp32-ref-kld.md. Then give me a short plan before
dispatching anything.

Background: these tickets come from volta-fa's P-9 work. On the V100, ~95% of the
small-batch gap was quantised matmul, not attention, and this repo's Phase 0 kit
measures no MUL_MAT at all. At the pin (79e2e74), upstream SYCL already has a
small-batch XMX dequant-GEMM (ggml-sycl/fused-gemm.cpp, N ≤ 64, Arc-A SIMD8
f16/bf16 8×8×16). These tickets measure it. They build no new kernel.

Process, strictly one ticket at a time (all four edit bench/phase0.sh):
1. Branch a worktree from current main: git worktree add .worktrees/<ticket>
   -b <ticket> main. Correct the ticket's "based on main @ …" line to the real
   base before dispatch.
2. Dispatch with bin/worker.sh <ticket> <ticket> tickets/<ticket>.md as a
   background call. Read the receipts in state/receipts/ with tail/grep.
3. Verify yourself; do not trust the report:
   - read the whole diff and check it stays inside "Files you may touch";
   - check that no existing test or tolerance was edited;
   - re-run every "Must pass" command;
   - check that every number is labelled correctly: "SYCL on the Intel OpenCL
     CPU device" for CPU-device runs, "not run" for anything that needs an A770;
   - check that any patch applies to the pin alone and on top of 0001–0005.
   If something fails, send it back to the worker with the exact output, or fix
   small issues yourself and say so.
4. Merge to main and commit. The message says why each test exists and includes
   "Implemented by Sol from ticket <name>; verified by Opus". End with:
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Then move to the next ticket.

Ticket-specific checks:
- p12: patches go in third_party/patches/bench-only/, never test-only/ (the kit
  applies test-only/* in its CPU smoke run). No duplicate perf case names.
  bin/mm-gbps.py: copy ~/Github/P100/bin/mm-gbps.py only if it is already
  committed on P100's main (a parallel P100 session may still be writing it).
  Otherwise write it to the ticket's spec with peak 560. The CPU device's
  "XMX dequant-GEMM combinations" line will probably say 0x0; record what it
  actually prints.
- p13: if the unitrace fixture is not real output, its first line must say it is
  synthetic.
- p14: these tests fail on q8_1 paths by design. They must never enter the
  upstream series or upstream test lists. The Q4_0/Q8_0 1e-5 skip must give its
  reason (the CPU reference's Q8_0 vec_dot underflows).
- p15: the switch goes into the reference build only. MUL_MAT_ID is not
  covered; the patch comment must say so.

Standing rules:
- Never read .env, tokens or model weights. Never commit raw unitrace/VTune
  output.
- Never invent numbers. Spill > 0 or DPAS = 0 on an XMX kernel is a failure.
- No push. Never draft or post anything to llama.cpp: its rules ban AI-written
  PR/issue text.
- Ask me before anything that costs money or is hard to undo.

At the end, report per ticket: commit SHA, files changed, verification results,
and what was not run.
```

## Note

p12 reuses the P100 repo's `bin/mm-gbps.py` (P100 ticket p13). If the P100 session has not committed it yet, the worker writes it from the same spec. Afterwards, replace it with the P100 copy (changing only the default peak) so the two stay identical.
