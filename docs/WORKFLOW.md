# Multi-model workflow (adapted from opus-sol via volta-fa)

Source: `/home/p51/Github/opus-sol/PLAYBOOK.md`, as adapted in `~/Github/volta-fa/docs/WORKFLOW.md`. Same shape (manager writes narrow tickets, worker implements, manager verifies and owns git), for GPU work where **the profiler is the judge**, not either model.

## Roles

| Manager (Opus 5.5) | Worker (GPT-6.1 Sol via Codex) |
| --- | --- |
| Phase decisions, kernel design (tiling, SLM layout, GRF budget, sub-group size), ticket writing | Scaffolding: CMake, test harness, bench drivers, multi-GPU boilerplate, llama.cpp patch plumbing |
| Interprets VTune/unitrace output, decides next optimisation | First-draft kernels from a precise design ticket; mechanical refactors; rebases |
| Verifies on hardware, reads critical diffs, owns git | Stops and explains when a rule blocks it |
| Reviews kernels, anything touching numerics or invariants | Routine read-only reviews (`bin/worker-review.sh`) |

The pasted plan suggested Opus writes all kernels and Sol only scaffolds. Keep that as the default for the **hot** kernels, but let Sol draft kernels from a manager-written design ticket when the design is already decided; Opus then does the profile-driven rewrites.

## The hardware loop (replaces "run the tests" for kernel work)

1. Manager writes ticket from `tickets/TEMPLATE.md` including the sub-group size, GRF mode, SLM budget and target occupancy.
2. `bin/worker.sh <name> <worktree> <ticket>` (background, stdin closed).
3. Manager runs `bin/xe-report.py` (works without a GPU); reads SIMD, GRF, threads/XVE, spill, SLM, DPAS. Spill > 0, or DPAS = 0 on an XMX kernel → back to worker/self with the numbers.
4. `ctest` for parity. Failing parity blocks everything else.
5. VTune GPU hotspots (docs/PROFILING.md) → text summary in `profiles/<kernel>-<date>.txt`.
6. Feed **only the summary** (key metrics + flagged source lines) back into the next ticket using `prompts/profile-feedback.md`.
7. Repeat until the exit criterion for the phase is met or the gap is shown not to exist.
8. Commit with the profile summary referenced in the message.

## Review tiers

| Change | Review |
| --- | --- |
| CMake, docs, bench drivers, test plumbing | Worker read-only + CI |
| Any kernel, numerics, masking, softmax | Manager-spawned independent reviewer (fresh context), told the invariants to check |
| Multi-GPU lifetime/queues/events | Independent reviewer (races hide here) |
| Performance claims | Manager re-measures; single runs are not evidence (3+ interleaved rounds, medians) |

## Known failure modes to put in every kernel ticket

- **Xe2 bias**: models reach for SIMD16 DPAS, 8×16×16 `joint_matrix`, 2D block loads and Battlemage/PVC examples (sycl-tla, XeTLA). On acm-g10 the OpenCL SIMD16 DPAS builtin compiles **to nothing, without an error**; `bin/xe-report.py` (DPAS count) and `bin/check-forbidden.sh` catch it. Note the inverse trap: BF16 *is* legal on Xe-HPG XMX, so don't let a reviewer "fix" it away.
- **CUDA bias**: porting llama.cpp's CUDA FA wholesale. Ideas transfer (volta-fa's dispatch lessons); warp-level code does not (XMX code runs at sub-group 8, CUDA warps are 32, so shuffles and fragment layouts must be redone).
- **Register spills / SIMD drops** while the code looks perfect: IGC retries at a narrower SIMD after spilling unless the sub-group size is pinned. Read xe-report first.
- **SLM bank conflicts / scattered loads**: give the model the VTune-flagged lines, not a guess.
- **Invented numbers**: workers must write "not run" when no A770 was available. A Battlemage or Lunar Lake number is not an A770 number.
- **Tests edited to pass**: manager reads every edit to an existing test.

## Mechanics carried over from opus-sol

- Stdin must be closed for `codex exec` (`< /dev/null`).
- Do not pass `--add-dir`; worker cannot write outside its worktree.
- Worker cannot commit; manager commits after verifying.
- Worktrees live in `.worktrees/` (git-excluded). Tickets repeat the rules block verbatim; workers do not read CLAUDE.md on their own.
- Receipts land in `state/receipts/` (gitignored); read them with `tail`/`grep`, not whole.

## Setup note

`bin/env.sh` holds machine-specific paths. `WORKER_MODEL` is copied from opus-sol (`gpt-6.1-sol`). Worktrees: `git worktree add .worktrees/<name> -b <branch> main`. The worker builds into its own `build/` inside the worktree using the main checkout's `.venv` by absolute path; tickets spell this out.
