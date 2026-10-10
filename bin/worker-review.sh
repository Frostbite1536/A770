#!/bin/bash
# worker-review.sh <name> <worktree-name|/abs/path> "<claims to verify + focus>"
# Cheap read-only review by the worker model. Kernel/numerics/P2P changes need an independent manager-spawned reviewer instead.
# shellcheck source=bin/env.sh
. "$(dirname "$0")/env.sh"
N=$1; FOCUS=$3; T=$A770_RECEIPTS/$N.ticket.md
case $2 in /*) WT=$2 ;; *) WT=$A770_WT/$2 ;; esac
cat > "$T" <<EOT
You are an independent code reviewer. READ ONLY: do not edit files, no git writes, no network, never read .env or tokens.
Review the uncommitted change in $WT: run \`git -C $WT status\` and \`git -C $WT diff\`, and read every untracked file. Read docs/INVARIANTS.md first.
Target is Intel Arc A770 (Xe-HPG, acm-g10) only: XMX via DPAS at sub-group size 8 (SYCL joint_matrix 8x8x16 f16/bf16, OpenCL SIMD8 matrix_mad) is legal (ESIMD xmx::dpas is untested here); SIMD16 DPAS, 2D block loads, TF32, FP8 and FP64 are not. Check bin/xe-report.py output for spill, SLM and a DPAS count > 0 on XMX kernels.
$FOCUS
Report findings ranked BLOCKING / MEDIUM / LOW with file:line and a concrete failure scenario. Say plainly if nothing blocks. At most 450 words.
EOT
"$(dirname "$0")/worker.sh" "$N" "$WT" "$T" read-only
