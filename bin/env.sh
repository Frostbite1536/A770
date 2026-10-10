#!/bin/bash
# Sourced by every script. Override per machine.
A770_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export A770_HOME
export A770_RECEIPTS="$A770_HOME/state/receipts"
export A770_WT="$A770_HOME/.worktrees"
export A770_PROFILES="$A770_HOME/profiles"
export A770_PY="$A770_HOME/.venv/bin/python"
# Verify before first use: `codex exec -m <id>` must be a model your CLI accepts.
export WORKER_MODEL="${WORKER_MODEL:-gpt-6.1-sol}"
export WORKER_EFFORT="${WORKER_EFFORT:-high}"
