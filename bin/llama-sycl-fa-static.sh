#!/bin/bash
# Static IGC reports for llama.cpp's TILE and VEC FlashAttention TUs on Arc A770 only.
# Configure once, then compile one TU at a time; no llama.cpp build or runtime measurements.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export A770_TOOLS=${A770_TOOLS:-$ROOT/tools}
O=$A770_TOOLS/oneapi-2026.1
LLAMA=$ROOT/third_party/llama.cpp
CFG=$LLAMA/build-sycl-aot-cfg
OUTDIR=${1:-$ROOT/bench/results/llama-sycl-fa-static-$(date +%Y%m%d)}
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)

if [ ! -f "$CFG/compile_commands.json" ]; then
    cmake -S "$LLAMA" -B "$CFG" \
        -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF \
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER="$O/bin/icx" -DCMAKE_CXX_COMPILER="$O/bin/icpx" \
        -DCMAKE_PREFIX_PATH="$O" > "$OUTDIR/configure.txt" 2>&1 || {
            cat "$OUTDIR/configure.txt" >&2
            exit 2
        }
fi

{
    echo "Static IGC report: device acm-g10 (Intel Arc A770); no runtime measurements."
    echo "llama.cpp commit: $(git -C "$LLAMA" rev-parse HEAD)"
    echo "Flags: compile database -I/-isystem/-D/-std=; xe-report AOT -O2, default GRF."
    echo "Spill/private/SLM values in TU reports are bytes reported by IGC."
    echo "TU | kernels | max GRF | any SPILL | SIMD widths | min threads/XVE | compile wall seconds | xe-report rc"
} > "$OUTDIR/summary.txt"

failed=0
for tu in fattn-tile-instance-dkq64-dv64 fattn-tile-instance-dkq128-dv128 \
          fattn-tile-instance-dkq256-dv256 fattn-vec-instance-f16-f16 fattn-vec-instance-q8_0-q8_0; do
    echo "[llama-sycl-fa-static] $tu"
    start=$(python -c 'import time; print(time.monotonic())')
    rc=0
    python "$ROOT/bin/xe-report.py" "$LLAMA/ggml/src/ggml-sycl/template-instances/$tu.cpp" \
        --compile-commands "$CFG/compile_commands.json" --wide > "$OUTDIR/$tu.txt" 2>&1 || rc=$?
    elapsed=$(python -c 'import sys, time; print(f"{time.monotonic() - float(sys.argv[1]):.2f}")' "$start")
    # rc=1 reports resource violations (including upstream spills); rc>=2 means compile/report failure.
    if [ "$rc" -gt 1 ]; then
        failed=1
    fi
    if ! python - "$OUTDIR/$tu.txt" "$tu" "$elapsed" "$rc" >> "$OUTDIR/summary.txt" <<'PY'
import pathlib
import re
import sys

report, tu, elapsed, rc = sys.argv[1:]
rows = []
for line in pathlib.Path(report).read_text().splitlines():
    match = re.match(r"^\s+.+?\s+" + r"\s+".join([r"(\d+)"] * 8) + r"(?:\s+\D.*)?\s*$", line)
    if match:
        rows.append(tuple(map(int, match.groups())))
if rows:
    widths = ",".join(map(str, sorted({row[0] for row in rows})))
    print(f"{tu} | {len(rows)} | {max(row[1] for row in rows)} | "
          f"{'yes' if any(row[3] for row in rows) else 'no'} | {widths} | "
          f"{min(row[2] for row in rows)} | {elapsed} | {rc}")
else:
    print(f"{tu} | FAILED (no kernel rows) | wall seconds {elapsed} | xe-report rc {rc}")
    sys.exit(2)
PY
    then
        failed=1
    fi
    tail -n 1 "$OUTDIR/summary.txt"
done
cat "$OUTDIR/summary.txt"
exit "$failed"
