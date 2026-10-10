#!/bin/bash
# phase0.sh: Phase 0 baseline on the A770 box (docs/ROADMAP.md). Run from the repo root with oneAPI 2026.1 sourced.
#   bench/phase0.sh                        full run: env, build, FA support/correctness/perf, llama-bench grid
#   STEPS="env perf" bench/phase0.sh       run only some steps (env build support test perf bench)
#   BACKENDS=sycl bench/phase0.sh          one backend only (default: "sycl vulkan"; INV-PERF-002 needs both)
#   LLAMA=third_party/other-tree bench/phase0.sh   same kit against another llama.cpp tree (results tagged by tree name)
# Models: one GGUF path per line in bench/models.txt (see bench/models.example.txt).
# Output: bench/results/phase0-<tree>-<date>-<host>/ (commit the small text/sql/jsonl files, never models).
# Status: drafted on a machine without an Arc GPU; NOT RUN yet.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
LLAMA=${LLAMA:-third_party/llama.cpp}
STEPS=${STEPS:-"env build support test perf bench"}
BACKENDS=${BACKENDS:-"sycl vulkan"}
REPS=${REPS:-3}
TAG=${TAG:-$(basename "$LLAMA")}
OUT=bench/results/phase0-$TAG-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
uses() { [[ " $BACKENDS " == *" $1 "* ]]; }

# Refuse to produce "baseline" numbers on the wrong hardware or toolchain (INV-PERF-001, INV-ARCH-003).
if uses sycl; then
  command -v sycl-ls > /dev/null || { echo "sycl-ls not found: source /opt/intel/oneapi/setvars.sh" >&2; exit 2; }
  gpu=$(sycl-ls 2>/dev/null | grep -m1 'level_zero:gpu' || true)
  [[ "$gpu" == *A770* || "${FORCE:-0}" == 1 ]] || { echo "Level Zero GPU is '$gpu', not an A770 (FORCE=1 to override)" >&2; exit 2; }
  ver=$(icpx --version | sed -n 's/.*Compiler \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
  [ "$ver" = 2026.1 ] || [ "${FORCE:-0}" = 1 ] || { echo "icpx is '$ver'; oneAPI 2026.1 required (INV-ARCH-003). FORCE=1 to override" >&2; exit 2; }
fi

if has env; then
  log "env"
  # Subshell with set +e: a missing tool must not abort the whole run.
  ( set +e; echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "A770 repo: $(git rev-parse --short HEAD)"
    echo "llama.cpp tree: $LLAMA ($TAG) $(git -C "$LLAMA" rev-parse HEAD)"; uname -r
    echo; echo "## GPU / PCIe / kernel driver"; lspci -nnk | grep -A3 -iE 'vga|display'
    for d in /sys/bus/pci/devices/*; do
      [ "$(cat "$d/vendor" 2>/dev/null)" = 0x8086 ] && [ -e "$d/current_link_speed" ] && [ "$(cat "$d/class")" = 0x030000 ] &&
        echo "$d: link $(cat "$d/current_link_speed") x$(cat "$d/current_link_width") (max $(cat "$d/max_link_speed") x$(cat "$d/max_link_width"))"
    done
    echo; echo "## Resizable BAR (BAR sizes of Intel display devices)"; lspci -vv -d 8086: 2>/dev/null | grep -iE 'resizable|BAR [0-9]: current'
    echo; echo "## SYCL"; command -v icpx > /dev/null && icpx --version | head -1; sycl-ls --verbose 2>/dev/null | grep -iE 'name|driver|version|local mem|global mem|max compute|sub.?group' | head -40
    echo; echo "## OpenCL / Level Zero runtime"; clinfo 2>/dev/null | grep -iE 'driver version|device name' | head -6
    echo; echo "## Vulkan"; vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName|driverInfo|driverName|apiVersion' | head -8
    vulkaninfo 2>/dev/null | grep -i -m2 cooperative_matrix
    echo; echo "## Profilers"; command -v vtune > /dev/null && vtune --version | head -1; command -v unitrace > /dev/null && unitrace --version
  ) > "$OUT/env.txt" 2>&1 || true
fi

bdir() { echo "$LLAMA/build-$1"; }
if has build; then
  for be in $BACKENDS; do
    log "build llama.cpp ($be)"
    case $be in
      sycl)   cfg=(-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx) ;;
      vulkan) cfg=(-DGGML_VULKAN=ON) ;;
      *) echo "unknown backend $be" >&2; exit 2 ;;
    esac
    if ! { cmake -S "$LLAMA" -B "$(bdir "$be")" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "${cfg[@]}" > "$OUT/build-$be.log" 2>&1 &&
           cmake --build "$(bdir "$be")" --target llama-bench test-backend-ops llama-cli >> "$OUT/build-$be.log" 2>&1; }; then
      log "BUILD FAILED ($be; last lines below; full log $OUT/build-$be.log)"; tail -20 "$OUT/build-$be.log" | tee -a "$OUT/run.log"; exit 1
    fi
  done
fi

dev() { case $1 in sycl) echo SYCL0 ;; vulkan) echo Vulkan0 ;; esac; }
for be in $BACKENDS; do
  TBO=$(bdir "$be")/bin/test-backend-ops; D=$(dev "$be")
  # Which FLASH_ATTN_EXT cases the backend supports on the A770 (model-compatibility gaps).
  has support && { log "FA support matrix ($D)"; "$TBO" support -o FLASH_ATTN_EXT -b "$D" > "$OUT/fa-support-$be.txt" 2>&1 || true; }
  # Correctness vs CPU (a backend bug on the A770 is itself a finding).
  has test && { log "FA correctness ($D)"; "$TBO" test -o FLASH_ATTN_EXT -b "$D" > "$OUT/fa-test-$be.txt" 2>&1 || true
                { grep -E "FAIL|passed" "$OUT/fa-test-$be.txt" || echo "  (no pass/fail summary found)"; } | tail -3 | tee -a "$OUT/run.log"; }
  # Kernel-level timing of every FA test case.
  has perf && { log "FA kernel perf ($D)"; stdbuf -oL "$TBO" perf -o FLASH_ATTN_EXT -b "$D" --output sql > "$OUT/fa-perf-$be.sql" 2> "$OUT/fa-perf-$be.err" || true; }
done
# SYCL with the MKL FA path off: isolates what the XMX GEMM path buys for >= 32 tokens.
if has perf && uses sycl; then
  log "FA kernel perf (SYCL0, GGML_SYCL_ENABLE_MKL_FA=0)"
  GGML_SYCL_ENABLE_MKL_FA=0 stdbuf -oL "$(bdir sycl)/bin/test-backend-ops" perf -o FLASH_ATTN_EXT -b SYCL0 --output sql \
    > "$OUT/fa-perf-sycl-nomkl.sql" 2> "$OUT/fa-perf-sycl-nomkl.err" || true
fi

if has bench; then
  [ -f bench/models.txt ] || { log "no bench/models.txt; skipping llama-bench"; log "done: $OUT"; exit 0; }
  while read -r m; do
    [[ -z "$m" || "$m" == \#* ]] && continue
    n=$(basename "$m" .gguf)
    for be in $BACKENDS; do
      LB=$(bdir "$be")/bin/llama-bench; D=$(dev "$be"); log "llama-bench $n ($D)"
      # Prefill and decode vs. context depth, FA on/off (FA off = the non-fused baseline).
      "$LB" -m "$m" -dev "$D" -ngl 99 -fa on,off -p 512,4096 -n 128 -d 0,4096,16384 \
        -r "$REPS" -o jsonl > "$OUT/lb-$n-$be-depth.jsonl" 2> "$OUT/lb-$n-$be-depth.err" || log "  depth sweep failed (see .err)"
      # Quantised KV cache only works with FA on. SYCL B <= 2 goes to VEC, B >= 3 converts K/V to F16 (research §3).
      "$LB" -m "$m" -dev "$D" -ngl 99 -fa on -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -p 4096 -n 128 -d 0,16384 \
        -r "$REPS" -o jsonl > "$OUT/lb-$n-$be-kvq.jsonl" 2> "$OUT/lb-$n-$be-kvq.err" || log "  kv-quant sweep failed (see .err)"
      # Small batches at depth: below 32 tokens SYCL uses the non-XMX TILE kernel, and for GQA 3/5/6/7 it streams
      # K/V G/ncols2 times (research §2). Speculative-verify and multi-user shapes.
      "$LB" -m "$m" -dev "$D" -ngl 99 -fa on -p 1,2,4,8,16,32 -n 0 -d 4096,16384 \
        -r "$REPS" -o jsonl > "$OUT/lb-$n-$be-smallbatch.jsonl" 2> "$OUT/lb-$n-$be-smallbatch.err" || log "  small-batch sweep failed (see .err)"
      # Ubatch size changes the Q-tile count per FA call during prefill.
      "$LB" -m "$m" -dev "$D" -ngl 99 -fa on -ub 256,512,1024,2048 -p 4096 -n 0 -d 0,16384 \
        -r "$REPS" -o jsonl > "$OUT/lb-$n-$be-ubatch.jsonl" 2> "$OUT/lb-$n-$be-ubatch.err" || log "  ubatch sweep failed (see .err)"
    done
  done < bench/models.txt
fi
log "done: $OUT"
