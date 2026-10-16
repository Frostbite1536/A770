#!/bin/bash
# phase0.sh: Phase 0 baseline on the A770 box (docs/ROADMAP.md). Run from the repo root with oneAPI 2026.1 sourced.
#   bench/phase0.sh                        full run: env, build, FA/MUL_MAT correctness/perf, llama-bench grid
#   STEPS="env perf" bench/phase0.sh       run only some steps (env build support test perf mmperf bench; opt-in prof)
#   STEPS=prof bench/phase0.sh            opt-in 8/16-token device-time breakdown (needs unitrace and models.txt)
#   PROF_REPS=4 STEPS=prof bench/phase0.sh  differential captures at -r 1 and -r (1 + PROF_REPS); default 4
#   BACKENDS=sycl bench/phase0.sh          one backend only (default: "sycl vulkan"; INV-PERF-002 needs both)
#   LLAMA=third_party/other-tree bench/phase0.sh   same kit against another llama.cpp tree (results tagged by tree name)
# Models: one GGUF path per line in bench/models.txt (see bench/models.example.txt).
# Output: bench/results/phase0-<tree>-<date>-<host>/ (commit the small text/sql/jsonl files, never models).
# MUL_MAT: ROUNDS=3 interleaved rounds of sycl (SYCL0 defaults), sycl-noopt (SYCL0,
# GGML_SYCL_ENABLE_OPT=0 disables reordered-weight optimization), sycl-nofg (SYCL0,
# GGML_SYCL_XMX_GATHER_TYPES=0 disables fused XMX dequant-GEMM), vulkan (Vulkan0 defaults).
# Pin 79e2e74: ggml/src/ggml-sycl/ggml-sycl.cpp:418,431-432 reads the switches;
# :240,4201-4248,4861-4862 gates reorder; :3134 tries fused GEMM. GGML_SYCL_XMX_GATHER_SHAPES
# masks its matrix combinations (ggml/src/ggml-sycl/fused-gemm.cpp:116,244 logs availability).
# See docs/research/volta-fa-p9-lessons-2026-10-11.md sections 1-3 for small-batch MUL_MAT context.
# Status: drafted on a machine without an Arc GPU; NOT RUN yet.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
LLAMA=${LLAMA:-third_party/llama.cpp}
STEPS=${STEPS:-"env build support test perf mmperf bench"}
BACKENDS=${BACKENDS:-"sycl vulkan"}
REPS=${REPS:-3}
ROUNDS=${ROUNDS:-3}
PROF_REPS=${PROF_REPS:-4}
TAG=${TAG:-$(basename "$LLAMA")}
OUT=bench/results/phase0-$TAG-$(date +%Y%m%d-%H%M%S)-$(hostname -s 2>/dev/null || hostname)
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
uses() { [[ " $BACKENDS " == *" $1 "* ]]; }

# A prof-only invocation must skip cleanly even on a host without the GPU toolchain.
if has prof; then
  if [ ! -f bench/models.txt ]; then
    log "no bench/models.txt; skipping prof"
    STEPS=${STEPS//prof/}
  elif ! command -v unitrace > /dev/null; then
    log "unitrace not found on PATH; skipping prof"
    STEPS=${STEPS//prof/}
  fi
  [[ "$STEPS" =~ [^[:space:]] ]] || exit 0
fi

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
  # Explicit every-build list: future reference-only bench patches must not enter this list.
  bench_patches=(
    third_party/patches/bench-only/0001-tests-mul_mat-perf-cases-1-16.patch
    third_party/patches/bench-only/0002-tests-mul_mat-activation-range.patch
  )
  for patch in "${bench_patches[@]}"; do
    patch=$PWD/$patch
    if git -C "$LLAMA" apply --check --reverse "$patch" 2>/dev/null; then
      log "bench-only patch already applied: $patch"
    elif git -C "$LLAMA" apply --check "$patch"; then
      log "apply bench-only patch (every build): $patch"
      git -C "$LLAMA" apply "$patch"
    else
      log "PATCH FAILED: $patch neither applies nor is already applied in $LLAMA"
      exit 1
    fi
  done
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
  # Excludes bench-only 0002's src1_scale cases (they fail on q8_1 paths by design; reported separately below).
  has test && { log "MUL_MAT correctness ($D)"; "$TBO" test -o MUL_MAT -b "$D" -p '^(?!.*src1_scale=)' > "$OUT/mm-test-$be.txt" 2>&1 || true
                { grep -E "FAIL|passed" "$OUT/mm-test-$be.txt" || echo "  (no pass/fail summary found)"; } | tail -3 | tee -a "$OUT/run.log"; }
  if has test; then
    log "MUL_MAT activation range: these fail on q8_1 paths by design ($D)"
    range_arms=("$be")
    [[ $D == SYCL0 ]] && range_arms+=(sycl-nofg sycl-noopt)
    for arm in "${range_arms[@]}"; do
      arm_env=()
      case $arm in
        sycl-nofg) arm_env=(GGML_SYCL_XMX_GATHER_TYPES=0) ;;
        sycl-noopt) arm_env=(GGML_SYCL_ENABLE_OPT=0) ;;
      esac
      range_rc=0
      env -u GGML_SYCL_ENABLE_OPT -u GGML_SYCL_XMX_GATHER_TYPES -u GGML_SYCL_XMX_GATHER_SHAPES \
        "${arm_env[@]}" "$TBO" test -o MUL_MAT -b "$D" -p src1_scale \
        > "$OUT/range-$arm.txt" 2>&1 || range_rc=$?
      log "activation range ($arm, rc=$range_rc): $OUT/range-$arm.txt"
      { grep -E '[0-9]+/[0-9]+ tests passed' "$OUT/range-$arm.txt" || echo "  (no pass/fail summary found)"; } | tee -a "$OUT/run.log"
    done
  fi
  # Kernel-level timing of every FA test case.
  has perf && { log "FA kernel perf ($D)"; stdbuf -oL "$TBO" perf -o FLASH_ATTN_EXT -b "$D" --output sql > "$OUT/fa-perf-$be.sql" 2> "$OUT/fa-perf-$be.err" || true; }
done
# SYCL with the MKL FA path off: isolates what the XMX GEMM path buys for >= 32 tokens.
if has perf && uses sycl; then
  log "FA kernel perf (SYCL0, GGML_SYCL_ENABLE_MKL_FA=0)"
  GGML_SYCL_ENABLE_MKL_FA=0 stdbuf -oL "$(bdir sycl)/bin/test-backend-ops" perf -o FLASH_ATTN_EXT -b SYCL0 --output sql \
    > "$OUT/fa-perf-sycl-nomkl.sql" 2> "$OUT/fa-perf-sycl-nomkl.err" || true
fi

if has mmperf; then
  [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || { log "mmperf failed: ROUNDS must be a positive integer"; exit 1; }
  # Full params anchor excludes batched, strided, permuted and overlapping variants.
  MM_FILTER='^type_a=(q4_0|q8_0|q4_K|q5_K|q6_K),type_b=f32,(m=(4096|1024|14336),n=([1-9]|1[0-6]),k=4096|m=4096,n=([1-9]|1[0-6]),k=14336),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0$'
  arms=()
  uses sycl && arms+=(sycl sycl-noopt sycl-nofg)
  uses vulkan && arms+=(vulkan)
  for ((round=1; round<=ROUNDS; round++)); do
    for arm in "${arms[@]}"; do
      be=sycl; arm_env=()
      case $arm in
        sycl-noopt) arm_env=(GGML_SYCL_ENABLE_OPT=0) ;;
        sycl-nofg) arm_env=(GGML_SYCL_XMX_GATHER_TYPES=0) ;;
        vulkan) be=vulkan ;;
      esac
      log "MUL_MAT perf ($arm, round $round/$ROUNDS)"
      # Each arm starts from the default switches, even if the caller exported one.
      if ! env -u GGML_SYCL_ENABLE_OPT -u GGML_SYCL_XMX_GATHER_TYPES -u GGML_SYCL_XMX_GATHER_SHAPES \
          "${arm_env[@]}" stdbuf -oL "$(bdir "$be")/bin/test-backend-ops" perf -o MUL_MAT \
          -b "$(dev "$be")" --output sql -p "$MM_FILTER" \
          > "$OUT/mm-perf-$arm.r$round.sql" 2> "$OUT/mm-perf-$arm.r$round.err"; then
        log "mmperf failed: $arm round $round (see .err)"
        exit 1
      fi
      if [[ "$arm" == sycl && "$round" == 1 ]]; then
        # Combinations are logged lazily on the first eligible MUL_MAT, after start-up.
        cp "$OUT/mm-perf-sycl.r1.err" "$OUT/sycl-env.txt"
        log "SYCL start-up settings and XMX availability: $OUT/sycl-env.txt"
      fi
    done
  done
  groups=()
  for arm in "${arms[@]}"; do
    group=
    for ((round=1; round<=ROUNDS; round++)); do
      group+=${group:+,}$OUT/mm-perf-$arm.r$round.sql
    done
    groups+=("$group")
  done
  python bin/mm-gbps.py "${groups[@]}" --labels "$(IFS=,; echo "${arms[*]}")" > "$OUT/mmperf.md"
fi

if has prof; then
  m=
  while IFS= read -r candidate || [[ -n "$candidate" ]]; do
    # Ignore blank/comment lines, including leading whitespace; preserve spaces inside paths.
    candidate=${candidate#"${candidate%%[![:space:]]*}"}
    candidate=${candidate%"${candidate##*[![:space:]]}"}
    [[ -z "$candidate" || "$candidate" == \#* ]] && continue
    m=$candidate
    break
  done < bench/models.txt
  if [ -z "$m" ]; then
    log "no model paths in bench/models.txt; skipping prof"
  else
    if [[ ! "$PROF_REPS" =~ ^[1-9][0-9]*$ ]]; then
      log "prof failed: PROF_REPS must be a positive integer"
      exit 1
    fi
    mkdir -p "$OUT/prof"
    # Some releases return nonzero for --help. Inspect the saved help text in either case.
    unitrace --help > "$OUT/prof/unitrace-help.txt" 2>&1 || true
    if ! grep -q -- '--device-timing' "$OUT/prof/unitrace-help.txt"; then
      log "prof failed: unitrace --help lacks --device-timing (see $OUT/prof/unitrace-help.txt)"
      exit 1
    fi
    for be in $BACKENDS; do
      LB=$(bdir "$be")/bin/llama-bench; D=$(dev "$be")
      for p in 8 16; do
        for depth in 0 16384; do
          for run in 1 N; do
            repetitions=1
            [[ "$run" == N ]] && repetitions=$((1 + PROF_REPS))
            capture=$OUT/prof/$be-p$p-d$depth-r$run.txt
            log "prof ($be, p=$p, depth=$depth, r=$repetitions)"
            if ! unitrace --device-timing --kernel-submission "$LB" -m "$m" -dev "$D" \
                -ngl 99 -fa 1 -p "$p" -n 0 -d "$depth" -r "$repetitions" \
                > "$capture" 2>&1; then
              log "prof failed ($be, p=$p, depth=$depth, r=$repetitions; see $capture)"
              exit 1
            fi
          done
        done
      done
    done
    if ! python bin/step-breakdown.py "$OUT/prof" --reps "$PROF_REPS"; then
      log "prof failed: could not parse device-timing summaries in $OUT/prof"
      exit 1
    fi
  fi
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
