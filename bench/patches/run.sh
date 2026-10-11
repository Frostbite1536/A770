#!/usr/bin/env bash
# A770 patch-series A/B on bench/LLAMA_COMMIT; no network or source-worktree writes.
# Default: upstream, p6 (0001-2), conv (0001-3), conv8 (0001-4), Vulkan reference.
# Opt in to P-10 with VARIANTS="upstream p6 conv conv8 grf"; grf is upstream +
# p10-draft only, and participates in perf only (no correctness or llama-bench).
# STEPS="env micro build test perf lb"; MODELS is a whitespace-separated GGUF list.
# WORK defaults to gitignored third_party/patches-ab; builds are incremental.
# CPU_SMOKE=1 applies test-only/* ONLY for the required CPU plumbing smoke test.
# It requires FORCE=1 and allows env/build/test only. CPU timing: not run.
# Supply BUILD_FLAGS_EXTRA to clear DEVICE_ARCH and turn oneDNN/Level Zero OFF.
# BUILD_FLAGS_EXTRA is whitespace-separated CMake arguments (no shell evaluation).
# TEST_FILTER applies to FA only; CPY/GET_ROWS always run in full for conv/conv8.
# Raw output stays local. No shipping decision; A770 measurements elsewhere: not run.
set -euo pipefail
export PATH=/home/p51/Github/A770/.venv/bin:$PATH
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
ROOT=$PWD
LLAMA_SRC=${LLAMA_SRC:-$ROOT/third_party/llama.cpp}
WORK=${WORK:-$ROOT/third_party/patches-ab}
OUT=${OUT:-$ROOT/bench/results/patches-$(date +%Y%m%d-%H%M%S)-$$}
VARIANTS=${VARIANTS:-"upstream p6 conv conv8"}
STEPS=${STEPS:-"env micro build test perf lb"}
ROUNDS=${ROUNDS:-3}
LB_ROUNDS=${LB_ROUNDS:-2}
REPS=${REPS:-3}
JOBS=${JOBS:-8}
CPU_SMOKE=${CPU_SMOKE:-0}
PATCHES=$ROOT/third_party/patches
read -r -a variants <<< "$VARIANTS"
read -r -a extra <<< "${BUILD_FLAGS_EXTRA:-}"
read -r -a models <<< "${MODELS:-}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
die() { echo "$*" >&2; exit 2; }
for s in $STEPS; do
    case $s in env|micro|build|test|perf|lb) ;; *) die "unknown step: $s" ;; esac
done
for n in "$ROUNDS" "$LB_ROUNDS" "$REPS" "$JOBS"; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || die "rounds, reps and jobs must be positive integers"
done
[[ $CPU_SMOKE == 0 || $CPU_SMOKE == 1 ]] || die "CPU_SMOKE must be 0 or 1"
if [[ $CPU_SMOKE == 1 ]]; then
    [[ ${FORCE:-0} == 1 ]] || die "CPU_SMOKE requires FORCE=1"
    if has micro || has perf || has lb; then die "CPU_SMOKE allows env/build/test only"; fi
    O=/home/p51/Github/A770/tools/oneapi-2026.1
    export PATH=$O/bin:$PATH OCL_ICD_VENDORS=$O/etc/OpenCL/vendors LD_LIBRARY_PATH=$O/lib
    export ONEAPI_DEVICE_SELECTOR=opencl:cpu GGML_SYCL_TEST_CPU=1
    export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-/tmp/p9-sycl-cache}
else
    export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}
    [[ ${GGML_SYCL_TEST_CPU:-0} != 1 ]] || die "GGML_SYCL_TEST_CPU requires CPU_SMOKE=1"
fi
devices=$(sycl-ls 2>&1 || true)
ver=$(icpx --version 2>&1 || true)
if [[ ${FORCE:-0} != 1 ]]; then
    [[ $devices == *A770* ]] || die "sycl-ls did not show an A770; FORCE=1 overrides"
    [[ $ver =~ Compiler[[:space:]]2026\.1([.[:space:]]|$) ]] || die "icpx 2026.1 required; FORCE=1 overrides"
fi
base=$(cat "$ROOT/bench/LLAMA_COMMIT")
[[ $(git -C "$LLAMA_SRC" rev-parse HEAD) == "$base" ]] || die "LLAMA_SRC must be pinned at $base"

patches_for() {
    local n=0 k files
    case $1 in
        upstream|vulkan) ;; p6) n=2 ;; conv) n=3 ;; conv8) n=4 ;;
        grf) echo "$PATCHES/p10-draft-SYCL-FA-TILE-grf-size-automatic.patch"; return ;;
        *) die "unknown variant: $1" ;;
    esac
    for ((k=1; k<=n; k++)); do
        files=("$PATCHES/$(printf '%04d' "$k")"-*.patch)
        [[ ${#files[@]} == 1 && -f ${files[0]} ]] || die "missing/ambiguous patch $k"
        printf '%s\n' "${files[0]}"
    done
}
[[ ${#variants[@]} -gt 0 ]] || die "VARIANTS is empty"
declare -A seen=()
for v in "${variants[@]}"; do
    [[ $v != vulkan ]] || die "Vulkan is added automatically for perf/lb"
    [[ ! ${seen[$v]+yes} ]] || die "duplicate variant: $v"
    seen[$v]=1
    while IFS= read -r p; do [[ -f $p ]] || die "missing $p"; done < <(patches_for "$v")
    case $v in upstream|p6|conv|conv8|grf) ;; *) die "unknown variant: $v" ;; esac
done
if has lb && (( ${#models[@]} )); then
    for m in "${models[@]}"; do [[ -f $m ]] || die "missing model: $m"; done
    for v in upstream p6 conv8; do
        if [[ ! ${seen[$v]+yes} ]]; then variants+=("$v"); seen[$v]=1; fi
    done
fi
if has perf || { has lb && (( ${#models[@]} )); }; then variants+=(vulkan); fi
mkdir -p "$WORK" "$OUT"
WORK=$(cd "$WORK" && pwd -P)
OUT=$(cd "$OUT" && pwd -P)
[[ $WORK != "$ROOT" && $WORK != "$(cd "$LLAMA_SRC" && pwd -P)" ]] || die "WORK must be a scratch directory"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
failed=0
export CCACHE_BASEDIR=$WORK CCACHE_NOHASHDIR=1 CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}
export CCACHE_DIR=${CCACHE_DIR:-$WORK/ccache}
launcher=()
if command -v ccache >/dev/null; then
    launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
else
    log "ccache unavailable; compiling without cache (no installation)"
fi
{
    printf 'base=%s\nvariants=%s\nSTEPS=%s\nCPU_SMOKE=%s\nFORCE=%s\n' "$base" "${variants[*]}" "$STEPS" "$CPU_SMOKE" "${FORCE:-0}"
    printf 'BUILD_FLAGS_EXTRA=%s\nTEST_FILTER=%s\n' "${BUILD_FLAGS_EXTRA:-}" "${TEST_FILTER:-}"
    printf 'ROUNDS=%s\nLB_ROUNDS=%s\nREPS=%s\nJOBS=%s\n' "$ROUNDS" "$LB_ROUNDS" "$REPS" "$JOBS"
    printf '%s\n%s\n' "$devices" "$ver"
    sha256sum "$PATCHES"/*.patch
    [[ $CPU_SMOKE == 0 ]] || sha256sum "$PATCHES"/test-only/*.patch
} > "$OUT/manifest.txt"
python3 - > "$OUT/env-vars.txt" <<'PY'
import os
for k in sorted(os.environ):
    if k.startswith("GGML_SYCL_") or k in (
        "ONEAPI_DEVICE_SELECTOR", "OCL_ICD_VENDORS", "LD_LIBRARY_PATH",
        "SYCL_CACHE_PERSISTENT", "SYCL_CACHE_DIR", "CCACHE_DIR", "CCACHE_BASEDIR"):
        print(f"{k}={os.environ[k]}")
PY
if has env; then
    # phase0 hardcodes OUT: import its capture rather than duplicate its probes.
    log "env via phase0"
    STEPS=env LLAMA="$LLAMA_SRC" BACKENDS="sycl vulkan" TAG="patches-$$" \
        bash "$ROOT/bench/phase0.sh" > "$OUT/phase0.log" 2>&1
    phase_out=$(sed -n 's/^.*done: //p' "$OUT/phase0.log" | tail -1)
    [[ -f $phase_out/env.txt ]] || die "phase0 env capture missing; see phase0.log"
    cp "$phase_out/env.txt" "$OUT/env.txt"
    printf 'phase0 capture imported from %s\n' "$phase_out" >> "$OUT/manifest.txt"
fi
if has micro; then
    log "dequant microbenchmark (bit-exact required)"
    icpx -O2 -std=c++17 -fsycl -fsycl-device-code-split=per_kernel \
        "$ROOT/bench/micro/dequant_nc_sycl.cpp" -o "$WORK/dequant_nc_sycl" > "$OUT/micro.build.log" 2>&1
    "$WORK/dequant_nc_sycl" --reps "$REPS" > "$OUT/micro.txt" 2>&1 || die "micro failed: $OUT/micro.txt"
    grep -Fq '48/48 type/destination/shape combinations bit-exact; 144/144 comparisons PASS' "$OUT/micro.txt" || die "micro bit-exact summary missing"
fi

prepare() {
    local v=$1 src=$WORK/$1 p signature
    local -a ps=()
    mapfile -t ps < <(patches_for "$v")
    if [[ $CPU_SMOKE == 1 ]]; then ps+=("$PATCHES"/test-only/*.patch); fi
    signature=$({ printf '%s\n' "$base" "$v" "$CPU_SMOKE" 'shared-cases-v2';
        if (( ${#ps[@]} )); then sha256sum "${ps[@]}"; fi; } | sha256sum | cut -d ' ' -f1)
    if [[ -e $src ]]; then
        [[ ! -L $src && -f $src/.ab-signature && $(cat "$src/.ab-signature") == "$signature" ]] || die "scratch recipe changed/incomplete at $src; choose a fresh WORK"
        [[ $(git -C "$src" rev-parse HEAD) == "$base" ]] || die "scratch base changed: $src"
        return
    fi
    git clone --quiet --local --no-hardlinks --no-checkout "$LLAMA_SRC" "$src"
    git -C "$src" checkout --quiet --detach "$base"
    for p in "${ps[@]}"; do git -C "$src" apply --check "$p"; git -C "$src" apply "$p"; done
    # Same eval GQA corpus even upstream; same perf corpus for every backend.
    # Scratch-only test additions: never alter kernels or tolerances here.
    python3 - "$src/tests/test-backend-ops.cpp" "$PATCHES/0002-tests-FA-non-power-of-two-GQA-small-batch.patch" <<'PY'
import pathlib, sys
t, patch = map(pathlib.Path, sys.argv[1:])
s = t.read_text()
marker = "// Non-power-of-two GQA ratios at small batch sizes:"
if marker not in s:
    block = "\n".join(line[1:] for line in patch.read_text().splitlines()
                      if line.startswith("+") and not line.startswith("+++")) + "\n"
    anchor = "    // KV not a multiple of the compaction workgroup size."
    if s.count(anchor) != 1:
        raise SystemExit("eval anchor missing/ambiguous")
    s = s.replace(anchor, block + anchor)
anchor = "    test_cases.emplace_back(new test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776,"
if s.count(anchor) != 1:
    raise SystemExit("perf anchor missing/ambiguous")
block = """    // P9 shared P6/P8 perf cases, including power-of-two controls; 4 KV heads like Qwen2.5-7B.
    for (int hs : {128, 256}) for (int g : {1, 2, 3, 4, 5, 6, 7, 8})
    for (int kv : {4096, 16384}) for (int nb : {1, 2, 3, 4, 8, 16})
    for (ggml_type ty : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 4, {g, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, ty, ty));
    }
"""
s = s.replace(anchor, block + anchor)
t.write_text(s)
PY
    printf '%s\n' "$signature" > "$src/.ab-signature"
}
printf '| variant | op | rc | pass | fail | unsupported |\n|---|---|---:|---:|---:|---:|\n' > "$OUT/correctness.md"
correctness() {
    local v=$1 op=$2 tbo=$WORK/$1/build/bin/test-backend-ops rc=0 counts
    local -a filter=()
    if [[ $op == FLASH_ATTN_EXT && -n ${TEST_FILTER:-} ]]; then filter=(-p "$TEST_FILTER"); fi
    log "correctness $v $op"
    "$tbo" test -b SYCL0 -o "$op" "${filter[@]}" > "$OUT/$v.test-$op.txt" 2>&1 || rc=$?
    counts=$(python3 - "$OUT/$v.test-$op.txt" "$OUT/$v.fail-$op.txt" "$v" "$op" "$rc" "$OUT/correctness.md" <<'PY'
import pathlib, re, sys
path, failures, variant, op, rc, table = sys.argv[1:]
s = re.sub(r"\x1b\[[0-9;]*m", "", pathlib.Path(path).read_text(errors="replace"))
summary = re.findall(r"^  (\d+)/(\d+) tests passed$", s, re.M)
passed, total = map(int, summary[-1]) if summary else (0, 0)
unsupported = sum("not supported" in line and line.startswith(f"  {op}(") for line in s.splitlines())
failed = total - passed
details = s.split("Failing tests:\n", 1)[-1] if "Failing tests:\n" in s else "\n".join(
    line for line in s.splitlines() if "FAIL" in line or "Error" in line)
pathlib.Path(failures).write_text(details + "\n")
with open(table, "a") as f:
    f.write(f"| {variant} | {op} | {rc} | {passed} | {failed} | {unsupported} |\n")
print(f"rc={rc} PASS={passed} FAIL={failed} UNSUPPORTED={unsupported} executed={total}")
PY
)
    log "  $v $op $counts"
    if [[ $rc != 0 || $counts != *'FAIL=0 '* || $counts == *'executed=0' ]]; then
        failed=1
        log "  failures/incomplete execution: $OUT/$v.fail-$op.txt (full log: $v.test-$op.txt)"
        cat "$OUT/$v.fail-$op.txt" | tee -a "$OUT/run.log"
    fi
}
for v in "${variants[@]}"; do
    prepare "$v"
    git -C "$WORK/$v" diff > "$OUT/$v.patch"
    if has build; then
        cfg=(-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx)
        [[ $v != vulkan ]] || cfg=(-DGGML_VULKAN=ON)
        log "build $v"
        cmake -S "$WORK/$v" -B "$WORK/$v/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
            "${cfg[@]}" "${launcher[@]}" "${extra[@]}" > "$OUT/$v.build.log" 2>&1
        cmake --build "$WORK/$v/build" --target test-backend-ops llama-bench -j "$JOBS" >> "$OUT/$v.build.log" 2>&1
    fi
    if has test && [[ $v != vulkan && $v != grf ]]; then
        tbo=$WORK/$v/build/bin/test-backend-ops
        [[ -x $tbo ]] || die "missing build for $v; include STEPS=build"
        "$tbo" test -b SYCL0 -o FLASH_ATTN_EXT -p '^$' > "$OUT/$v.preflight.txt" 2>&1 || die "SYCL preflight failed: $v"
        grep -Eq 'Backend [0-9]+/[0-9]+: SYCL0$' "$OUT/$v.preflight.txt" || die "SYCL0 missing: $v"
        # sycl-ls listing an A770 does not make it SYCL0 (an iGPU can come first).
        if [[ $CPU_SMOKE == 0 && ${FORCE:-0} != 1 ]]; then
            grep -Eq 'Device description: .*A770' "$OUT/$v.preflight.txt" || die "SYCL0 is not the A770 (see $v.preflight.txt); set ONEAPI_DEVICE_SELECTOR"
        fi
        if [[ $CPU_SMOKE == 1 ]]; then
            grep -Fq 'GGML_SYCL_TEST_CPU: SYCL0 is Intel OpenCL CPU:' "$OUT/$v.preflight.txt" || die "CPU SYCL0 missing: $v"
        fi
        correctness "$v" FLASH_ATTN_EXT
        if [[ $v == conv || $v == conv8 ]]; then correctness "$v" CPY; correctness "$v" GET_ROWS; fi
    fi
done
gpu_state() {
    # intel_gpu_top has no portable sample-count flag. Bound our own child,
    # retain its first complete JSON object, and record n/a if no sample arrived.
    python3 - > "$1" <<'PY'
import json, shutil, subprocess
if not shutil.which("intel_gpu_top"):
    print("n/a (intel_gpu_top unavailable)")
else:
    child = subprocess.Popen(["intel_gpu_top", "-J", "-s", "1000"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = child.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        child.terminate()  # Only the child started above.
        try:
            out, err = child.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            child.kill()
            out, err = child.communicate()
    s = out.decode(errors="replace")
    try:
        sample, _ = json.JSONDecoder().raw_decode(s[s.index("{"):])
        print(json.dumps(sample, indent=2))
    except (ValueError, json.JSONDecodeError):
        print(f"n/a (no complete intel_gpu_top sample; rc={child.returncode})")
        print(err.decode(errors="replace").strip())
PY
}
device() { if [[ $1 == vulkan ]]; then echo Vulkan0; else echo SYCL0; fi; }
PERF_FILTER=${PERF_FILTER:-'hsk=(128|256),hsv=(128|256),nh=4,nr23=\[[1-8],1\],kv=(4096|16384),nb=(1|2|3|4|8|16),mask=1,sinks=0,'}
if has perf; then
    for ((i=1; i<=ROUNDS; i++)); do
        for v in "${variants[@]}"; do
            gpu_state "$OUT/$v.perf.r$i.gpu.txt"
            log "perf $v round $i; GPU state: $v.perf.r$i.gpu.txt"
            "$WORK/$v/build/bin/test-backend-ops" perf -b "$(device "$v")" -o FLASH_ATTN_EXT -p "$PERF_FILTER" --output sql \
                > "$OUT/$v.perf.r$i.sql" 2> "$OUT/$v.perf.r$i.err" || { log "perf failed: $v round $i"; failed=1; }
        done
    done
fi
if has lb; then
    (( ${#models[@]} )) || log "MODELS empty; llama-bench not run"
    model_id=0
    for m in "${models[@]}"; do
        model_id=$((model_id+1))
        printf 'model %s=%s\n' "$model_id" "$m" >> "$OUT/manifest.txt"
        for tag in small kvq8; do
            if [[ $tag == small ]]; then args=(-p '1,2,4,8,16' -n 0 -d '4096,16384')
            else args=(-ctk q8_0 -ctv q8_0 -p '4,8' -n 32 -d 16384); fi
            for ((i=1; i<=LB_ROUNDS; i++)); do
                for v in upstream p6 conv8 vulkan; do
                    gpu_state "$OUT/$v.m$model_id.$tag.r$i.gpu.txt"
                    log "llama-bench model $model_id $tag $v round $i; GPU state: $v.m$model_id.$tag.r$i.gpu.txt"
                    "$WORK/$v/build/bin/llama-bench" -m "$m" -dev "$(device "$v")" -ngl 99 -fa on -r "$REPS" -o jsonl "${args[@]}" \
                        > "$OUT/$v.m$model_id.$tag.r$i.jsonl" 2> "$OUT/$v.m$model_id.$tag.r$i.err" || { log "llama-bench failed: $v"; failed=1; }
                done
            done
        done
    done
fi
analyze() {
    local kind=$1 suffix=$2 rounds=$3 v i g f
    shift 3
    local -a groups=() labels=()
    for v in "$@"; do
        g=""
        for ((i=1; i<=rounds; i++)); do
            f=$OUT/$v.$suffix.r$i.sql
            if [[ $kind == lb ]]; then
                f=$OUT/$v.$suffix.r$i.compare.jsonl
                # analyze.py matches on devices: compare the same physical card
                # across backends while preserving the original JSONL verbatim.
                if ! python3 - "$OUT/$v.$suffix.r$i.jsonl" "$f" "$(device "$v")" <<'PY'
import json, pathlib, sys
source, dest, device = sys.argv[1:]
rows = []
for line in pathlib.Path(source).read_text().splitlines():
    if not line.lstrip().startswith("{"):
        continue
    row = json.loads(line)
    if row.get("devices") != device:
        raise SystemExit(f"unexpected device in {source}: {row.get('devices')!r}")
    row["devices"] = "A7700"
    rows.append(json.dumps(row))
if not rows:
    raise SystemExit(f"no llama-bench rows: {source}")
pathlib.Path(dest).write_text("\n".join(rows) + "\n")
PY
                then echo "Comparison input failed: $v round $i"; failed=1; return; fi
            fi
            g+="${g:+,}$f"
        done
        groups+=("$g"); labels+=("$v")
    done
    python3 "$ROOT/bench/analyze.py" "$kind" "${groups[@]}" --labels "$(IFS=,; echo "${labels[*]}")" || { echo "Analysis failed; inspect raw output."; failed=1; }
}
{
    echo '# A770 patch-series A/B'; echo
    echo "Base: $base. CPU_SMOKE=$CPU_SMOKE, FORCE=${FORCE:-0}."
    if [[ $CPU_SMOKE == 1 ]]; then
        echo 'CPU smoke proves plumbing only. A770 timing: not run.'
    else
        echo 'Only executed steps have results below; missing steps are not run. FORCE runs require independent hardware verification.'
    fi
    echo 'Read manifest.txt, env.txt (if requested), env-vars.txt and round GPU samples with these results.'
    echo 'INV-PERF-002: use the faster upstream SYCL/Vulkan result for each shape; missing/unsupported rows have no established baseline.'
    echo; echo '## Correctness'; echo
    if has test; then cat "$OUT/correctness.md"; else echo 'not run'; fi
    echo; echo 'Per-case failures: *.fail-*.txt; full output: *.test-*.txt. Known upstream CPU flake:'
    # shellcheck disable=SC2016 # Markdown code span, deliberately literal.
    echo '`CPY(type_src=f16,type_dst=q4_0,ne_src=[256,4,4,4],...)`; counted, never suppressed.'
    echo; echo '## Conversion microbenchmark'; echo
    if [[ -f $OUT/micro.txt ]]; then
        echo '| type/dst/D/KV/heads | upstream μs | rows μs | vec8 μs | D2D μs | bit-exact |'
        echo '|---|---:|---:|---:|---:|---|'
        awk -F ' \| ' 'NF==3 {split($2,t," "); printf "| %s | %s | %s | %s | %s | %s |\n",$1,t[1],t[2],t[3],t[4],$3}' "$OUT/micro.txt"
    else echo 'not run'; fi
    echo; echo '## Kernel medians and ranges'; echo
    if has perf; then
        analyze tbo perf "$ROUNDS" "${variants[@]}"
        echo; echo '### Relative to upstream Vulkan'; echo
        refs=(vulkan); for v in "${variants[@]}"; do [[ $v == vulkan ]] || refs+=("$v"); done
        analyze tbo perf "$ROUNDS" "${refs[@]}"
    else echo 'not run'; fi
    if ! has lb || (( ${#models[@]} == 0 )); then
        echo; echo '## llama-bench'; echo; echo 'not run'
    else
        for ((model_id=1; model_id<=${#models[@]}; model_id++)); do
            for tag in small kvq8; do
                echo; echo "## llama-bench model $model_id $tag"; echo
                analyze lb "m$model_id.$tag" "$LB_ROUNDS" upstream p6 conv8 vulkan
                echo; echo '### Relative to upstream Vulkan'; echo
                analyze lb "m$model_id.$tag" "$LB_ROUNDS" vulkan upstream p6 conv8
            done
        done
    fi
} > "$OUT/summary.md"
log "done: $OUT/summary.md (failure status=$failed)"
exit "$failed"
