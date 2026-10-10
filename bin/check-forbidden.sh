#!/bin/bash
# Fails if source uses device features the A770 (Xe-HPG, acm-g10) does not have, or that come from
# another architecture's code (INV-ARCH-001). bin/xe-report.py catches what grep cannot (SIMD16 DPAS
# that compiles to nothing, FP64 in SYCL device code); this catches code paths no report compiled.
# tests/xe/ holds deliberate negative fixtures for xe-report and is not scanned.
cd "$(dirname "$0")/.." || exit 2
# 2D block load/store/prefetch (Xe-HPC/Xe2 only; undeclared for acm-g10), TF32 and FP8 (not on Xe-HPG XMX),
# SIMD16 IGC DPAS builtins, CUDA intrinsics (code copied from the CUDA backend instead of ported).
PATTERN='_2d_block_(read|write|prefetch)|lsc_(load|store|prefetch)_2d|block_2d|precision::tf32|\btf32\b|matrix_mad_k8|\bfp8\b|e4m3|e5m2|__builtin_IB_sub_group16_\w*dpas|__shfl_|__syncthreads|nvcuda::|mma\.sync|cp\.async|ldmatrix'
# FP64 in OpenCL C kernels (host C++ legitimately uses double, e.g. the reference).
PATTERN_CL='\bdouble\b|cl_khr_fp64'
dirs=()
for d in src include tests bench; do [ -d "$d" ] && dirs+=("$d"); done
[ ${#dirs[@]} -eq 0 ] && { echo "ok (nothing to scan)"; exit 0; }
rc=0
if grep -rnE "$PATTERN" "${dirs[@]}" --exclude-dir=xe --include='*.cpp' --include='*.hpp' --include='*.h' --include='*.cl' --include='*.comp' --include='*.glsl'; then
  echo "forbidden non-Xe-HPG feature found (INV-ARCH-001)" >&2; rc=1
fi
if grep -rnE "$PATTERN_CL" "${dirs[@]}" --exclude-dir=xe --include='*.cl'; then
  echo "FP64 in an OpenCL kernel: the A770 has no FP64 hardware (INV-ARCH-001)" >&2; rc=1
fi
[ $rc -eq 0 ] && echo "ok"
exit $rc
