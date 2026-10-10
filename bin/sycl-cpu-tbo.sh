#!/usr/bin/env bash
# Test-only SYCL correctness on the Intel OpenCL CPU device.
set -euo pipefail
export PATH=/home/p51/Github/A770/.venv/bin:$PATH

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd -- "$root" || exit 1
llama=${LLAMA:-third_party/llama.cpp}
patch=$root/third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch
O=/home/p51/Github/A770/tools/oneapi-2026.1
export OCL_ICD_VENDORS=$O/etc/OpenCL/vendors
export LD_LIBRARY_PATH=$O/lib
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
export GGML_SYCL_TEST_CPU=1

cd -- "$llama" || exit 1
if ! git apply --check --reverse "$patch" 2>/dev/null; then
    git apply --check "$patch"
    git apply "$patch"
fi

# Reconfigure and run incremental Ninja so changed sources and flags are checked.
cmake -S . -B build-sycl-cpu -G Ninja \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF \
    -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DGGML_SYCL_DEVICE_ARCH= \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$O/bin/icx" -DCMAKE_CXX_COMPILER="$O/bin/icpx" \
    -DCMAKE_PREFIX_PATH="$O"
cmake --build build-sycl-cpu --target test-backend-ops -j 8

# The upstream runner can return success when every backend was skipped.
preflight=$(mktemp)
trap 'rm -f -- "$preflight"' EXIT
if ! build-sycl-cpu/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT -p '^$' >"$preflight" 2>&1; then
    cat "$preflight" >&2
    echo 'SYCL CPU preflight failed' >&2
    exit 1
fi
if ! grep -Fq 'GGML_SYCL_TEST_CPU: SYCL0 is Intel OpenCL CPU:' "$preflight" ||
   ! grep -Eq 'Backend [0-9]+/[0-9]+: SYCL0$' "$preflight"; then
    cat "$preflight" >&2
    echo 'SYCL0 is missing or is not the Intel OpenCL CPU device' >&2
    exit 1
fi
cat "$preflight"
rm -f -- "$preflight"
trap - EXIT

args=("$@")
has_backend=0
for arg in "${args[@]}"; do
    if [[ $arg == -b ]]; then
        has_backend=1
    fi
done
if (( ! has_backend )); then
    args+=(-b SYCL0)
fi
exec build-sycl-cpu/bin/test-backend-ops "${args[@]}"
