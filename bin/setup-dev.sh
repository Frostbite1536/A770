#!/bin/bash
# setup-dev.sh: no-sudo dev toolchain for a machine without an Arc GPU or oneAPI (docs/TOOLCHAIN.md).
#   .venv/                cmake, ninja, shellcheck, numpy
#   tools/uv              pinned uv binary (python3-venv/pip may be missing)
#   tools/neo/            ocloc 26.35 + IGC 2.41.5 extracted from Intel's .debs (static acm-g10 reports)
#   tools/oneapi-2026.1/  icpx/DPC++ 2026.1, oneMKL SYCL 2026.1 (llama.cpp's SYCL backend needs it) and the
#                         OpenCL CPU runtime (runs SYCL on this CPU for correctness), from conda-forge
#                         (optional: WITH_DPCPP=1, ~4.4 GB)
# Everything is gitignored and checksum-verified. Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2
UV_VERSION=0.12.17
MAMBA_VERSION=2.8.1-1
NEO_VERSION=26.35.39758.10
IGC_VERSION=2.41.5
IGC_BUILD=22716
GMM_VERSION=22.10.0
mkdir -p tools
fetch() {  # url dest: download and verify against the release's .sha256 file
    curl -fsSL -o "$2" "$1"
    curl -fsSL -o "$2.sha256" "$1.sha256"
    echo "$(awk '{print $1}' "$2.sha256")  $2" | sha256sum -c --quiet
    rm -f "$2.sha256"
}
fetch_pinned() {  # url dest sha256: download and verify against a hash recorded here
    curl -fsSL -o "$2" "$1"
    echo "$3  $2" | sha256sum -c --quiet
}

if [ ! -x tools/uv ]; then
    echo "[setup-dev] uv $UV_VERSION"
    tmp=$(mktemp -d)
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" "$tmp/uv.tgz"
    tar xzf "$tmp/uv.tgz" -C "$tmp" && cp "$tmp/uv-x86_64-unknown-linux-gnu/uv" tools/uv && rm -rf "$tmp"
fi

echo "[setup-dev] .venv"
export UV_CACHE_DIR=$PWD/tools/.uv-cache
[ -x .venv/bin/python ] || tools/uv venv -q --python 3.12 .venv
tools/uv pip install -q --python .venv/bin/python cmake ninja shellcheck-py numpy

if [ ! -e tools/neo/bin/ocloc ]; then
    # ocloc and gmmlib hashes match compute-runtime's ww35.sum; IGC publishes no sum file, so its hashes
    # were recorded on first download (2026-10-10). A mismatch means the release asset changed: stop.
    echo "[setup-dev] ocloc $NEO_VERSION + IGC $IGC_VERSION (~110 MB)"
    CR=https://github.com/intel/compute-runtime/releases/download/$NEO_VERSION
    IG=https://github.com/intel/intel-graphics-compiler/releases/download/v$IGC_VERSION
    tmp=$(mktemp -d)
    fetch_pinned "$IG/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" "$tmp/igc-core.deb" \
        0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628
    fetch_pinned "$IG/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" "$tmp/igc-opencl.deb" \
        779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058
    fetch_pinned "$CR/intel-ocloc_${NEO_VERSION}-0_amd64.deb" "$tmp/ocloc.deb" \
        c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7
    fetch_pinned "$CR/libigdgmm12_${GMM_VERSION}_amd64.deb" "$tmp/gmm.deb" \
        6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54
    rm -rf tools/neo && mkdir -p tools/neo/bin
    for d in "$tmp"/*.deb; do dpkg-deb -x "$d" tools/neo; done
    rm -rf "$tmp"
    # Wrapper so icpx (which calls `ocloc` from PATH) and scripts find IGC's libraries.
    cat > tools/neo/bin/ocloc <<'EOT'
#!/bin/bash
NEO="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$NEO/usr/lib/x86_64-linux-gnu:$NEO/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$(ls "$NEO"/usr/bin/ocloc-* | head -1)" "$@"
EOT
    chmod +x tools/neo/bin/ocloc
fi

if [ "${WITH_DPCPP:-0}" = 1 ] && { [ ! -x tools/oneapi-2026.1/bin/icpx ] || [ ! -d tools/oneapi-2026.1/lib/cmake/mkl ] ||
                                  [ ! -e tools/oneapi-2026.1/etc/OpenCL/vendors/intel-ocl-cpu.icd ] || [ ! -d tools/oneapi-2026.1/lib/cmake/TBB ]; }; then
    echo "[setup-dev] icpx/DPC++ + oneMKL SYCL + OpenCL CPU runtime 2026.1 via conda-forge (~4.4 GB)"
    [ -x tools/micromamba ] || { fetch "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" tools/micromamba; chmod +x tools/micromamba; }
    export MAMBA_ROOT_PREFIX=$PWD/tools/mamba
    # tbb-devel: oneMKL's CMake config needs TBB's (MKL_SYCL_THREADING defaults to tbb_thread).
    pkgs=("dpcpp_linux-64=2026.1.*" "mkl-devel-dpcpp=2026.1.*" "intel-opencl-rt=2026.1.*" tbb-devel)
    if [ -d tools/oneapi-2026.1/conda-meta ]; then tools/micromamba install -y -q -p tools/oneapi-2026.1 -c conda-forge "${pkgs[@]}"
    else tools/micromamba create -y -q -p tools/oneapi-2026.1 -c conda-forge "${pkgs[@]}"; fi
    tools/micromamba clean -a -y -q > /dev/null
fi

echo "[setup-dev] check"
tools/neo/bin/ocloc --version 2>&1 | head -1 | sed 's/^/  ocloc: /'
[ -x tools/oneapi-2026.1/bin/icpx ] && echo "  icpx: $(tools/oneapi-2026.1/bin/icpx --version | head -1)"
echo "next: export PATH=\$PWD/.venv/bin:\$PATH; cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build"
