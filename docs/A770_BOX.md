# A770 box: first measuring session

Runtime status before this session: **not run** on A770. CPU smoke proves plumbing only.
Target: Intel Arc A770 (Xe-HPG, acm-g10), llama.cpp at `bench/LLAMA_COMMIT`.

## 0. Quick path (execute in this order)

1. Complete [TOOLCHAIN.md, A770 box](TOOLCHAIN.md#a770-box-when-available):
   ReBAR, bound kernel driver, client GPU runtime/IGC/Level Zero, oneAPI 2026.1,
   Vulkan ANV/cooperative-matrix support, board/VRAM/power/local-memory record,
   profilers and PCIe/IOMMU/peer-access record if multiple cards are installed.
   Have CMake, Ninja and ccache available; this harness installs nothing.
2. Enter the repo and start a persistent `tmux` session. Source the box toolkit,
   then run the repo checks and Phase 0 before the patch A/B:

```bash
cd /home/p51/Github/A770
tmux new -s a770
source /opt/intel/oneapi/setvars.sh
export PATH=/home/p51/Github/A770/.venv/bin:$PATH
export A770_TOOLS=/home/p51/Github/A770/tools
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
icpx --version
sycl-ls
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
ctest --test-dir build --output-on-failure
STEPS="env build support test perf" bench/phase0.sh
bench/patches/run.sh
```

3. Inspect the new `bench/results/patches-*/summary.md`, correctness logs and
   per-case `*.fail-*.txt`. Do not proceed past unexplained failures or empty
   test selections. The CPU-only CPY f16→q4_0 `[256,4,4,4]` flake remains visible;
   rerun its exact case on upstream and conv/conv8 before attributing a regression.
   The harness retains failures and exits nonzero; its timing steps still collect evidence.
4. Stage the four locally supplied models from `bench/models.example.txt`.
   Check available VRAM with 16K KV caches; record OOM/unsupported cases as failures.
   Model paths below assume the example names under `/models`:

```bash
export MODELS="/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf /models/Qwen2.5-7B-Instruct-Q8_0.gguf /models/Qwen2.5-14B-Instruct-Q4_K_M.gguf /models/gemma-2-9b-it-Q6_K.gguf"
STEPS="build lb" bench/patches/run.sh
```

This reuses scratch clones/builds and ccache under `third_party/patches-ab`.
`lb` interleaves upstream/p6/conv8/Vulkan over two rounds; `perf` defaults to three.
SYCL oneDNN remains at its default ON; preserve selectors/flags when comparing runs.
Opt-in P-10 experiment: `VARIANTS="upstream p6 conv conv8 grf" STEPS="build perf" bench/patches/run.sh`.
`grf` is upstream plus the P-10 draft only; it appears only in kernel perf tables.
Use the faster upstream SYCL/Vulkan result for each shape (INV-PERF-002).
Tables show both references; overlap in round ranges is not a gain. For analysis,
`*.compare.jsonl` uses a shared device label; original JSONL stays verbatim.

5. Cross-check static TILE/VEC resource reports against runtime findings:

```bash
LLAMA="$PWD/third_party/patches-ab/upstream" bin/llama-sycl-fa-static.sh bench/results/day1-static-upstream
LLAMA="$PWD/third_party/patches-ab/p6" bin/llama-sycl-fa-static.sh bench/results/day1-static-p6
```

Record compile/report failures and spill changes; static reports are not runtime timing.
6. Copy results home after the kernel run and again after the model/static runs:

```bash
tar czf /tmp/a770-day1-results.tgz bench/results
# On the home machine, replace A770_HOST with the configured SSH host:
scp A770_HOST:/tmp/a770-day1-results.tgz .
tar tzf a770-day1-results.tgz
```

Keep `summary.md`, phase0 env capture, `manifest.txt`, `env-vars.txt`, correctness
and per-case failures, SQL/JSONL, per-round GPU samples, build logs and static reports
in the local archive. Phase0 hardcodes its directory; the harness imports its env capture
into the A/B OUT. Never commit raw logs, models, scratch clones/builds, profiler traces,
credentials or local paths. Commit only reviewed, curated findings/artifacts separately.
Follow [the session prompt](../prompts/a770-session-1.md) for acceptance review.

## 1. BIOS / driver checklist

- Enable Above 4G decoding and Re-Size BAR; record BIOS setting and actual BAR size.
- Record board vendor, A770 8/16 GB, power limit, kernel (`uname -r`), PCIe width/speed.
- Record `lspci -nnk` bound driver (`i915` or `xe`), compute-runtime, IGC and Level Zero
  versions, `sycl-ls --verbose`, `clinfo`, `icpx --version`, Mesa ANV and Vulkan version.
- Static pins: compute-runtime 26.35.39758.10, IGC 2.41.5; runtime IGC must be recorded.
  Confirm 64 KB local memory. Inspect `intel_gpu_top` samples for clocks/load/thermal drift;
  missing samples are explicitly `n/a`, and cannot establish stable GPU state.

## 2. Troubleshooting stubs

- **Device not found:** source oneAPI, check `sycl-ls`, `/dev/dri` access and bound
  driver/runtime. Select `level_zero:gpu`; confirm SYCL0 is the A770 in test output
  (the harness refuses otherwise). With an Intel iGPU present, use `level_zero:<n>` from
  `sycl-ls`, and `GGML_VK_VISIBLE_DEVICES=<n>` so Vulkan0 is the A770 too.
  `FORCE=1` bypasses the guard for diagnosis; it does not validate the hardware.
- **DEVICE_LOST / watchdog:** save stderr and kernel messages; isolate depth/shape.
  Pinned llama.cpp `docs/backend/SYCL.md` documents `GGML_SYCL_FA_ONEDNN_MAX_KV`:
  default 0 uses oneDNN regardless of KV length; a positive cap falls back to native
  kernels beyond it. Choose a cap near the observed reset depth, label and rerun
  every reference with the same setting; do not mix capped/default timing tables.
- **Level Zero vs OpenCL:** record `ONEAPI_DEVICE_SELECTOR` and allocation flags.
  Changing `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` only changes allocation support;
  it does not select an OpenCL device. Test `opencl:gpu` separately with labelled
  output. `CPU_SMOKE=1` uses `opencl:cpu` plus a test-only patch; no A770 timings.
