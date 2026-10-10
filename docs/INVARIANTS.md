# System Invariants

Non-negotiable rules. A change that violates one is a bug or an explicit product decision, never an implementation detail. Changes to this file require a justification in the change log at the bottom.

The numerics, performance-honesty and process invariants are inherited from volta-fa unchanged (they do not depend on the GPU). The architecture invariants are new, and each one cites a compile probe in `docs/research/xe-hpg-static-probes-2026-10-10.md` instead of memory.

## Architecture invariants

### INV-ARCH-001: Xe-HPG acm-g10 only
**Rule**: Device code targets `acm-g10` (Arc A770/A750) only. XMX is reached through DPAS at sub-group size 8: SYCL `joint_matrix` with M=8, N=8, K=16 and f16 or bf16 inputs with f32 accumulate, or the OpenCL SIMD8 builtins (`intel_sub_group_f16_f16_matrix_mad_k16(int8, int8, float8)` and the bf16 equivalents).
**Rationale**: Xe-HPG's XMX runs SIMD8 DPAS. Code written for Xe2 (Battlemage, Lunar Lake) or Xe-HPC (PVC) uses SIMD16 DPAS and 2D block messages. On acm-g10 the SIMD16 OpenCL builtin **compiles without an error and without a DPAS instruction**, so its output registers are never written. The 8×16×16 `joint_matrix` shape crashes IGC. 2D block builtins are undeclared, and FP64 is refused (no FP64 hardware).
**Valid**: `joint_matrix<sub_group, half|bfloat16, use::a, 8, 16>` × `<..., use::b, 16, 8>` → `<float, accumulator, 8, 8>` under `[[sycl::reqd_sub_group_size(8)]]`; OpenCL `matrix_mad_k16` with an `int8` A operand under `intel_reqd_sub_group_size(8)`; non-XMX code at sub-group 8, 16 or 32.
**Invalid**: SIMD16 DPAS forms and shapes; `intel_sub_group_2d_block_*` and ESIMD `lsc_*_2d`; TF32; FP8; `double`/`cl_khr_fp64`/`sycl::aspect::fp64` in device code; kernels copied from Xe2/PVC libraries.
**Enforcement**: `bin/xe-report.py` compiles for acm-g10 with the pinned IGC and fails on DPAS = 0 when XMX was requested, DPAS at SIMD ≠ 8, or FP64 instructions. `bin/check-forbidden.sh` greps what no report compiled. `tests/xe/` keeps a negative fixture for every trap above.

### INV-ARCH-002: Per-work-group SLM and GRF budget
**Rule**: Static + dynamic SLM per work-group ≤ 64 KB. Each kernel declares its sub-group size and GRF mode (128 default, or 256 with `-ze-opt-large-register-file` / `-cl-intel-256-GRF-per-thread`) next to its launch config, and documents its occupancy target.
**Rationale**: IGC accepts 96 KB of SLM for acm-g10 at compile time (probe 6), so the limit has to be checked by us. 256-GRF mode halves hardware threads per XVE from 8 to 4 (probe 5): fewer spills, half the latency hiding. The 64 KB figure is from Intel's Xe-HPG documentation; confirm `local_mem_size` on the box (`sycl-ls --verbose`) and record it in `bench/BASELINE.md`.
**Enforcement**: `bin/xe-report.py` (SLM > 64 KB fails; the `thr` column shows the GRF-mode cost).

### INV-ARCH-003: Pinned toolchain
**Rule**: oneAPI DPC++ 2026.1.x (llama.cpp's SYCL CI uses 2026.1), compute-runtime 26.35.39758.10 with IGC 2.41.5 for static reports, and on the box a driver stack whose IGC version is recorded with every result.
**Rationale**: IGC decides SIMD width, spills and whether DPAS is emitted. The same source can change results across IGC versions, so a report or a timing is only meaningful with its compiler version attached.
**Enforcement**: `bin/setup-dev.sh` pins and checksums the debs; CMake refuses icpx outside 2026.1.x; `bench/phase0.sh` records versions in `env.txt`.

### INV-ARCH-004: Explicit sub-group size
**Rule**: Every kernel declares its sub-group size.
**Rationale**: Without it IGC picks SIMD8/16/32 per kernel and may change its choice when it retries after a spill (probe 3). Both DPAS legality and the meaning of per-work-item fragments depend on that choice.
**Enforcement**: review; `bin/xe-report.py` prints the SIMD width that was actually compiled.

## Correctness invariants

### INV-NUM-001: No materialised score matrix
**Rule**: The fused path never writes the N×N scores/probabilities to global memory. Only O (and the log-sum-exp statistic if needed for split-KV) leave the Xe-core.
**Enforcement**: memory-traffic check in `bench/` (DRAM bytes ≈ Q+K+V+O) from VTune's GPU memory bandwidth.

### INV-NUM-002: Online softmax is numerically safe
**Rule**: Running max and sum are kept in FP32. Fully masked rows produce O = 0 and LSE = -inf, never NaN. No intermediate overflows for logits up to the FP16 range.
**Enforcement**: `tests/` includes fully-masked-row, very-large-logit and all-equal-logit cases. icpx builds are `-fp-model=precise` (its default fast model assumes finite math).

### INV-NUM-005: QK^T accumulates in FP32
**Rule**: The S = QK^T product accumulates in FP32 (`joint_matrix` float accumulator, `float8` DPAS accumulator). P and V may be FP16 or BF16.
**Rationale**: The CPU numerics model (`docs/research/error-budget-2026-10-09.txt`, from volta-fa) shows FP16 QK accumulation reaching 0.46 absolute / 11% relative output error with LLM-style outlier channels, against ~1e-3 with FP32 accumulation. The OpenCL DPAS builtin also has a `half8` accumulator overload. That overload is the FP16 path, and it is not allowed for QK.
**Enforcement**: design review; parity tests with the outlier distribution; the tolerances in `include/a770/tolerance.h` cannot be met with FP16 accumulation.

### INV-NUM-004: Causal alignment is bottom-right
**Rule**: With `causal`, query row `i` (0-based) may attend key `j` iff `j <= i + (seq_kv - seq_q)`. This matches a KV cache where the new queries are the last `seq_q` positions.
**Rationale**: Top-left alignment is the classic silent bug when `seq_q < seq_kv` (decode, chunked prefill).
**Enforcement**: reference + tests with `seq_q < seq_kv`.

### INV-NUM-003: Reference parity
**Rule**: Output matches an FP32 reference within the documented tolerance for every supported shape (head dim, GQA ratio, causal/non-causal, seq lens including non-multiples of the tile).
**Enforcement**: `ctest`. Tolerances live in `include/a770/tolerance.h` (element-wise `|o-ref| <= 1e-3 + 2e-3*|ref|`, LSE `1e-3`), derived from the CPU numerics model. They change only with a manager decision. `error_budget` fails if the kernel-equivalent model cannot meet them.

## Performance honesty invariants

### INV-PERF-001: No claim without an artifact
**Rule**: Every speedup/percentage in README, PRs or write-ups links to a committed file in `profiles/` or `bench/results/` with hardware (A770 8 GB or 16 GB, board vendor), driver/IGC, oneAPI version, kernel driver (i915 or xe), Resizable BAR state, commit and baseline commit recorded.
**Enforcement**: review checklist in REVIEW.md.

### INV-PERF-002: Compare against the real baseline
**Rule**: The baseline for a shape is the **faster** of llama.cpp's SYCL backend and Vulkan backend on the A770 at a pinned commit, with that backend's default FA dispatch. A naive kernel or FA-off is not a baseline.
**Rationale**: The two backends take different paths on Arc (SYCL: oneDNN/MKL for ≥ 32 tokens, otherwise the non-XMX TILE or VEC kernel; Vulkan: coopmat1 or scalar shaders), and either can be the faster one for a given shape. Beating the slower backend is not a result.
**Enforcement**: baseline commit pinned in `bench/BASELINE.md`; `bench/phase0.sh` measures both.

### INV-PERF-003: No spills by default
**Rule**: Shipped kernels have 0 bytes of spill unless a documented exception exists.
**Enforcement**: `bin/xe-report.py` on every kernel PR.

## Multi-GPU invariants

### INV-MGPU-001: Capability is checked, with a fallback
**Rule**: Query peer access (`zeDeviceCanAccessPeer` / `sycl::device::ext_oneapi_can_access_peer`) before using it; fall back to host-staged copies and log which path is active. Never assume the topology.
**Rationale**: Peer access between consumer Arc cards depends on the platform, kernel driver and BIOS (Resizable BAR, IOMMU). llama.cpp's own `GGML_SYCL_DEV2DEV_MEMCPY` documents crashes on the Level Zero path.
**Enforcement**: tests cover the fallback path via a forced-disable env var.

### INV-MGPU-002: Events, not device-wide waits
**Rule**: Communication overlaps compute through in-order queues and events; no `queue::wait()` / `zeCommandQueueSynchronize` of the whole device in the hot path.
**Enforcement**: review + unitrace/VTune timeline in `profiles/`.

### INV-MGPU-003: Buffer lifetime
**Rule**: Peer/staging buffers are freed only after every consuming queue has passed an event recorded after the last use.

## Agent-process invariants

### INV-PROC-001: Workers do not decide
**Rule**: Workers implement and report; they do not decide ship/no-ship or loosen invariants/tolerances. Stop conditions are honoured.

### INV-PROC-002: "Not run" is a valid result
**Rule**: Anything not executed on real A770 hardware is labelled "not run", never inferred.

## Change log

| Date | Invariant | Change | Justification |
| --- | --- | --- | --- |
| 2026-10-10 | all | Initial draft. NUM-*, PERF-001/003 and PROC-* carried over from volta-fa; ARCH-* rewritten for Xe-HPG from compile probes; PERF-002 takes the faster of two backends; MGPU-* restated for Level Zero | Repo created as volta-fa's sibling |
