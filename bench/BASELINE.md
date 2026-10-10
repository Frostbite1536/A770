# Baseline (Phase 0)

Not run yet. Fill in before any kernel or patch work (INV-PERF-002: the faster of SYCL and Vulkan per shape).

| Field | Value |
| --- | --- |
| llama.cpp commit | `79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6` (2026-10-09; also in `bench/LLAMA_COMMIT`, same pin as volta-fa) |
| GPU | (A770 8 GB or 16 GB, board vendor, power limit) |
| Driver / IGC / oneAPI | (from env.txt) |
| Kernel driver / kernel | (i915 or xe; uname -r) |
| Resizable BAR | (on/off, BAR size) |
| SLM per work-group (sycl-ls) | (confirms INV-ARCH-002) |
| Vulkan driver + coopmat | (Mesa version; VK_KHR_cooperative_matrix yes/no) |
| Measured streaming bandwidth | (from F16 decode at depth; the roofline uses this, not the datasheet) |
| Models + quant | |
| Flags | |

Results: link files in `bench/results/` and `profiles/`.
