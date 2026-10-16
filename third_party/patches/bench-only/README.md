# Bench-only patches

These local benchmark patches are **never upstream** and are separate from the
kernel series and CPU-only `test-only/` patches. Phase 0 applies only its explicit
every-build list, once to the shared llama.cpp tree before building either backend.

| Patch | Provenance | Purpose | Builds |
|---|---|---|---|
| `0001-tests-mul_mat-perf-cases-1-16.patch` | A770 ticket p12, against llama.cpp `79e2e74eb`; same cases as P100 ticket p13 | Complete contiguous F32-RHS MUL_MAT perf coverage for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K at four Llama shapes and n=1–16; skip existing cases (280 additions, 320 total). Eval cases and tolerances unchanged. **Never upstream.** | Every build: SYCL and Vulkan, all timing arms |
