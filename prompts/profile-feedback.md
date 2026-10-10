# Profile feedback prompt

Paste into the manager (or into a ticket Background section).

```
Kernel: <name>, global <range>, local <range>, sub-group <n>, GRF <128|256>, SLM/work-group <bytes>
xe-report: simd=<n> grf=<n> thr/XVE=<n> spill=<n> private=<n> slm=<n> dpas=<n>
VTune: XVE active <x>% / stalled <y>% / idle <z>%, threads occupancy <x>% (limiter: <GRF|SLM|work-group size>),
       memory bandwidth <x> GB/s (measured streaming peak <y> GB/s), hot lines <file:line ...>
Baseline: <backend> <commit> <metric>.

Constraints: Xe-HPG acm-g10; XMX via SIMD8 DPAS only (no SIMD16 DPAS, 2D block loads, TF32, FP8, FP64). Invariants: docs/INVARIANTS.md.
Task: propose ONE change targeting the largest measured limiter (e.g. 256-GRF mode to remove spills at half occupancy, a smaller Q tile to fit 128 GRF, padding an SLM row to break conflicts). State the expected effect on each metric above so the next run can falsify it.
```

One change per iteration, so the next profile attributes the effect.
