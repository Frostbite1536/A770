#!/usr/bin/env python3
"""ctest driver: bin/xe-report.py must keep catching the acm-g10 traps it exists for.

Each case compiles a fixture in tests/xe/ and checks the exit code and the flagged line. Exit 77 (skip)
if the static toolchain is missing (bin/setup-dev.sh; WITH_DPCPP=1 for the SYCL case).
  dpas_simd8          legal 8x8x16 DPAS at sub-group 8: passes, 1 DPAS, 8 threads/XVE at 128 GRF
  large_grf           256-GRF mode halves threads per XVE (8 -> 4): the occupancy cost must be visible
  dpas_simd16_silent  the Xe2 form compiles cleanly on acm-g10 with no DPAS: must fail NO-DPAS
  spill               register pressure beyond 128 GRFs: must fail SPILL (INV-PERF-003)
  fp64                no FP64 on the A770: compile must fail
  block2d             2D block loads are Xe-HPC/Xe2 only: compile must fail
  slm                 64 KB passes; 96 KB compiles but must fail SLM>64K (IGC does not enforce it)
  sycl_joint_matrix   SYCL joint_matrix 8x8x16 at sub-group 8: passes with DPAS
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
XE = ROOT / "tests/xe"


def report(src: str, *args: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(ROOT / "bin/xe-report.py"), str(XE / src), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def expect(rc: int, out: str, want_rc: int, pattern: str) -> None:
    if rc != want_rc or not re.search(pattern, out, re.M):
        print(out)
        sys.exit(f"FAIL: rc={rc} (want {want_rc}), pattern {pattern!r} {'found' if re.search(pattern, out, re.M) else 'missing'}")


def main() -> int:
    case = sys.argv[1]
    if not (ROOT / "tools/neo/bin/ocloc").exists():
        print("skip: no tools/neo (bin/setup-dev.sh)")
        return 77
    if case == "dpas_simd8":
        expect(*report("dpas_simd8.cl", "--expect-dpas", "dpas"), 0, r"dpas_simd8\s+8\s+128\s+8\s+0\s+0\s+0\s+1\s+0\s*$")
    elif case == "large_grf":
        expect(*report("dpas_simd8.cl", "--large-grf"), 0, r"dpas_simd8\s+8\s+256\s+4\s")
    elif case == "dpas_simd16_silent":
        expect(*report("dpas_simd16.cl"), 1, r"^NO-DPAS")
    elif case == "spill":
        expect(*report("spill.cl"), 1, r"spill\s+16\s+128\s+\d+\s+[1-9]\d*.*SPILL")
    elif case == "fp64":
        expect(*report("fp64.cl"), 2, r"requires cl_khr_fp64")
    elif case == "block2d":
        expect(*report("block2d.cl"), 2, r"undeclared identifier 'intel_sub_group_2d_block_read")
    elif case == "slm":
        expect(*report("slm.cl"), 0, r"slm\s+\d+\s+\d+\s+\d+\s+0\s+0\s+65536\s")
        expect(*report("slm.cl", "-D", "SLM_FLOATS=24576"), 1, r"98304.*SLM>64K")
    elif case == "sycl_joint_matrix":
        if not (ROOT / "tools/oneapi-2026.1/bin/icpx").exists():
            print("skip: no tools/oneapi-2026.1 (WITH_DPCPP=1 bin/setup-dev.sh)")
            return 77
        expect(*report("joint_matrix.cpp", "--expect-dpas", "mm<8, 8, 8, 16>"), 0, r"mm<8, 8, 8, 16>.*\s8\s+128\s+8\s+0\s+0\s+0\s+[1-9]")
    else:
        sys.exit(f"unknown case {case}")
    print(f"PASS {case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
