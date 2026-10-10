#!/usr/bin/env python3
"""Static Xe-HPG (Arc A770, acm-g10) resource report for a kernel file, no GPU needed.

ocloc + IGC (the same compiler the driver uses) compile the file ahead of time for acm-g10 with IGC
shader dumps on; we read per-kernel SIMD width, GRF mode, hardware threads per XVE, spill, private and SLM bytes from the
.zeinfo metadata and count DPAS (XMX) and FP64 instructions in the final assembly.

  .cl  : OpenCL C, compiled by ocloc directly (tools/neo, bin/setup-dev.sh)
  .cpp : SYCL, compiled by icpx 2026.1 -fsycl-targets=spir64_gen (WITH_DPCPP=1 bin/setup-dev.sh)

Exit 1 if any kernel spills (INV-PERF-003), uses > 64 KB SLM (INV-ARCH-002), executes FP64, runs DPAS
at a SIMD width other than 8, or if XMX was requested but no DPAS was emitted (INV-ARCH-001: on
acm-g10 the SIMD16 OpenCL DPAS builtin compiles WITHOUT error and WITHOUT a DPAS instruction).
Exit 2 if the compile fails.

Usage: .venv/bin/python bin/xe-report.py src/attention/foo.cpp [-D NAME=VAL ...] [--large-grf]
           [--expect-dpas REGEX] [--keep DIR]
Templates are only compiled when instantiated: add explicit instantiations in the file under test.
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NEO = ROOT / "tools/neo"
ONEAPI = ROOT / "tools/oneapi-2026.1"
DEVICE = "acm-g10"
SLM_LIMIT = 64 * 1024
# Source text that means "this file wants XMX"; if it matches and no kernel emits DPAS, that is a failure.
XMX_REQUEST = re.compile(r"joint_matrix_mad|matrix_mad_k\d+|\bxmx::dpas|__builtin_IB_\w*dpas")
FIELDS = ("simd_size", "grf_count", "eu_thread_count", "spill_size", "private_size", "slm_size")


def tool_env() -> dict:
    ocloc = next(NEO.glob("usr/bin/ocloc*"), None)
    if ocloc is None:
        sys.exit("no tools/neo (ocloc + IGC); run bin/setup-dev.sh (docs/TOOLCHAIN.md)")
    env = dict(os.environ)
    libs = [str(NEO / "usr/lib/x86_64-linux-gnu"), str(NEO / "usr/local/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = str(NEO / "bin") + ":" + env.get("PATH", "")
    env["IGC_ShaderDumpEnable"] = "1"
    return env


def compile_kernels(src: pathlib.Path, defines: list[str], large_grf: bool, dump: pathlib.Path, env: dict):
    env = dict(env, IGC_DumpToCustomDir=str(dump))
    incs = [f"-I{ROOT / 'include'}"] + [f"-D{d}" for d in defines]
    if src.suffix == ".cl":
        opts = incs + (["-cl-intel-256-GRF-per-thread"] if large_grf else [])
        cmd = [str(NEO / "bin/ocloc"), "compile", "-file", str(src), "-device", DEVICE,
               "-out_dir", str(dump / "out"), "-options", " ".join(opts)]
    elif src.suffix in (".cpp", ".cc"):
        icpx = ONEAPI / "bin/icpx"
        if not icpx.exists():
            sys.exit("no tools/oneapi-2026.1; run WITH_DPCPP=1 bin/setup-dev.sh (docs/TOOLCHAIN.md)")
        backend = f"-device {DEVICE}" + (" -options -ze-opt-large-register-file" if large_grf else "")
        # -shared so the device link (where AOT happens) runs without a main().
        cmd = [str(icpx), "-fsycl", "-fsycl-targets=spir64_gen", "-Xsycl-target-backend", backend,
               "-std=c++17", "-O2", "-shared", "-fPIC", *incs, str(src), "-o", str(dump / "out.so")]
    else:
        sys.exit(f"unsupported source type {src.suffix} (.cl or .cpp)")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = (r.stdout + r.stderr).strip()
    return r.returncode, log


def parse_zeinfo(dump: pathlib.Path) -> list[dict]:
    kernels = []
    for z in sorted(dump.glob("*.zeinfo")):
        section, k = None, None
        for line in z.read_text().splitlines():
            if re.match(r"^\S", line):
                section, k = line.rstrip(":"), None
                continue
            if section != "kernels":
                continue
            m = re.match(r"^  - name:\s+(\S+)", line)
            if m:
                k = {"name": m.group(1), **{f: 0 for f in FIELDS}}
                kernels.append(k)
                continue
            m = re.match(r"^\s+(" + "|".join(FIELDS) + r"):\s+(\d+)", line)
            if k is not None and m:
                k[m.group(1)] = int(m.group(2))
    return kernels


def asm_counts(dump: pathlib.Path) -> dict:
    """(kernel name, simd) -> (dpas, fp64) instruction counts from the final ISA dumps."""
    out = {}
    for a in dump.glob("*_simd*_entry_*.asm"):
        simd = int(re.search(r"_simd(\d+)_", a.name).group(1))
        name, dpas, df = None, 0, 0
        for line in a.read_text(errors="replace").splitlines():
            if line.startswith("//"):
                m = re.match(r"//\.kernel (\S+)", line)
                if m:
                    name = m.group(1)
                continue
            code = line.split("//")[0]
            dpas += bool(re.search(r"\bdpasw?\b", code))
            df += ":df" in code
        if name:
            out[(name, simd)] = (dpas, df)
    return out


def pretty(name: str) -> str:
    r = subprocess.run(["c++filt"], input=name, capture_output=True, text=True) if shutil.which("c++filt") else None
    s = r.stdout.strip() if r and r.returncode == 0 else name
    return s.removeprefix("typeinfo name for ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--large-grf", action="store_true", help="256-GRF mode (halves threads per XVE)")
    ap.add_argument("--expect-dpas", metavar="REGEX", help="kernels whose name matches must emit DPAS")
    ap.add_argument("--keep", type=pathlib.Path, help="copy the IGC dump (asm, zeinfo) here")
    a = ap.parse_args()

    env = tool_env()
    with tempfile.TemporaryDirectory() as td:
        dump = pathlib.Path(td)
        rc, log = compile_kernels(a.src.resolve(), a.defines, a.large_grf, dump, env)
        # IGC reports SIMD retries and spill estimates as warnings; they matter, so always show them.
        for line in log.splitlines():
            if re.search(r"error|warning|spill", line, re.I):
                print(line, file=sys.stderr)
        if rc:
            print(f"compile for {DEVICE} failed (rc={rc})", file=sys.stderr)
            return 2
        kernels = parse_zeinfo(dump)
        counts = asm_counts(dump)
        if a.keep:
            shutil.copytree(dump, a.keep, dirs_exist_ok=True)

    if not kernels:
        print("no kernels found (templates need explicit instantiation)", file=sys.stderr)
        return 2
    bad = False
    want_xmx = bool(XMX_REQUEST.search(a.src.read_text(errors="replace")))
    expect = re.compile(a.expect_dpas) if a.expect_dpas else None
    print(f"device {DEVICE}  {'kernel':56} {'simd':>4} {'grf':>4} {'thr':>3} {'spill':>6} {'priv':>7} {'slm':>6} {'dpas':>4} {'fp64':>4}")
    total_dpas = 0
    for k in kernels:
        dpas, df = counts.get((k["name"], k["simd_size"]), (0, 0))
        total_dpas += dpas
        name = pretty(k["name"])
        flags = []
        if k["spill_size"]:
            flags.append("SPILL")
        if k["slm_size"] > SLM_LIMIT:
            flags.append("SLM>64K")
        if df:
            flags.append("FP64")
        if dpas and k["simd_size"] != 8:
            flags.append("DPAS@SIMD!=8")
        if expect and (expect.search(name) or expect.search(k["name"])) and not dpas:
            flags.append("NO-DPAS")
        bad |= bool(flags)
        if k["private_size"]:
            flags.append("(private mem)")
        print(f"               {name[:56]:56} {k['simd_size']:>4} {k['grf_count']:>4} {k['eu_thread_count']:>3} {k['spill_size']:>6} "
              f"{k['private_size']:>7} {k['slm_size']:>6} {dpas:>4} {df:>4} {' '.join(flags)}")
    if want_xmx and total_dpas == 0:
        print("NO-DPAS: the source requests XMX but no kernel emitted a DPAS instruction "
              "(SIMD16 DPAS forms are silently dropped on acm-g10; use sub-group 8, INV-ARCH-001)")
        bad = True
    print("note: spill/private/slm in bytes as reported by IGC; slm is static only, dynamic SLM is budgeted in the design doc.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
