#!/usr/bin/env python3
"""Turn paired phase0 prof/<backend>-p<tokens>-d<depth>-r{1,N}.txt into breakdown.md.

Usage: python bin/step-breakdown.py bench/results/<run>/prof --reps 4
Only the standard library is required. Accepts unitrace's comma-separated
Device Timing Results table, with nanoseconds converted to microseconds.
"""
import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sys


# Ordered: fused/dequant GEMMs are matmul before the conversion fallback.
# oneMKL/oneDNN-generated GEMM names need not contain the library's name.
# These patterns MUST be checked against real kernel names on the A770 box.
# SYCL's oneMKL FA path (>= 32 tokens) appears as gemm -> matmul; it does not arise at p=8/16.
CATEGORY_PATTERNS = {
    "transfer": r"MemoryCopy|MemoryFill|memcpy|memset|Memcpy|Memset",
    "attention": r"flash_attn|fattn|sdpa",
    "matmul": r"mul_mat|mmvq|dmmv|mmq|fused|dequant_gemm|gemm",
    "conversion": r"dequantize|quantize|convert|cpy",
    "norm/elementwise": r"norm|softmax|soft_max|rope|silu|gelu|relu|scale|clamp|"
                        r"(?:^|[^a-z])(?:add|sub|mul|div|sigmoid|tanh)(?:[^a-z]|$)",
    "other": r".*",
}
PROFILE_NAME = re.compile(r"(?P<backend>[\w-]+)-p(?P<p>8|16)-d(?P<depth>0|16384)-r(?P<run>1|N)\.txt\Z")


@dataclass
class Kernel:
    name: str
    calls: Decimal
    total_ns: Decimal

    @property
    def category(self):
        return next(category for category, pattern in CATEGORY_PATTERNS.items()
                    if re.search(pattern, self.name, re.IGNORECASE))

    @property
    def counted_ns(self):
        return max(self.total_ns, Decimal(0))


def parse_summary(text):
    """Read only device tables, ignoring API/submission tables and their totals.

    unitrace prints aligned CSV; demangled names may contain quoted or unquoted
    template commas. Rejoin surplus name fields using the numeric header's width.
    Repeated kernel rows (e.g. separate device tables) are summed.
    """
    kernels = {}
    in_device = False
    header = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if "Device Timing Results" in line:
            in_device = True
            header = None
            continue
        if line.startswith("==="):
            in_device = False
            header = None
        if not in_device:
            continue
        if not line or "," not in line:
            header = None
            continue
        fields = [field.strip() for field in next(csv.reader([line], skipinitialspace=True))]
        if "Calls" in fields and "Time (ns)" in fields and ("Kernel" in fields or "Name" in fields):
            header = fields
            continue
        if header is None:
            continue
        name_index = header.index("Kernel" if "Kernel" in header else "Name")
        surplus = len(fields) - len(header)
        if surplus > 0:
            fields[name_index:name_index + surplus + 1] = [
                ", ".join(fields[name_index:name_index + surplus + 1])]
        if len(fields) != len(header):
            raise ValueError(f"line {number}: malformed device-timing row: {line}")
        try:
            name = fields[name_index]
            calls = int(fields[header.index("Calls")])
            total_ns = Decimal(fields[header.index("Time (ns)")])
            if not name or calls <= 0 or not total_ns.is_finite() or total_ns < 0:
                raise ValueError("invalid name, calls or time")
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(f"line {number}: invalid device-timing row: {line}") from exc
        if name in kernels:
            kernels[name].calls += calls
            kernels[name].total_ns += total_ns
        else:
            kernels[name] = Kernel(name, calls, total_ns)
    if not kernels:
        raise ValueError("no unitrace Device Timing Results kernel rows found (check unitrace's output format)")
    if sum(kernel.total_ns for kernel in kernels.values()) <= 0:
        raise ValueError("device-timing total must be positive")
    return sorted(kernels.values(), key=lambda kernel: (-kernel.total_ns, kernel.name))


def differential(baseline, repeated, reps):
    """Subtract the shared setup/warm-up/depth prefill, then average added repetitions."""
    if reps <= 0:
        raise ValueError("reps must be a positive integer")
    first = {kernel.name: kernel for kernel in baseline}
    later = {kernel.name: kernel for kernel in repeated}
    kernels = []
    for name in first.keys() | later.keys():
        before = first.get(name, Kernel(name, 0, Decimal(0)))
        after = later.get(name, Kernel(name, 0, Decimal(0)))
        calls = after.calls - before.calls
        if calls < 0:
            raise ValueError(f"negative call delta for kernel {name!r}: {after.calls} - {before.calls}")
        if calls > 0:
            kernels.append(Kernel(name, Decimal(calls) / reps,
                                  (after.total_ns - before.total_ns) / reps))
    return sorted(kernels, key=lambda kernel: (-kernel.counted_ns, kernel.name))


def cell(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "&#124;")


def calls_per_step(calls):
    return f"{calls:.2f}".rstrip("0").rstrip(".")


def share(time, total):
    return f"{time * 100 / total:.2f}%" if total else "0.00%"


def render(profiles, reps):
    lines = ["# Step breakdown", "",
             f"Per-step calls and device times are (rN − r1) / {reps}, using captures at "
             f"-r {1 + reps} and -r 1 with identical arguments and default warm-up. "
             "Shared setup, warm-up and one-off depth prefill cancel; shares use summed "
             "nonnegative compute device times, not wall-clock latency.", "",
             "Check category patterns against real kernel names on the A770 box.", ""]
    for backend, p, depth, paths, texts, kernels in profiles:
        compute = [kernel for kernel in kernels if kernel.category != "transfer"]
        total = sum(kernel.counted_ns for kernel in compute)
        lines += [f"## {backend}, p={p}, depth={depth}", "",
                  f"Sources: {paths[0].name}, {paths[1].name}", ""]
        if any(text.startswith("# SYNTHETIC") for text in texts):
            lines += ["**Synthetic fixture; these are not measured GPU timings.**", ""]
        lines += ["| Kernel | Category | Calls / step | µs / step | % of step device time |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for kernel in kernels:
            noise = " (noise)" if kernel.total_ns <= 0 else ""
            percent = "—" if kernel.category == "transfer" else share(kernel.counted_ns, total)
            lines.append(f"| {cell(kernel.name)}{noise} | {kernel.category} | "
                         f"{calls_per_step(kernel.calls)} | {kernel.total_ns / 1000:.3f} | {percent} |")
        lines += [f"| **Step total** | | {calls_per_step(sum(kernel.calls for kernel in compute))} | "
                  f"{total / 1000:.3f} | {share(total, total)} |", "",
                  "Transfer rows (including per-repetition KV state restore at depth > 0) "
                  "are excluded from the step total and all percentages because restore is not part "
                  "of a real decode step.", "",
                  "Rows marked (noise) have nonpositive time deltas and contribute 0 to totals and shares.", "",
                  "| Category | µs / step | % of step device time |", "| --- | ---: | ---: |"]
        for category in CATEGORY_PATTERNS:
            subtotal = sum(kernel.counted_ns for kernel in kernels if kernel.category == category)
            percent = "—" if category == "transfer" else share(subtotal, total)
            lines.append(f"| {category} | {subtotal / 1000:.3f} | {percent} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="phase0 output's prof directory")
    parser.add_argument("--reps", type=int, default=4, help="extra repetitions in rN (default: 4)")
    args = parser.parse_args()
    if args.reps <= 0:
        parser.error("--reps must be a positive integer")
    profiles = []
    try:
        pairs = {}
        for path in sorted(args.directory.glob("*.txt")):
            match = PROFILE_NAME.fullmatch(path.name)
            if not match:
                continue
            key = (match["backend"], int(match["p"]), int(match["depth"]))
            pairs.setdefault(key, {})[match["run"]] = path
        if not pairs:
            raise ValueError(f"{args.directory}: no <backend>-p<8|16>-d<0|16384>-r{{1,N}}.txt profiles found")
        for (backend, p, depth), pair in sorted(pairs.items()):
            for run in ("1", "N"):
                if run not in pair:
                    missing = args.directory / f"{backend}-p{p}-d{depth}-r{run}.txt"
                    raise ValueError(f"missing partner file: {missing}")
            paths = (pair["1"], pair["N"])
            texts = []
            summaries = []
            for path in paths:
                text = path.read_text(encoding="utf-8", errors="replace")
                texts.append(text)
                try:
                    summaries.append(parse_summary(text))
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
            try:
                kernels = differential(*summaries, args.reps)
            except ValueError as exc:
                raise ValueError(f"{paths[1]} minus {paths[0]}: {exc}") from exc
            profiles.append((backend, p, depth, paths, texts, kernels))
        output = args.directory / "breakdown.md"
        output.write_text(render(profiles, args.reps), encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"step-breakdown: {exc}\n")
    print(f"wrote {output} ({len(profiles)} profiles)")


if __name__ == "__main__":
    main()
