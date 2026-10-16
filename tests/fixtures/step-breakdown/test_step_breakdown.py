#!/usr/bin/env python3
"""Synthetic parser assertions and phase0 orchestration checks; no GPU or models."""
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPT = ROOT / "bin/step-breakdown.py"
MODULE = runpy.run_path(str(SCRIPT))
PARSE = MODULE["parse_summary"]
DIFFERENTIAL = MODULE["differential"]
FIXTURES = tuple((HERE / f"sycl-p8-d0-r{run}.txt").read_text() for run in ("1", "N"))


def fixture_steps(reps=4):
    return DIFFERENTIAL(*(PARSE(text) for text in FIXTURES), reps)


def fixture_report(texts=FIXTURES, reps=4):
    kernels = DIFFERENTIAL(*(PARSE(text) for text in texts), reps)
    paths = tuple(HERE / f"sycl-p8-d0-r{run}.txt" for run in ("1", "N"))
    return MODULE["render"]([("sycl", 8, 0, paths, texts, kernels)], reps)


class BreakdownTest(unittest.TestCase):
    def test_fixture_categories_and_percentages(self):
        for text in FIXTURES:
            self.assertTrue(text.startswith("# SYNTHETIC"))
        kernels = fixture_steps()
        self.assertEqual(len(kernels), 20)
        self.assertEqual(sum(kernel.calls for kernel in kernels), Decimal('20.25'))
        total = sum(kernel.counted_ns for kernel in kernels if kernel.category != "transfer")
        self.assertEqual(total, Decimal(1000000))
        expected = {"matmul": 60, "attention": 25, "conversion": 5,
                    "norm/elementwise": 5, "other": 5}
        for category, percent in expected.items():
            subtotal = sum(kernel.counted_ns for kernel in kernels if kernel.category == category)
            self.assertEqual(subtotal * 100 / total, percent, category)
        names = {kernel.name: kernel for kernel in kernels}
        self.assertNotIn("one_off_prefill_mul_mat", names)
        # Every retained row: expected (calls per step, microseconds per step).
        expected_rows = {
            "void mul_mat<half, float>(int)": (2, 300),
            "kernel_mmvq_q4_0": (1, 50), "kernel_dmmv_q4_0": (1, 50),
            "kernel_mmq_q4_0": (1, 50), "fused_dequant_gemm_q4_0": (1, 50),
            "oneMKL_gemm_half": (1, 50), "oneDNN_gemm_float": (1, 50),
            "flash_attn_ext<half, float, 8>": (1, 150),
            "fattn_tile_half": (1, 50), "sdpa_forward": (1, 50),
            "dequantize_q4_0": (1, 30), "convert_half": (1, 5), "cpy_f16": (1, 5),
            "quantize_q8_1": (1, 10), "rms_norm": (1, 30), "kernel_add": (1, 10),
            "soft_max": (1, 10), "unknown_kernel": (1, 50),
            "noisy_kernel": (Decimal('0.25'), -1), "MemoryCopy": (1, 1000),
        }
        self.assertEqual(set(names), set(expected_rows))
        for name, (calls, microseconds) in expected_rows.items():
            self.assertEqual((names[name].calls, names[name].total_ns / 1000),
                             (calls, microseconds), name)
        self.assertEqual(names["void mul_mat<half, float>(int)"].total_ns, 300000)
        self.assertEqual(names["void mul_mat<half, float>(int)"].calls, 2)
        self.assertEqual(names["flash_attn_ext<half, float, 8>"].category, "attention")
        self.assertEqual(names["flash_attn_ext<half, float, 8>"].total_ns, 150000)
        for name in ("kernel_mmvq_q4_0", "kernel_dmmv_q4_0", "kernel_mmq_q4_0",
                     "fused_dequant_gemm_q4_0", "oneMKL_gemm_half", "oneDNN_gemm_float"):
            self.assertEqual(names[name].category, "matmul", name)
        for name in ("dequantize_q4_0", "convert_half", "cpy_f16", "quantize_q8_1"):
            self.assertEqual(names[name].category, "conversion", name)
        self.assertEqual(names["soft_max"].category, "norm/elementwise")
        self.assertEqual(names["MemoryCopy"].category, "transfer")
        self.assertEqual(names["MemoryCopy"].calls, 1)
        self.assertEqual(names["MemoryCopy"].total_ns, 1000000)
        self.assertEqual(names["noisy_kernel"].calls, Decimal('0.25'))
        self.assertEqual(names["noisy_kernel"].total_ns, -1000)
        self.assertEqual(names["noisy_kernel"].counted_ns, 0)

    def test_percentages_are_recomputed(self):
        # Deliberately corrupt source Time (%) columns: report shares use only time deltas.
        texts = tuple(re.sub(r",\d+\.\d{2},", ",99.99,", text) for text in FIXTURES)
        rendered = fixture_report(texts)
        self.assertIn("| 2 | 300.000 | 30.00% |", rendered)
        self.assertIn("| matmul | 600.000 | 60.00% |", rendered)
        self.assertIn("| attention | 250.000 | 25.00% |", rendered)
        self.assertIn("| conversion | 50.000 | 5.00% |", rendered)
        self.assertIn("| norm/elementwise | 50.000 | 5.00% |", rendered)
        self.assertIn("| other | 50.000 | 5.00% |", rendered)
        self.assertIn("| MemoryCopy | transfer | 1 | 1000.000 | — |", rendered)
        self.assertIn("| transfer | 1000.000 | — |", rendered)
        self.assertIn("| **Step total** | | 19.25 | 1000.000 | 100.00% |", rendered)
        self.assertIn("| noisy_kernel (noise) | other | 0.25 | -1.000 | 0.00% |", rendered)
        self.assertNotIn("one_off_prefill_mul_mat", rendered)
        self.assertIn("KV state restore", rendered)
        self.assertIn("excluded from the step total and all percentages", rendered)
        self.assertIn("Synthetic fixture", rendered)
        self.assertIn("mul_mat&lt;half, float&gt;", rendered)

    def test_multiple_device_tables_aggregate(self):
        original = PARSE(FIXTURES[0])
        kernels = PARSE(FIXTURES[0] + FIXTURES[0])
        self.assertEqual(len(kernels), 21)
        for before, after in zip(original, kernels):
            self.assertEqual(after.calls, 2 * before.calls)
            self.assertEqual(after.total_ns, 2 * before.total_ns)

    def test_name_header_and_column_order(self):
        kernels = PARSE("=== Device Timing Results: ===\n"
                        "Calls, Time (ns), Name\n2, 123000, custom_kernel\n")
        self.assertEqual(kernels[0].name, "custom_kernel")
        self.assertEqual(kernels[0].total_ns, 123000)

    def test_unquoted_template_commas(self):
        kernels = PARSE('=== Device Timing Results: ===\nKernel, Calls, Time (ns)\n'
                        'flash_attn_ext<half, float, 8>, 2, 123000\n')
        self.assertEqual(kernels[0].name, 'flash_attn_ext<half, float, 8>')
        self.assertEqual(kernels[0].calls, 2)

    def test_gemm_names_and_fused_attention_priority(self):
        kernel = MODULE["Kernel"]
        for name in ("gen_gemm_kernel", "jit_brgemm", "oneapi::mkl::gemm", "dnnl_gemm"):
            self.assertEqual(kernel(name, 1, Decimal(1)).category, "matmul", name)
        self.assertEqual(kernel("fused_sdpa_forward", 1, Decimal(1)).category, "attention")
        for name in ("MemoryCopy", "MemoryFill", "memcpy", "memset", "Memcpy", "Memset",
                     "MemoryCopy_mul_mat", "MemoryFill_sdpa", "memcpy_convert"):
            self.assertEqual(kernel(name, 1, Decimal(1)).category, "transfer", name)

    def test_differential_missing_kernel_and_negative_calls(self):
        kernel = MODULE["Kernel"]
        new = kernel("new_kernel", 5, Decimal(100000))
        self.assertEqual(DIFFERENTIAL([], [new], 4)[0].calls, Decimal('1.25'))
        for after in ([], [kernel("new_kernel", 4, Decimal(100000))]):
            with self.subTest(after=after), self.assertRaisesRegex(ValueError, "negative call delta.*new_kernel"):
                DIFFERENTIAL([new], after, 4)
        # Zero call delta is omitted even if the observed time changes.
        self.assertEqual(DIFFERENTIAL([new], [kernel("new_kernel", 5, Decimal(90000))], 4), [])

    def test_zero_time_delta_is_noise(self):
        kernel = MODULE["Kernel"]
        rows = DIFFERENTIAL([kernel("zero_delta", 1, Decimal(100))],
                            [kernel("zero_delta", 5, Decimal(100))], 4)
        report = MODULE["render"]([("sycl", 8, 0, (Path("r1"), Path("rN")), ("", ""), rows)], 4)
        self.assertIn("| zero_delta (noise) | other | 1 | 0.000 | 0.00% |", report)
        self.assertIn("| **Step total** | | 1 | 0.000 | 0.00% |", report)

    def test_missing_and_invalid_summaries_fail(self):
        for text in ("", "Kernel, Calls, Time (ns)\nx, 1, 100",
                     "=== API Timing Results: ===\nKernel, Calls, Time (ns)\nx, 1, 100"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                PARSE(text)
        for row in ("x, 1", "x, bad, 100", "x, 1, NaN", "x, 1, Infinity",
                    "x, 0, 100", "x, 1, -1", "x, 1, 0"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                PARSE("=== Device Timing Results: ===\nKernel, Calls, Time (ns)\n" + row)

    def test_cli_groups_and_ignores_help(self):
        with tempfile.TemporaryDirectory() as directory:
            prof = Path(directory)
            for backend in ("sycl", "vulkan"):
                for p in (8, 16):
                    for depth in (0, 16384):
                        for run, text in zip(("1", "N"), FIXTURES):
                            (prof / f"{backend}-p{p}-d{depth}-r{run}.txt").write_text(text)
            (prof / "unitrace-help.txt").write_text("--device-timing")
            result = subprocess.run([sys.executable, str(SCRIPT), directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (prof / "breakdown.md").read_text()
            self.assertEqual(report.count("## "), 8)
            self.assertIn("## vulkan, p=16, depth=16384", report)
            self.assertEqual(report.count("| attention | 250.000 | 25.00% |"), 8)
            (prof / "sycl-p8-d0-r1.txt").write_text("unrecognized format")
            result = subprocess.run([sys.executable, str(SCRIPT), directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sycl-p8-d0-r1.txt", result.stderr)

    def test_missing_partner_names_file(self):
        for present, missing, text in (("1", "N", FIXTURES[0]), ("N", "1", FIXTURES[1])):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                prof = Path(directory)
                (prof / f"sycl-p8-d0-r{present}.txt").write_text(text)
                result = subprocess.run([sys.executable, str(SCRIPT), directory], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"missing partner file: {prof / f'sycl-p8-d0-r{missing}.txt'}", result.stderr)
                self.assertFalse((prof / "breakdown.md").exists())

    def test_cli_reps_override_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            prof = Path(directory)
            for run, text in zip(("1", "N"), FIXTURES):
                (prof / f"sycl-p8-d0-r{run}.txt").write_text(text)
            for reps, expected in (("2", "600.000"), ("8", "150.000")):
                result = subprocess.run([sys.executable, str(SCRIPT), directory, "--reps", reps],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"| {expected} | 30.00% |", (prof / "breakdown.md").read_text())
            for reps in ("0", "-1", "bad"):
                result = subprocess.run([sys.executable, str(SCRIPT), directory, "--reps", reps],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)


class Phase0Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "bench").mkdir()
        (self.root / "bin").mkdir()
        self.tools = self.root / "tools"
        self.tools.mkdir()
        shutil.copy(ROOT / "bench/phase0.sh", self.root / "bench/phase0.sh")
        shutil.copy(SCRIPT, self.root / "bin/step-breakdown.py")
        self.env = dict(os.environ, PATH=f"{self.tools}:{os.environ['PATH']}",
                        STEPS="prof", BACKENDS="sycl vulkan", LLAMA="third_party/llama.cpp")
        self.env.pop("PROF_REPS", None)

    def executable(self, name, source):
        path = self.tools / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)

    def run_phase0(self):
        return subprocess.run(["bash", "bench/phase0.sh"], cwd=self.root, env=self.env,
                              capture_output=True, text=True)

    def prepare(self, help_text="--device-timing --kernel-submission", status=0):
        (self.root / "bench/models.txt").write_text("# comment\n\n  # indented comment\n/model one.gguf\n/second.gguf\n")
        self.executable("sycl-ls", "print('[level_zero:gpu:0] Intel Arc A770')\n")
        self.executable("icpx", "print('Intel oneAPI Compiler 2026.1.0')\n")
        self.executable("unitrace", f"""import json, pathlib, sys
if sys.argv[1:] == ['--help']:
    print({help_text!r})
    sys.exit(0)
with pathlib.Path('calls.jsonl').open('a') as output:
    output.write(json.dumps(sys.argv[1:]) + '\\n')
print({FIXTURES[0]!r} if sys.argv[sys.argv.index('-r') + 1] == '1' else {FIXTURES[1]!r})
sys.exit({status})
""")

    def test_missing_models_skips_before_gpu_checks(self):
        result = self.run_phase0()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no bench/models.txt; skipping prof", result.stdout)
        self.assertEqual(len(result.stdout.splitlines()), 1)

    def test_missing_unitrace_skips_before_gpu_checks(self):
        (self.root / "bench/models.txt").write_text("/model.gguf\n")
        # An isolated PATH exposes only utilities needed before the skip, never a host unitrace.
        for name in ("dirname", "basename", "date", "hostname", "mkdir", "tee"):
            (self.tools / name).symlink_to(shutil.which(name))
        self.env["PATH"] = str(self.tools)
        result = subprocess.run([shutil.which("bash"), "bench/phase0.sh"], cwd=self.root,
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unitrace not found on PATH; skipping prof", result.stdout)

    def test_profile_grid_and_first_model(self):
        self.prepare()
        result = self.run_phase0()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 16)
        actual = set()
        for call in calls:
            self.assertEqual(call[:2], ["--device-timing", "--kernel-submission"])
            flags = dict(zip(call[3::2], call[4::2]))
            for flag, value in {"-m": "/model one.gguf", "-ngl": "99", "-fa": "1", "-n": "0"}.items():
                self.assertEqual(flags[flag], value)
            self.assertNotIn("--no-warmup", call)
            backend = "sycl" if flags["-dev"] == "SYCL0" else "vulkan"
            self.assertEqual(call[2], f"third_party/llama.cpp/build-{backend}/bin/llama-bench")
            actual.add((backend, flags["-p"], flags["-d"], flags["-r"]))
        self.assertEqual(actual, {(be, p, d, r) for be in ("sycl", "vulkan")
                                 for p in ("8", "16") for d in ("0", "16384") for r in ("1", "5")})
        for first, later in zip(calls[::2], calls[1::2]):
            self.assertEqual(first[:-1], later[:-1])
            self.assertEqual((first[-1], later[-1]), ("1", "5"))
        prof = next((self.root / "bench/results").glob("*/prof"))
        self.assertTrue((prof / "unitrace-help.txt").exists())
        self.assertEqual(len(list(prof.glob("*-p*-d*.txt"))), 16)
        self.assertEqual((prof / "breakdown.md").read_text().count("## "), 8)

    def test_custom_prof_reps_passed_to_capture_and_parser(self):
        self.prepare()
        self.env["PROF_REPS"] = "2"
        result = self.run_phase0()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 16)
        self.assertEqual([call[-1] for call in calls], ["1", "3"] * 8)
        prof = next((self.root / "bench/results").glob("*/prof"))
        report = (prof / "breakdown.md").read_text()
        self.assertIn("(rN − r1) / 2", report)
        self.assertIn("| matmul | 1200.000 | 60.00% |", report)

    def test_invalid_prof_reps_fails(self):
        self.prepare()
        self.env["PROF_REPS"] = "0"
        result = self.run_phase0()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PROF_REPS must be a positive integer", result.stdout)
        self.assertFalse((self.root / "calls.jsonl").exists())

    def test_missing_device_timing_flag_fails(self):
        self.prepare(help_text="--kernel-submission")
        result = self.run_phase0()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unitrace --help lacks --device-timing", result.stdout)
        self.assertFalse((self.root / "calls.jsonl").exists())

    def test_profile_failure_is_reported(self):
        self.prepare(status=9)
        result = self.run_phase0()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prof failed (sycl, p=8, depth=0", result.stdout)
        self.assertFalse(list((self.root / "bench/results").glob("*/prof/breakdown.md")))


if __name__ == "__main__":
    unittest.main()
