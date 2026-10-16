#!/usr/bin/env python3
"""Check synthetic round medians, byte counts, roofline and missing data."""
import importlib.util
import pathlib
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parents[2] / "bin/mm-gbps.py"
spec = importlib.util.spec_from_file_location("mm_gbps", SCRIPT)
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


class MmGbps(unittest.TestCase):
    def test_rounds_and_labels(self):
        result = run(f"{HERE / 'r1.sql'},{HERE / 'r2.sql'}", HERE / "r2.sql",
                     "--labels", "sycl,vulkan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Roofline: 560 GB/s", result.stdout)
        self.assertIn("n=1 µs | n=1 GB/s | n=1 % peak", result.stdout)
        self.assertIn("n=16 µs | n=16 GB/s | n=16 % peak", result.stdout)
        # 9,437,184 weight bytes + 32,768 activation/output bytes, divided by 20 us.
        self.assertIn("| Q4_K | 4096 | 4096 | sycl | 20.000 | 473.498 | 84.55% | "
                      "50.000 | 199.229 | 35.58% |", result.stdout)
        self.assertIn("| Q4_K | 4096 | 4096 | vulkan | 10.000 | 946.995 | 169.11% | "
                      "40.000 | 249.037 | 44.47% |", result.stdout)
        self.assertIn("| Q8_0 | 1024 | 4096 | sycl | 40.000 | 111.923 | 19.99% | "
                      "not measured | not measured | not measured |", result.stdout)
        self.assertIn("| Q6_K | 14336 | 4096 | sycl | not measured | not measured | "
                      "not measured | unsupported | unsupported | unsupported |", result.stdout)
        self.assertIn("15.000 | unknown type | unknown type", result.stdout)
        self.assertNotIn("ADD", result.stdout)

    def test_byte_counts(self):
        expected = {"Q4_0": 18, "Q8_0": 34, "Q4_K": 18, "Q5_K": 22, "Q6_K": 26.25}
        for kind, bytes_per_32 in expected.items():
            with self.subTest(kind=kind):
                # m=2, k=256, n=16: weights plus (256*16 + 2*16)*4 bytes.
                self.assertEqual(mm.byte_count(kind, 2, 256, 16),
                                 bytes_per_32 * 16 + 16512)
        self.assertIsNone(mm.byte_count("unknown", 2, 256, 1))
        with self.assertRaises(ValueError):
            mm.byte_count("Q4_K", 2, 32, 1)

    def test_peak_override(self):
        result = run(HERE / "r2.sql", "--peak-gbps", "1000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Roofline: 1000 GB/s", result.stdout)
        self.assertIn("10.000 | 946.995 | 94.70%", result.stdout)

    def test_invalid_inputs(self):
        for args in (("--labels", "a,b"), ("--peak-gbps", "0"),
                     ("--peak-gbps", "nan"), ("--peak-gbps", "-1")):
            with self.subTest(args=args):
                self.assertNotEqual(run(HERE / "r1.sql", *args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
