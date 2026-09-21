"""Regression checks for lines rustfmt can leave untouched inside Rust macros."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_rust_line_width.py"


class RustStyleTests(unittest.TestCase):
    def check(self, text):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "macro.rs"
            source.write_text(text)
            return subprocess.run(
                [sys.executable, str(CHECKER), str(source)],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_exact_limit_passes_and_overflow_fails(self):
        self.assertEqual(self.check("//" + "x" * 98 + "\n").returncode, 0)
        failed = self.check("//" + "x" * 99 + "\n")
        self.assertEqual(failed.returncode, 1)
        self.assertIn("macro.rs:1: 101 columns", failed.stderr)

    def test_json_macro_and_literal_are_not_exempt(self):
        failed = self.check('json!({"value": "' + "x" * 100 + '"});\n')
        self.assertEqual(failed.returncode, 1)
        self.assertIn("max_width=100", failed.stderr)

    def test_tabs_and_crlf(self):
        self.assertEqual(self.check("\t//" + "x" * 94 + "\r\n").returncode, 0)
        self.assertEqual(self.check("\t//" + "x" * 95 + "\r\n").returncode, 1)

    def test_multiline_unicode_macro_passes(self):
        self.assertEqual(
            self.check('json!({\n    "message": "你好",\n});\n').returncode, 0
        )

    def test_unicode_separator_does_not_hide_physical_overflow(self):
        value = "x" * 50 + "\u2028" + "x" * 50
        failed = self.check('json!({"value": "' + value + '"});\n')
        self.assertEqual(failed.returncode, 1)
        self.assertIn("macro.rs:1:", failed.stderr)


if __name__ == "__main__":
    unittest.main()
