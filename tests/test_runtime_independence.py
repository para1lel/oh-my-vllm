"""Independent execution must reject legacy imports, libraries and interpreters."""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from oh_my_vllm.worker.runtime import identity, verify_loaded_modules

ROOT = Path(__file__).resolve().parents[1]


class IndependenceTest(unittest.TestCase):
    def test_owned_executable_sources_never_import_vllm(self):
        for folder in ("python", "tests", "scripts", "benchmarks"):
            for path in (ROOT / folder).rglob("*.py"):
                # User-authorized isolated reference collection, never imported by
                # project execution/tests. Other benchmark code stays independent.
                if path == ROOT / "benchmarks/baseline/enginecore.py":
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    self.assertFalse(
                        any(
                            name == "vllm" or name.startswith("vllm.") for name in names
                        ),
                        str(path),
                    )

    def test_installed_legacy_package_rejected(self):
        with (
            patch("importlib.util.find_spec", return_value=object()),
            self.assertRaisesRegex(RuntimeError, "must not provide"),
        ):
            identity()

    def test_lazy_import_and_mapped_library_rejected(self):
        with (
            patch.dict(sys.modules, {"vllm.forbidden": object()}),
            self.assertRaisesRegex(RuntimeError, "was imported"),
        ):
            verify_loaded_modules()
        with (
            patch.object(
                Path,
                "read_text",
                return_value="0-1 r-xp 0 00:00 0 /env/vllm/lib/bad.so",
            ),
            self.assertRaisesRegex(RuntimeError, "legacy mapped library"),
        ):
            verify_loaded_modules()


if __name__ == "__main__":
    unittest.main()
