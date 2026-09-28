"""Steady-state collection must reject new compilation and graph capture."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "benchmarks/measurement.py"
spec = importlib.util.spec_from_file_location("measurement_audit", path)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class AuditTest(unittest.TestCase):
    def test_capture_before_measurements_allowed_but_after_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            phase = "2026-09-22T00:00:00Z BENCH_PHASE measured=true\n"
            log.write_text("Capture target graph\n" + phase)
            self.assertTrue(audit_module.audit(log, [])["steady_state_verified"])
            log.write_text(phase + "Capture proposal graph\n")
            self.assertFalse(audit_module.audit(log, [])["steady_state_verified"])
            log.write_text("no measured phase")
            self.assertFalse(audit_module.audit(log, [])["steady_state_verified"])

    def test_compiled_file_mtime_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            log.write_text("2026-09-22T00:00:00Z BENCH_PHASE measured=true\n")
            compiled = root / "kernel.cubin"
            compiled.write_bytes(b"diagnostic")
            os.utime(compiled, (1, 1))
            self.assertTrue(
                audit_module.audit(log, [directory])["steady_state_verified"]
            )
            os.utime(compiled, (2000000000, 2000000000))
            self.assertFalse(
                audit_module.audit(log, [directory])["steady_state_verified"]
            )

    def test_text_and_extensionless_tvm_cache_writes_fail_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "tvm-ffi"
            cache.mkdir()
            log = root / "run.log"
            log.write_text("2026-09-22T00:00:00Z BENCH_PHASE measured=true\n")
            for name in ("metadata.json", "build.ninja", "lock"):
                path = cache / name
                path.write_text("cached")
                os.utime(path, (1, 1))
            before = audit_module.audit(log, [cache])
            self.assertTrue(before["steady_state_verified"])
            self.assertIn(str(cache), before["cache_tree_sha256"])
            changed = cache / "build.ninja"
            changed.write_text("recompiled")
            os.utime(changed, (2000000000, 2000000000))
            after = audit_module.audit(log, [cache])
            self.assertFalse(after["steady_state_verified"])
            self.assertNotEqual(
                before["cache_tree_sha256"][str(cache)],
                after["cache_tree_sha256"][str(cache)],
            )

    def test_unset_or_missing_cache_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text("2026-09-22T00:00:00Z BENCH_PHASE measured=true\n")
            for root in (None, "", str(Path(directory) / "missing")):
                with self.subTest(root=root):
                    self.assertFalse(
                        audit_module.audit(log, [root])["steady_state_verified"]
                    )

    def test_nested_roots_hash_each_file_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inner = root / "tvm-ffi"
            inner.mkdir()
            (inner / "metadata.json").write_text("cached")
            os.utime(inner / "metadata.json", (1, 1))
            log = root / "run.log"
            log.write_text("2026-09-22T00:00:00Z BENCH_PHASE measured=true\n")
            with patch.object(
                audit_module, "_sha256_file", wraps=audit_module._sha256_file
            ) as hashed:
                result = audit_module.audit(log, [root, inner])
            self.assertTrue(result["steady_state_verified"])
            self.assertEqual(hashed.call_count, 1)
