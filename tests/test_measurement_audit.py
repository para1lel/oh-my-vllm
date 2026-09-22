"""Steady-state collection must reject new compilation and graph capture."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

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
