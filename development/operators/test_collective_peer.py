import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import collective_peer


class CollectivePeer(unittest.TestCase):
    def test_capture_error_terminates_even_when_diagnostics_fail(self):
        with TemporaryDirectory() as temporary:
            marker = Path(temporary) / "unwound"
            script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from development.operators.collective_peer import Probe

class BrokenStderr:
    def write(self, value):
        raise OSError("diagnostic failure")
    def flush(self):
        raise OSError("diagnostic failure")

def enqueue():
    raise RuntimeError("enqueue failure")

probe = object.__new__(Probe)
probe.prepared = [SimpleNamespace(run=enqueue)]
sys.stderr = BrokenStderr()
try:
    probe.run()
finally:
    Path(sys.argv[1]).write_text("capture unwound")
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(marker)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 32)
            self.assertFalse(marker.exists())

    def test_native_record_uses_the_loaded_library_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "binary-a" / "collectives.so"
            library.parent.mkdir()
            library.write_bytes(b"loaded library A")
            metadata = {
                "source_id": "source-a",
                "binary_id": hashlib.sha256(library.read_bytes()).hexdigest(),
                "library": str(library),
            }
            manifest = library.with_name("manifest.json")
            manifest.write_text(json.dumps(metadata))
            (root / "collectives.json").write_text('{"source_id":"source-b"}')
            with (
                patch.dict(os.environ, {"OH_MY_VLLM_KERNEL_CACHE": str(root)}),
                patch.object(
                    collective_peer,
                    "load",
                    return_value=(object(), "source-a", library),
                ),
            ):
                self.assertEqual(collective_peer.loaded_native_identity(), metadata)
                for field, value in (
                    ("source_id", "source-b"),
                    ("binary_id", "invalid"),
                    ("library", str(root / "another.so")),
                ):
                    with self.subTest(field=field):
                        manifest.write_text(json.dumps({**metadata, field: value}))
                        with self.assertRaises(ValueError):
                            collective_peer.loaded_native_identity()
