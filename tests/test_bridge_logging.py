"""Exercise the real -m entrypoint without initializing CUDA or loading weights."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import msgpack
import zmq


class BridgeLoggingTest(unittest.TestCase):
    def test_module_entrypoint_emits_correlated_json(self):
        with tempfile.TemporaryDirectory() as directory:
            address = "ipc://" + str(Path(directory) / "test.ipc")
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(address)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "oh_my_vllm.worker.zmq_bridge",
                    "--socket",
                    address,
                ],
                env={**os.environ, "OH_MY_VLLM_RUN_ID": "logging-test"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                socket.send(msgpack.packb({"type": "shutdown"}))
                _, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr)
                events = [json.loads(line) for line in stderr.splitlines()]
                self.assertTrue(
                    any(e["message"] == "Shutdown received" for e in events)
                )
                self.assertTrue(all(e["run_id"] == "logging-test" for e in events))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                socket.close()
                context.term()


if __name__ == "__main__":
    unittest.main()
