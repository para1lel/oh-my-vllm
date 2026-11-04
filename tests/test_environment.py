import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/with-env.sh"


class EnvironmentWrapper(unittest.TestCase):
    def make_prefix(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "oh-my-vllm"
        (root / "bin").mkdir(parents=True)
        (root / "conda-meta").mkdir()
        (root / "bin/python").symlink_to(sys.executable)
        return root

    def run_wrapper(self, *, prefix=None, environment="oh-my-vllm"):
        values = {
            "PATH": os.environ["PATH"],
            "CONDA_DEFAULT_ENV": environment,
        }
        if prefix is not None:
            values["OH_MY_VLLM_CONDA_PREFIX"] = str(prefix)
        return subprocess.run(
            [
                "/bin/bash",
                str(WRAPPER),
                "python",
                "-c",
                "import os; print(os.environ['OH_MY_VLLM_CONDA_PREFIX'])",
            ],
            cwd=ROOT,
            env=values,
            capture_output=True,
            text=True,
        )

    def test_accepts_the_named_conda_environment(self):
        prefix = self.make_prefix()
        result = self.run_wrapper(prefix=prefix)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(prefix))

    def test_rejects_a_different_conda_environment(self):
        prefix = self.make_prefix()
        result = self.run_wrapper(prefix=prefix, environment="other-env")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("oh-my-vllm", result.stderr)

    def test_rejects_a_missing_environment_prefix(self):
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OH_MY_VLLM_CONDA_PREFIX", result.stderr)

    def test_runtime_imports_use_the_selected_conda_environment(self):
        prefix = os.environ.get("OH_MY_VLLM_CONDA_PREFIX")
        self.assertIsNotNone(prefix)
        result = subprocess.run(
            [
                str(Path(prefix) / "bin/python"),
                "-c",
                "import torch, tvm_ffi, zmq; print(torch.__version__)",
            ],
            env={
                **os.environ,
                "CONDA_DEFAULT_ENV": "oh-my-vllm",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
