import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/with-env.sh"
REQUIREMENTS = ROOT / "requirements"


class EnvironmentWrapper(unittest.TestCase):
    def make_prefix(self, name="oh-my-vllm"):
        root = Path(tempfile.mkdtemp()) / name
        (root / "bin").mkdir(parents=True)
        (root / "conda-meta").mkdir()
        (root / "conda-meta/history").write_text("# test conda environment\n")
        (root / "bin/python").symlink_to(sys.executable)
        return root, name

    def run_wrapper(self, *, prefix=None, environment="oh-my-vllm", virtual=None):
        values = {
            "PATH": os.environ["PATH"],
            "CONDA_DEFAULT_ENV": environment,
        }
        if prefix is not None:
            values["OH_MY_VLLM_CONDA_PREFIX"] = str(prefix)
        if virtual is not None:
            values["VIRTUAL_ENV"] = str(virtual)
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
        prefix, _ = self.make_prefix()
        result = self.run_wrapper(prefix=prefix)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(prefix))

    def test_rejects_a_different_conda_environment(self):
        prefix, _ = self.make_prefix()
        result = self.run_wrapper(prefix=prefix, environment="other-env")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("oh-my-vllm", result.stderr)

    def test_does_not_fall_back_to_virtualenv(self):
        prefix, _ = self.make_prefix()
        result = self.run_wrapper(virtual=prefix)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OH_MY_VLLM_CONDA_PREFIX", result.stderr)

    def test_requirements_directory_separates_runtime_and_development(self):
        development = (REQUIREMENTS / "development.txt").read_text()
        self.assertTrue((REQUIREMENTS / "runtime.txt").is_file())
        self.assertIn("-r runtime.txt", development)

    def test_install_file_targets_python_315_and_cuda_134(self):
        runtime = (REQUIREMENTS / "runtime.txt").read_text()
        environment = (ROOT / "environment.yml").read_text()
        self.assertIn("torch==2.15.0+cu134", runtime)
        self.assertIn("https://download.pytorch.org/whl/cu134/", runtime)
        self.assertIn("python=3.15.2", environment)

    def test_project_install_file_does_not_require_uv(self):
        requirements = (REQUIREMENTS / "development.txt").read_text()
        self.assertNotRegex(requirements, r"(?m)^uv(?:[<=>!~].*)?$")

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
