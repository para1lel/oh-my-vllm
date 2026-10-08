"""CPU regression checks for benchmark process ownership and configuration."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare", ROOT / "benchmarks/compare_vllm.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class BenchmarkTests(unittest.TestCase):
    def test_historical_comparison_never_launches_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "worker"
            binary.write_bytes(b"fixture binary")
            artifact = Path(directory) / "original.json"
            measurement = {
                "output_tps": 100.0,
                "output_tokens": 4096,
                "prefix_hit_tokens": 0,
                "proposed_draft_tokens": 4,
                "accepted_draft_tokens": 3,
            }
            fixture = {
                "protocol": {
                    "model": "fixture-model",
                    "block_size": 784,
                    "max_model_len": 65536,
                    "max_num_seqs": 32,
                    "max_num_batched_tokens": 32768,
                    "language_model_only": True,
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": True,
                    "async_scheduling": False,
                    "temperature": 0,
                    "ignore_eos": True,
                    "detokenize": False,
                    "mtp_ssm_dtype": "bfloat16",
                    "ordinary_prefix_ssm_dtype": "auto",
                },
                "warmup_per_engine": 2,
                "rows": [
                    {
                        "mode": "mtp",
                        "batch_size": 1,
                        "input_len": 32768,
                        "output_len": 4096,
                        "num_gpu_blocks": 1024,
                        "speculative_tokens": 4,
                        "vllm_version": "fixture",
                        "baseline": [measurement] * 3,
                    }
                ],
            }
            artifact.write_text(json.dumps(fixture))
            self.enterContext(
                patch.object(
                    compare,
                    "BASELINE_SHA256",
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                )
            )
            args = SimpleNamespace(
                binary=binary,
                baseline_json=artifact,
                mode="mtp",
                batch_sizes=[1],
                input_len=32768,
                output_len=4096,
                num_gpu_blocks=1024,
                speculative_tokens=4,
                warmup=2,
                repetitions=5,
                output=None,
                model="fixture-model",
            )
            with self.assertRaisesRegex(ValueError, "five measured repetitions"):
                compare.historical_baseline(artifact, args, 1)
            original = json.loads(artifact.read_text())
            row = next(
                row
                for row in original["rows"]
                if row["mode"] == "mtp" and row["batch_size"] == 1
            )
            row["baseline"] = [row["baseline"][0]] * 5
            output = "\n".join(
                ["WARMUP_RESULT " + json.dumps(row["baseline"][0])] * 2
                + ["BENCH_RESULT " + json.dumps(run) for run in row["baseline"]]
            )
            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-fixture"}),
                patch.object(
                    compare,
                    "source_identity",
                    return_value={
                        "binary_sha256": hashlib.sha256(
                            binary.read_bytes()
                        ).hexdigest(),
                    },
                ),
                patch.object(compare, "runtime_identity", return_value={}),
                patch.object(compare, "historical_baseline", return_value=row),
                patch.object(compare, "run_engine", return_value=output) as run,
                patch("builtins.print"),
            ):
                compare.compare(args)
            self.assertEqual(run.call_count, 1)
            self.assertIn("bench", run.call_args.args[0])
            self.assertNotIn("--baseline", run.call_args.args[0])
            invalid = Path(directory) / "invalid.json"
            for mutation in (
                "model",
                "block_size",
                "missing",
                "list",
                "warmup",
                "throughput",
            ):
                data = json.loads(artifact.read_text())
                if mutation in ("model", "block_size"):
                    data["protocol"][mutation] = "wrong"
                elif mutation == "missing":
                    del data["protocol"]
                elif mutation == "list":
                    data = original["rows"]
                elif mutation == "throughput":
                    data["rows"][0]["baseline"][0]["output_tps"] += 1
                else:
                    data["warmup_per_engine"] = 99
                invalid.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    compare.historical_baseline(invalid, args, 1)
            with self.assertRaisesRegex(ValueError, "baseline hash"):
                compare.historical_baseline(invalid, args, 2)
            args.input_len = 123
            with self.assertRaisesRegex(ValueError, "exactly one matching workload"):
                compare.historical_baseline(artifact, args, 1)

    def test_sample_count_and_spread_are_hard_gates(self):
        for values, message in (
            ([100] * 3, "five measured repetitions"),
            ([100, 100, 100, 100, 112], "spread"),
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, message),
            ):
                compare.stable_median(
                    [{"output_tps": value} for value in values], "test"
                )

    def test_timeout_terminates_worker_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            code = (
                "import subprocess,sys,time; "
                "from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                compare.run_engine(
                    [sys.executable, "-c", code, str(pid_file)], timeout=1
                )
            pid = int(pid_file.read_text())
            # Orphan zombies may await PID 1; they consume no GPU/CPU resources.
            state = Path(f"/proc/{pid}/stat")
            if state.exists():
                self.assertIn(state.read_text().split(") ")[1][0], ("Z", "X"))

    def test_contention_detection_ignores_other_gpus_and_own_group(self):
        with (
            patch.object(
                compare.subprocess,
                "check_output",
                return_value="GPU-own, 11\nGPU-other, 12\n",
            ),
            patch.object(compare.os, "getpgid", return_value=77),
        ):
            compare.assert_gpu_exclusive("GPU-own", 77)
        with (
            patch.object(
                compare.subprocess, "check_output", return_value="GPU-own, 11\n"
            ),
            patch.object(compare.os, "getpgid", return_value=88),
            self.assertRaisesRegex(RuntimeError, "external process 11"),
        ):
            compare.assert_gpu_exclusive("GPU-own", 77)

    def test_numeric_gpu_id_cannot_bypass_contention_monitor(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/compare_vllm.py"),
                "--model",
                "fixture-model",
                "--baseline-json",
                "unused.json",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("idle GPU UUID", result.stderr)

    def test_zero_drafts_rejected_before_loading_model(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/compare_vllm.py"),
                "--model",
                "fixture-model",
                "--baseline-json",
                "unused.json",
                "--mode",
                "mtp",
                "--speculative-tokens",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("MTP requires positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
