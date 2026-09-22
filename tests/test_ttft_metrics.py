"""The batch tail, not the fastest request, defines the TTFT regression gate."""

import importlib.util
import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("ttft_metrics", BENCH / "ttft.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
sys.path.pop(0)


class TTFTMetricsTest(unittest.TestCase):
    def artifact(self, times, rate=100):
        artifact = {
            "hardware": {
                "gpu": "GPU-test",
                "cpu_affinity": [8],
                "gpu_info": "GPU-test, B200, 183359 MiB, driver",
            },
            "cache_config": {"gpu_memory_utilization": 0.85},
            "capacities": {"fa": 1400, "mamba": 128},
            "engine": "vllm",
            "commit": metrics.FROZEN_SHA,
            "config": {
                key: 1
                for key in (
                    "model",
                    "max_model_len",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "block_size",
                    "enable_prefix_caching",
                    "enable_chunked_prefill",
                    "async_scheduling",
                    "mamba_ssm_cache_dtype",
                )
            },
            "measurement_audit": {"steady_state_verified": True},
            "workload": {
                "mode": "ordinary",
                "batch_size": 2,
                "input_len": 32,
                "output_len": 4,
            },
            "runs": [
                {
                    "ttft_s": list(times),
                    "output_tokens": 8,
                    "output_tps": rate,
                    "preemptions": 0,
                    "prefix_hit_tokens": 0,
                }
                for _ in range(5)
            ],
        }

        artifact["warmups"] = artifact["runs"][:2]
        return artifact

    def test_slow_request_cannot_hide_behind_fast_first_token(self):
        result = metrics.compare(self.artifact([1, 2]), self.artifact([0.5, 2.3]))
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["ttft_ratio"], 1.15)

    def test_both_gates_and_variance_are_required(self):
        baseline = self.artifact([1, 2])
        self.assertTrue(
            metrics.compare(baseline, self.artifact([1.1, 2.1], 96))["passed"]
        )
        self.assertFalse(metrics.compare(baseline, self.artifact([1, 2], 94))["passed"])
        unstable = self.artifact([1, 2])
        unstable["runs"][0]["ttft_s"][1] = 2.2
        result = metrics.compare(baseline, unstable)
        self.assertFalse(result["stable"])
        self.assertFalse(result["passed"])

    def test_invalid_or_partial_measurements_are_rejected(self):
        for field, value in [
            ("ttft_s", [1]),
            ("ttft_s", [1, float("nan")]),
            ("output_tokens", 7),
            ("preemptions", 1),
            ("output_tps", float("inf")),
        ]:
            artifact = self.artifact([1, 2])
            artifact["runs"][0][field] = value
            with self.assertRaises(ValueError):
                metrics.summarize(artifact["runs"], 2, 4)
        with self.assertRaises(ValueError):
            metrics.summarize(self.artifact([1, 2])["runs"][:3], 2, 4)

    def test_identity_warmup_cache_and_capture_evidence_are_required(self):
        import copy

        baseline = self.artifact([1, 2])
        for field, value in [
            ("warmups", []),
            ("measurement_audit", {"steady_state_verified": False}),
            ("config", {}),
        ]:
            candidate = copy.deepcopy(baseline)
            candidate[field] = value
            with self.assertRaises(ValueError):
                metrics.compare(baseline, candidate)
        baseline["commit"] = "wrong"
        with self.assertRaises(ValueError):
            metrics.compare(baseline, self.artifact([1, 2]))
        baseline = self.artifact([1, 2])
        candidate = self.artifact([1, 2])
        candidate["runs"][0]["prefix_hit_tokens"] = 784
        with self.assertRaisesRegex(ValueError, "prefix hit"):
            metrics.compare(baseline, candidate)
