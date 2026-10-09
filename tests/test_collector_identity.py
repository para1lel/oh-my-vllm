"""The batch tail, not the fastest request, defines the TTFT regression gate."""

import importlib.util
import sys
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("ttft_metrics", BENCH / "common.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
sys.path.pop(0)


class CollectorIdentityTest(unittest.TestCase):
    def test_cuda_provenance_requires_loaded_module_identity(self):
        import json

        row = {
            "nvcc_path": "/cuda/bin/nvcc",
            "nvcc_version": "Cuda compilation tools, release 12.8, V12.8.1",
            "compiler_identity_source": "live",
            "so_path": "/tmp/loaded.so",
            "so_sha256": "a" * 64,
        }
        log = "CUDA_BUILD_PROVENANCE " + json.dumps(row)
        self.assertEqual(metrics._parse_cuda_provenance(log), row)
        for bad in (
            "",
            log + "\n" + log,
            "CUDA_BUILD_PROVENANCE " + json.dumps({**row, "so_sha256": None}),
            "CUDA_BUILD_PROVENANCE " + json.dumps({**row, "nvcc_path": None}),
            "CUDA_BUILD_PROVENANCE "
            + json.dumps({**row, "compiler_identity_source": None}),
            "CUDA_BUILD_PROVENANCE " + json.dumps({**row, "so_sha256": "bad"}),
            "CUDA_BUILD_PROVENANCE "
            + json.dumps({**row, "nvcc_version": "nvcc-unavailable: missing"}),
            "CUDA_BUILD_PROVENANCE "
            + json.dumps({**row, "nvcc_version": "nvcc-changed-during-build"}),
        ):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                metrics._parse_cuda_provenance(bad)

    def test_worker_and_scheduler_pool_capacities_must_be_observed(self):
        reported = (
            "INFO BENCH_CONFIG worker_fa_pool_blocks=1400 "
            "worker_gdn_pool_blocks=128 fa_pool_blocks=1400 gdn_pool_blocks=128"
        )
        worker, scheduler = metrics._observed_pool_capacities(reported, 1400, 128)
        self.assertEqual(worker, {"fa": 1400, "mamba": 128})
        self.assertEqual(scheduler, worker)
        for log in (
            reported.replace(
                "worker_gdn_pool_blocks=128", "worker_gdn_pool_blocks=127"
            ),
            reported.replace(" fa_pool_blocks=1400", " fa_pool_blocks=1399"),
            "INFO BENCH_CONFIG max_num_seqs=32",
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                metrics._observed_pool_capacities(log, 1400, 128)


if __name__ == "__main__":
    unittest.main()
