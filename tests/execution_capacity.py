import unittest


class ExecutionCapacity(unittest.TestCase):
    def test_cache_capacity_is_fixed_and_never_silently_reduced(self):
        from oh_my_vllm.models.qwen38.cache_budget import cache_budget

        config = dict(
            num_gpu_blocks=None,
            mamba_blocks=128,
            max_num_seqs=4,
            max_num_batched_tokens=32768,
            num_speculative_tokens=0,
        )
        gib = 1 << 30
        for speculative, expected in ((0, 1824), (4, 1608)):
            config["num_speculative_tokens"] = speculative
            for total, free in ((192 * gib, 162 * gib), (200 * gib, 160 * gib)):
                result = cache_budget(
                    config,
                    total_bytes=total,
                    free_bytes=free,
                    overhead_bytes=16 * gib,
                )
                self.assertEqual(result["num_gpu_blocks"], expected)
                self.assertEqual(result["policy"], "static_blocks")
            with self.assertRaisesRegex(ValueError, "cache does not fit"):
                cache_budget(
                    config,
                    total_bytes=180 * gib,
                    free_bytes=100 * gib,
                    overhead_bytes=16 * gib,
                )
            config["num_gpu_blocks"] = 512
            result = cache_budget(
                config,
                total_bytes=180 * gib,
                free_bytes=100 * gib,
                overhead_bytes=16 * gib,
            )
            self.assertEqual(result["num_gpu_blocks"], 512)
            for invalid in (0, 1, -1, True, 2.5, 1 << 32):
                config["num_gpu_blocks"] = invalid
                with self.assertRaises(ValueError):
                    cache_budget(
                        config,
                        total_bytes=180 * gib,
                        free_bytes=150 * gib,
                        overhead_bytes=16 * gib,
                    )
            config["num_gpu_blocks"] = None
