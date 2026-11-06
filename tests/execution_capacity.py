import unittest


class ExecutionCapacity(unittest.TestCase):
    def test_requested_usable_capacity_is_exact_or_fails(self):
        from oh_my_vllm.models.qwen38.cache_budget import cache_budget

        from tests.qwen_radix_physical_fixtures import config

        gib = 1 << 30
        for speculative in (0, 4):
            options = config(
                mamba_blocks=128,
                num_gpu_blocks=8192,
                num_speculative_tokens=speculative,
            )
            for total, free in ((192 * gib, 162 * gib), (200 * gib, 160 * gib)):
                result = cache_budget(
                    options, total_bytes=total, free_bytes=free, overhead_bytes=16 * gib
                )
                self.assertEqual(
                    (result["mamba_blocks"], result["num_gpu_blocks"]), (128, 8192)
                )
                self.assertEqual(result["policy"], "radix_fixed_slabs")
                self.assertFalse(result["capacity_verified"])
            with self.assertRaises(ValueError):
                cache_budget(
                    options,
                    total_bytes=100 * gib,
                    free_bytes=30 * gib,
                    overhead_bytes=16 * gib,
                )
            for invalid in (0, 1, -1, True, 2.5, 1 << 32):
                with self.subTest(pages=invalid), self.assertRaises(ValueError):
                    cache_budget(
                        dict(options, num_gpu_blocks=invalid),
                        total_bytes=180 * gib,
                        free_bytes=150 * gib,
                        overhead_bytes=16 * gib,
                    )
