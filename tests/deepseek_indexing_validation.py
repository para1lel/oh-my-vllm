import unittest

import torch
from oh_my_vllm.models.deepseek_v41.indexing import (
    score_chunks,
)

from tests.deepseek_indexing_fixtures import DeepSeekIndexingFixture


class DeepSeekIndexingValidation(DeepSeekIndexingFixture, unittest.TestCase):
    def test_rejects_noninteger_and_nonpositive_budgets(self):
        args = (torch.ones(1, 2, 3, 4), torch.ones(1, 5, 4), torch.ones(1, 2, 3))
        for budget in (True, False, 1.5, "256", None, torch.tensor(256)):
            with self.subTest(budget=budget), self.assertRaises(TypeError):
                list(score_chunks(*args, work_bytes=budget))
        for budget in (0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                list(score_chunks(*args, work_bytes=budget))

    def test_rejects_budget_below_one_row_with_required_bytes(self):
        args = (torch.ones(1, 2, 3, 4), torch.ones(1, 5, 4), torch.ones(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "120.*(row|bytes)"):
            list(score_chunks(*args, work_bytes=119))
        self.assertEqual(len(list(score_chunks(*args, work_bytes=120))), 2)

    def test_rejects_autocast_that_mixes_half_dtypes(self):
        for dtype, autocast_dtype in (
            (torch.float16, torch.bfloat16),
            (torch.bfloat16, torch.float16),
        ):
            with (
                self.subTest(dtype=dtype),
                torch.autocast("cpu", dtype=autocast_dtype),
                self.assertRaisesRegex(ValueError, "autocast"),
            ):
                list(
                    score_chunks(
                        torch.ones((1, 2, 3, 4), dtype=dtype),
                        torch.ones((1, 5, 4), dtype=dtype),
                        torch.ones((1, 2, 3), dtype=dtype),
                    )
                )

    def test_rejects_invalid_shapes(self):
        shapes = (
            ((2, 3, 4), (2, 5, 4), (2, 3, 1)),
            ((2, 3, 1, 4), (2, 5, 4, 1), (2, 3, 1)),
            ((2, 3, 1, 4), (2, 5, 4), (2, 3)),
            ((2, 3, 1, 4), (1, 5, 4), (2, 3, 1)),
            ((2, 3, 1, 4), (2, 5, 3), (2, 3, 1)),
            ((2, 3, 1, 4), (2, 5, 4), (2, 1, 1)),
            ((2, 3, 1, 4), (2, 5, 4), (2, 3, 2)),
            ((2, 3, 0, 4), (2, 5, 4), (2, 3, 0)),
            ((2, 3, 1, 0), (2, 5, 0), (2, 3, 1)),
        )
        for query, keys, weights in shapes:
            with (
                self.subTest(shapes=(query, keys, weights)),
                self.assertRaises(ValueError),
            ):
                list(
                    score_chunks(
                        torch.empty(query), torch.empty(keys), torch.empty(weights)
                    )
                )

    def test_rejects_invalid_dtypes_devices_and_nontensors(self):
        args = [torch.ones(1, 2, 3, 4), torch.ones(1, 5, 4), torch.ones(1, 2, 3)]
        for index in range(3):
            for dtype in (torch.int64, torch.bool, torch.complex64, torch.float64):
                invalid = args.copy()
                invalid[index] = invalid[index].to(dtype)
                with (
                    self.subTest(index=index, dtype=dtype),
                    self.assertRaises(TypeError),
                ):
                    list(score_chunks(*invalid))
            invalid = args.copy()
            invalid[index] = invalid[index].to("meta")
            with self.assertRaises(ValueError):
                list(score_chunks(*invalid))
            invalid[index] = None
            with self.assertRaises(TypeError):
                list(score_chunks(*invalid))
