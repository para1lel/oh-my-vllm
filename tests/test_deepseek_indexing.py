import unittest
import weakref
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.indexing import (
    INDEX_SCORE_WORK_BYTES,
    score_chunks,
)


def manual_scores(query, keys, weights):
    batch, rows, heads, _ = query.shape
    expected = torch.empty((batch, rows, keys.size(1)), dtype=query.dtype)
    for b in range(batch):
        for s in range(rows):
            for t in range(keys.size(1)):
                products = []
                for h in range(heads):
                    dot = sum(
                        float(q) * float(k)
                        for q, k in zip(query[b, s, h], keys[b, t], strict=True)
                    )
                    rounded = torch.tensor(max(dot, 0), dtype=query.dtype)
                    products.append(float(rounded * weights[b, s, h]))
                expected[b, s, t] = sum(products)
    return expected


class DeepSeekIndexing(unittest.TestCase):
    def test_rank_sum_rounds_once_and_budgets_the_fp32_accumulator(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        query = torch.ones((1, 3, 1, 1), dtype=torch.bfloat16)
        keys = torch.ones((1, 7, 1), dtype=query.dtype)
        weights = torch.ones((1, 3, 1), dtype=query.dtype)

        def peer_sum(value):
            value.add_(256)
            value.add_(-256)
            value.add_(0.25)

        scores = []
        case = self

        class ScoreLifetime(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if (
                    function == torch.ops.aten._to_copy.default
                    and args[0].dtype == torch.float32
                    and kwargs.get("dtype") == torch.bfloat16
                ):
                    case.assertIsNone(scores[-1]())
                output = function(*args, **kwargs)
                if function == torch.ops.aten.sum.dim_IntList:
                    scores.append(weakref.ref(output))
                return output

        with (
            ScoreLifetime(),
            patch("torch.distributed.all_reduce", side_effect=peer_sum),
        ):
            chunks = list(
                score_chunks(query, keys, weights, work_bytes=112, rank_sum=True)
            )
        self.assertEqual([(first, last) for first, last, _ in chunks], [(0, 2), (2, 3)])
        actual = torch.cat([value for _, _, value in chunks], dim=1)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 1.25), rtol=0, atol=0
        )
        self.assertEqual(actual.dtype, torch.bfloat16)

        def double_peer_sum(value):
            value.add_(0.25 + 2**-40)

        with patch("torch.distributed.all_reduce", side_effect=double_peer_sum):
            values = list(
                score_chunks(
                    query.double(),
                    keys.double(),
                    weights.double(),
                    work_bytes=448,
                    rank_sum=True,
                )
            )
        actual = torch.cat([value for _, _, value in values], dim=1)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 1.25 + 2**-40), rtol=0, atol=0
        )

    def test_rank_sum_reuses_storage_while_collectives_retain_inputs(self):
        query = torch.arange(1, 4, dtype=torch.float32).reshape(1, 3, 1, 1)
        keys = torch.ones((1, 7, 1))
        weights = torch.ones((1, 3, 1))
        retained = []

        def peer_sum(value):
            retained.append(value)
            value.add_(0.25)

        with patch("torch.distributed.all_reduce", side_effect=peer_sum):
            chunks = list(
                score_chunks(query, keys, weights, work_bytes=112, rank_sum=True)
            )
        expected = torch.tensor([1.25, 2.25, 3.25]).reshape(1, 3, 1).expand(1, 3, 7)
        actual = torch.cat([value for _, _, value in chunks], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(len(retained), 1)
        self.assertEqual(
            len({value.untyped_storage().data_ptr() for value in retained}), 1
        )

    def test_autocast_keeps_reference_product_dtype_and_rounding(self):
        generator = torch.Generator().manual_seed(18)
        query = torch.randn((1, 2, 3, 4), generator=generator)
        keys = torch.randn((1, 5, 4), generator=generator)
        weights = torch.randn((1, 2, 3), generator=generator)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = (
                torch.einsum("bshd,btd->bsht", query, keys).relu_()
                * weights.unsqueeze(-1)
            ).sum(dim=2)
            actual = next(score_chunks(query, keys, weights))[2]
        self.assertEqual(actual.dtype, expected.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_score_order_and_rounding_match_manual_reference(self):
        generator = torch.Generator().manual_seed(123)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                query = torch.randn((2, 5, 3, 4), generator=generator, dtype=dtype)
                keys = torch.randn((2, 7, 4), generator=generator, dtype=dtype)
                weights = torch.randn((2, 5, 3), generator=generator, dtype=dtype)
                original = [tensor.clone() for tensor in (query, keys, weights)]
                expected = manual_scores(query, keys, weights)
                row_bytes = 2 * 42 * query.element_size()
                for rows in (1, 2, 5, 20):
                    chunks = list(
                        score_chunks(query, keys, weights, work_bytes=rows * row_bytes)
                    )
                    self.assertEqual(
                        [(first, last) for first, last, _ in chunks],
                        [(first, min(first + rows, 5)) for first in range(0, 5, rows)],
                    )
                    actual = torch.cat([scores for _, _, scores in chunks], dim=1)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected)
                for actual, before in zip(
                    (query, keys, weights), original, strict=True
                ):
                    torch.testing.assert_close(actual, before, rtol=0, atol=0)

    def test_bfloat16_matches_untiled_model_operations_exactly(self):
        generator = torch.Generator().manual_seed(33)
        query = torch.randn((2, 7, 16, 8), generator=generator).bfloat16()
        keys = torch.randn((2, 13, 8), generator=generator).bfloat16()
        weights = torch.randn((2, 7, 16), generator=generator).bfloat16()
        expected = (
            torch.einsum("bshd,btd->bsht", query, keys).relu_() * weights.unsqueeze(-1)
        ).sum(dim=2)
        row_bytes = 2 * 416 * 2
        for rows in (1, 3, 7):
            actual = torch.cat(
                [
                    scores
                    for _, _, scores in score_chunks(
                        query, keys, weights, work_bytes=rows * row_bytes
                    )
                ],
                dim=1,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_each_real_einsum_fits_with_its_reduction(self):
        query = torch.ones((4, 11, 16, 3), dtype=torch.bfloat16)
        keys = torch.ones((4, 17, 3), dtype=query.dtype)
        weights = torch.ones((4, 11, 16), dtype=query.dtype)
        row_bytes = 4 * 544 * 2
        budget = 3 * row_bytes + row_bytes - 1
        original_einsum = torch.einsum
        ranges = []

        def checked_einsum(equation, q, k):
            result = original_einsum(equation, q, k)
            reduction_bytes = q.size(0) * q.size(1) * k.size(1) * q.element_size()
            per_head_bytes = result.numel() * result.element_size()
            self.assertLessEqual(
                max(
                    per_head_bytes + q.numel() * q.element_size(),
                    2 * per_head_bytes,
                    per_head_bytes + reduction_bytes,
                ),
                budget,
            )
            ranges.append(q.size(1))
            return result

        with patch("torch.einsum", side_effect=checked_einsum):
            chunks = list(score_chunks(query, keys, weights, work_bytes=budget))
        self.assertEqual(ranges, [3, 3, 3, 2])
        for _, _, scores in chunks:
            torch.testing.assert_close(scores, torch.full_like(scores, 48))

    def test_one_million_context_row_fits_default_budget_without_allocation(self):
        query = torch.empty((4, 2, 16, 128), device="meta", dtype=torch.bfloat16)
        keys = torch.empty((4, 1048576, 128), device="meta", dtype=query.dtype)
        weights = torch.empty((4, 2, 16), device="meta", dtype=query.dtype)
        chunks = list(score_chunks(query, keys, weights))
        self.assertEqual([(first, last) for first, last, _ in chunks], [(0, 1), (1, 2)])
        self.assertLessEqual(4 * 2 * 16 * 1048576 * 2, INDEX_SCORE_WORK_BYTES)

    def test_empty_rows_keys_and_batch(self):
        for batch, rows, length in ((2, 0, 7), (2, 5, 0), (0, 5, 7)):
            with self.subTest(shape=(batch, rows, length)):
                chunks = list(
                    score_chunks(
                        torch.empty((batch, rows, 3, 4)),
                        torch.empty((batch, length, 4)),
                        torch.empty((batch, rows, 3)),
                        work_bytes=1,
                    )
                )
                if rows == 0:
                    self.assertEqual(chunks, [])
                else:
                    self.assertEqual(len(chunks), 1)
                    first, last, scores = chunks[0]
                    self.assertEqual((first, last), (0, rows))
                    self.assertEqual(scores.shape, (batch, rows, length))

    def test_noncontiguous_inputs(self):
        query = torch.arange(120, dtype=torch.float32).reshape(2, 5, 3, 4) / 8
        keys = torch.arange(56, dtype=torch.float32).reshape(2, 7, 4) / 16
        weights = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 4
        query = query.transpose(1, 2).contiguous().transpose(1, 2)
        keys = keys.transpose(1, 2).contiguous().transpose(1, 2)
        weights = weights.transpose(1, 2).contiguous().transpose(1, 2)
        actual = torch.cat(
            [scores for _, _, scores in score_chunks(query, keys, weights)], dim=1
        )
        torch.testing.assert_close(actual, manual_scores(query, keys, weights))

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


if __name__ == "__main__":
    unittest.main()
