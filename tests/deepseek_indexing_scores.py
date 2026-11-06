import unittest
import weakref
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.indexing import (
    INDEX_SCORE_WORK_BYTES,
    score_chunks,
)

from tests.deepseek_indexing_fixtures import DeepSeekIndexingFixture, manual_scores


class DeepSeekIndexingScores(DeepSeekIndexingFixture, unittest.TestCase):
    def test_owner_scores_keep_mp4_head_rounding(self):
        query = torch.ones((1, 3, 32, 1), dtype=torch.bfloat16)
        keys = torch.ones((1, 7, 1), dtype=query.dtype)
        weights = torch.zeros((1, 3, 32), dtype=query.dtype)
        weights[:, :, 0] = 1
        weights[:, :, 1] = 2**-8
        weights[:, :, 8] = 2**-8
        full = torch.cat(
            [value for _, _, value in score_chunks(query, keys, weights)], dim=1
        )
        partials = [
            torch.cat([value for _, _, value in score_chunks(q, keys, w)], dim=1)
            for q, w in zip(query.chunk(4, dim=2), weights.chunk(4, dim=2), strict=True)
        ]
        expected = sum(partial.float() for partial in partials).to(query.dtype)
        self.assertTrue(torch.equal(full, torch.full_like(full, 1 + 2**-7)))
        self.assertTrue(torch.equal(expected, torch.ones_like(expected)))
        for work_bytes in (308, 616, INDEX_SCORE_WORK_BYTES):
            with self.subTest(work_bytes=work_bytes):
                with patch("torch.distributed.all_reduce") as collective:
                    actual = torch.cat(
                        [
                            value
                            for _, _, value in score_chunks(
                                query,
                                keys,
                                weights,
                                work_bytes=work_bytes,
                                head_partitions=4,
                            )
                        ],
                        dim=1,
                    )
                self.assertTrue(torch.equal(actual, expected))
                collective.assert_not_called()

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
