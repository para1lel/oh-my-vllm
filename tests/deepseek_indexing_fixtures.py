from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model_context, model_indexer
from oh_my_vllm.models.deepseek_v41.indexing import (
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


class DeepSeekIndexingFixture:
    def _check_request_score_chunks(self, device, *, heads=8, dim=128, width=1152):
        from oh_my_vllm.models.deepseek_v41 import model

        torch.manual_seed(0)
        query = torch.randn(4, 6, heads, dim, dtype=torch.bfloat16, device=device)
        keys = torch.randn(4, width, dim, dtype=torch.bfloat16, device=device)
        weights = torch.randn(4, 6, heads, dtype=torch.bfloat16, device=device)
        indexer = model.Indexer.__new__(model.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.freqs_cis = torch.empty(1, device=device)
        indexer.owns_k = False
        indexer.compress_ratio = 2
        indexer.n_heads = indexer.n_local_heads = heads
        indexer.index_head_dim, indexer.rope_head_dim = dim, 0
        indexer.softmax_scale = heads**0.5
        indexer.index_topk, indexer.candidate_topk_blocks = 4, 2
        indexer.candidate_block_size = 4
        indexer.wq_b = indexer.weights_proj = torch.nn.Identity()
        row_bytes = (
            max(heads * (dim + width), 2 * heads * width, (heads + 1) * width) * 2
            + width * 4
        )

        run = self._score_chunk_runner(
            indexer, width, device, model, row_bytes, keys, weights, query
        )

        self._check_cuda_score_chunks(
            device, query, keys, weights, row_bytes, heads, dim
        )
        self._check_request_batches(device, run)

    def _check_request_batches(self, device, run):
        for source in (False, True):
            for rows in (1, 3, 6):
                independent = [
                    run([index], rows, source, separate=False) for index in range(4)
                ]
                for batch in (1, 4, 16, 64):
                    with self.subTest(
                        device=device, batch=batch, rows=rows, source=source
                    ):
                        selection = [index % 4 for index in range(batch)]
                        result, blocks, seen, reductions = run(
                            selection, rows, source, separate=True
                        )
                        expected = [independent[index] for index in selection]
                        self.assertTrue(
                            torch.equal(
                                result, torch.cat([item[0] for item in expected])
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                blocks, torch.cat([item[1] for item in expected])
                            )
                        )
                        expected_seen = [
                            entry for item in expected for entry in item[2]
                        ]
                        self.assertEqual(
                            [(first, last) for first, last, _ in seen],
                            [(first, last) for first, last, _ in expected_seen],
                        )
                        self.assertEqual(
                            reductions,
                            [entry for item in expected for entry in item[3]],
                        )
                        for actual, reference in zip(seen, expected_seen, strict=True):
                            self.assertTrue(
                                torch.equal(
                                    actual[2].view(torch.uint8),
                                    reference[2].view(torch.uint8),
                                )
                            )

    def _check_cuda_score_chunks(
        self, device, query, keys, weights, row_bytes, heads, dim
    ):
        if device == "cuda":
            completed = []
            submitted = []

            def delayed_sum(value, async_op=False):
                self.assertTrue(async_op)
                ordinal = len(submitted)
                submitted.append(value)

                def wait():
                    if ordinal not in completed:
                        value.add_(0.25)
                        completed.append(ordinal)

                return SimpleNamespace(wait=wait)

            with patch("torch.distributed.all_reduce", side_effect=delayed_sum):
                chunks = score_chunks(
                    query,
                    keys,
                    weights,
                    work_bytes=row_bytes * 2 * query.size(0),
                    rank_sum=True,
                )
                first, last, actual = next(chunks)
                self.assertEqual((first, last), (0, 2))
                expected = (
                    (
                        torch.einsum(
                            "bshd,btd->bsht", query[:, :2].contiguous(), keys
                        ).relu_()
                        * weights[:, :2].unsqueeze(-1)
                    )
                    .sum(dim=2)
                    .float()
                    .add_(0.25)
                    .bfloat16()
                )
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.assertEqual(completed, [0])
                self.assertEqual(len(submitted), 2)
                self.assertNotEqual(submitted[0].data_ptr(), submitted[1].data_ptr())
                chunks.close()
                self.assertEqual(completed, [0, 1])
            self._check_unaligned_and_short_score_inputs(
                weights, device, query, keys, row_bytes
            )
            self._check_strided_score_weights(heads, weights, device, dim, query, keys)

    def _check_strided_score_weights(self, heads, weights, device, dim, query, keys):
        strided_weights = torch.ones(
            (1, 3, heads + 1), dtype=weights.dtype, device=device
        )[:, :, :heads]
        with (
            patch(
                "oh_my_vllm.models.deepseek_v41.indexing._rank_score_chunks",
                side_effect=AssertionError("strided weights must keep existing math"),
            ),
            patch("torch.distributed.all_reduce"),
        ):
            one_row_bytes = (
                max(heads * (dim + 8), 2 * heads * 8, (heads + 1) * 8) * 2 + 32
            )
            chunks = list(
                score_chunks(
                    query[:1, :3],
                    keys[:1, :8],
                    strided_weights,
                    work_bytes=one_row_bytes,
                    rank_sum=True,
                )
            )
            self.assertEqual(
                [(first, last) for first, last, _ in chunks],
                [(0, 1), (1, 2), (2, 3)],
            )

    def _check_unaligned_and_short_score_inputs(
        self, weights, device, query, keys, row_bytes
    ):
        unaligned = torch.empty(
            weights.numel() + 1, dtype=weights.dtype, device=device
        )[1:].view_as(weights)
        unaligned.copy_(weights)
        with (
            patch(
                "oh_my_vllm.models.deepseek_v41.indexing._rank_score_chunks",
                side_effect=AssertionError(
                    "unsupported storage must use existing math"
                ),
            ),
            patch("torch.distributed.all_reduce"),
        ):
            list(
                score_chunks(
                    query,
                    keys,
                    unaligned,
                    work_bytes=row_bytes * 2 * query.size(0),
                    rank_sum=True,
                )
            )
        with patch(
            "torch.distributed.all_reduce",
            side_effect=lambda value: value.add_(0.25),
        ):
            odd_scores = torch.cat(
                [
                    value
                    for _, _, value in score_chunks(
                        query,
                        keys[:, :17],
                        weights,
                        work_bytes=row_bytes * 2 * query.size(0),
                        rank_sum=True,
                    )
                ],
                dim=1,
            )
        odd_expected = (
            (
                torch.einsum("bshd,btd->bsht", query, keys[:, :17]).relu_()
                * weights.unsqueeze(-1)
            )
            .sum(dim=2)
            .float()
            .add_(0.25)
            .bfloat16()
        )
        self.assertTrue(
            torch.equal(odd_scores.view(torch.int16), odd_expected.view(torch.int16))
        )

    def _score_chunk_runner(
        self, indexer, width, device, model, row_bytes, keys, weights, query
    ):
        def run(selection, rows, candidate_source, *, separate):
            seen, reductions = [], []
            indexer.is_candidate_source = candidate_source
            indexer.uses_candidates = not candidate_source
            candidate_mask = torch.ones(
                (len(selection), 6, (width + 3) // 4), dtype=torch.bool, device=device
            )
            candidate_mask[:, :, ::2] = False

            def observed_chunks(*args, **kwargs):
                self.assertEqual(args[0].size(0), 1)
                for first, last, value in score_chunks(*args, **kwargs):
                    seen.append((first, last, value.clone()))
                    yield first, last, value

            def peer_sum(value, async_op=False):
                reductions.append(tuple(value.shape))
                done = False

                def wait():
                    nonlocal done
                    if not done:
                        value.add_(0.25)
                        done = True

                if async_op:
                    return SimpleNamespace(wait=wait)
                wait()

            with (
                patch.object(model_context, "world_size", 4),
                patch.object(
                    model_indexer,
                    "separate_request_math",
                    side_effect=lambda values, logical_rows: logical_rows == 1,
                )
                if device == "cpu"
                else nullcontext(),
                patch.object(model_indexer, "INDEX_SCORE_WORK_BYTES", row_bytes * rows),
                patch.object(
                    model_indexer, "score_chunks", side_effect=observed_chunks
                ),
                patch.object(model_indexer, "apply_rotary_emb"),
                patch.object(model_indexer, "position_rows"),
                patch.object(model_indexer, "fp4_act_quant"),
                patch.object(model_context.shared_attn, "index_k", keys[selection]),
                patch.object(model_context.shared_attn, "candidates", candidate_mask),
                patch("torch.distributed.all_reduce", side_effect=peer_sum),
            ):
                result = indexer(
                    weights[selection],
                    query[selection].flatten(2),
                    None,
                    width * 2 - 6,
                    128,
                    position=torch.full(
                        (len(selection), 1),
                        width * 2 - 6,
                        dtype=torch.int64,
                        device=device,
                    ),
                    logical_rows=1 if separate else 0,
                )
                blocks = model_context.shared_attn.candidates.clone()
            return result, blocks, seen, reductions

        return run
