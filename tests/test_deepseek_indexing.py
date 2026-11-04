import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.indexing import (
    INDEX_SCORE_WORK_BYTES,
    score_chunks,
    topk_indices,
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
    def test_request_math_scope_keeps_cpu_and_request_owned_dep4_layouts(self):
        from oh_my_vllm.models.deepseek_v41 import model

        for cuda, size, logical, expected in (
            (False, 4, 1, False),
            (True, 1, 1, False),
            (True, 4, 0, False),
            (True, 4, 4, False),
            (True, 4, 1, True),
        ):
            with (
                self.subTest(cuda=cuda, size=size, logical=logical),
                patch.object(model, "world_size", size),
            ):
                self.assertEqual(
                    model.separate_request_math(SimpleNamespace(is_cuda=cuda), logical),
                    expected,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_request_linear_keeps_bf16_head_bits_for_larger_request_batches(self):
        from oh_my_vllm.models.deepseek_v41.model import Linear, request_linear

        torch.manual_seed(0)
        layer = Linear(5120, 8, dtype=torch.bfloat16).cuda()
        for rows in (6, 1):
            for iteration in range(7):
                values = torch.randn(4, rows, 5120, dtype=torch.bfloat16, device="cuda")
                weights = (
                    torch.randn(8, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
                )
                with torch.no_grad():
                    layer.weight.copy_(weights)
                expected = torch.cat(
                    [layer(values[index : index + 1]) for index in range(4)]
                )
                for batch in (1, 4, 16, 64):
                    selection = torch.arange(batch, device="cuda") % 4
                    with self.subTest(iteration=iteration, batch=batch, rows=rows):
                        actual = request_linear(
                            layer, values[selection], separate_requests=True
                        )
                        self.assertTrue(
                            torch.equal(
                                actual.contiguous().view(torch.uint8),
                                expected[selection].contiguous().view(torch.uint8),
                            )
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_request_key_projection_and_grouped_output_match_independent_requests(self):
        from oh_my_vllm.models.deepseek_v41.model import (
            Linear,
            request_einsum,
            request_linear,
        )

        torch.manual_seed(0)
        for dtype in (torch.bfloat16, torch.float32):
            layer = Linear(512, 128, dtype=dtype).cuda()
            with torch.no_grad():
                layer.weight.copy_(torch.randn_like(layer.weight) * 0.02)
            for rows in (1, 3, 6):
                values = torch.randn(4, rows, 512, device="cuda", dtype=dtype)
                expected = torch.cat(
                    [layer(values[index : index + 1]) for index in range(4)]
                )
                for batch in (1, 4, 16, 64):
                    selection = torch.arange(batch, device="cuda") % 4
                    actual = request_linear(
                        layer, values[selection], separate_requests=True
                    )
                    self.assertTrue(
                        torch.equal(
                            actual.contiguous().view(torch.uint8),
                            expected[selection].contiguous().view(torch.uint8),
                        )
                    )
        weights = torch.randn(2, 1024, 4096, dtype=torch.bfloat16, device="cuda")
        for rows in (1, 6):
            values = torch.randn(4, rows, 2, 4096, dtype=torch.bfloat16, device="cuda")
            expected = torch.cat(
                [
                    torch.einsum("bsgd,grd->bsgr", values[index : index + 1], weights)
                    for index in range(4)
                ]
            )
            for batch in (1, 4, 16, 64):
                selection = torch.arange(batch, device="cuda") % 4
                actual = request_einsum(
                    "bsgd,grd->bsgr",
                    values[selection],
                    weights,
                    separate_requests=True,
                )
                self.assertTrue(
                    torch.equal(
                        actual.contiguous().view(torch.uint8),
                        expected[selection].contiguous().view(torch.uint8),
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_projection_call_sites_and_graph_replay_keep_request_shapes(self):
        from oh_my_vllm.models.deepseek_v41 import model

        class Copy(torch.nn.Module):
            def forward(self, values, **kwargs):
                return values.clone()

        torch.manual_seed(0)
        indexer = model.Indexer.__new__(model.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.freqs_cis = torch.empty(1, device="cuda")
        indexer.owns_k, indexer.is_candidate_source, indexer.uses_candidates = (
            True,
            False,
            False,
        )
        indexer.compress_ratio = 2
        indexer.n_heads = indexer.n_local_heads = 8
        indexer.index_head_dim, indexer.rope_head_dim = 128, 0
        indexer.softmax_scale, indexer.index_topk = 1, 4
        indexer.wq_b = indexer.k_norm = Copy()
        indexer.weights_proj = model.Linear(5120, 8, dtype=torch.bfloat16).cuda()
        indexer.wk = model.Linear(512, 128, dtype=torch.bfloat16).cuda()
        indexer.k_cache = torch.randn(4, 16, 128, dtype=torch.bfloat16, device="cuda")
        attention = model.Attention.__new__(model.Attention)
        torch.nn.Module.__init__(attention)
        attention.freqs_cis = torch.empty(1, device="cuda")
        attention.n_local_heads, attention.head_dim, attention.rope_head_dim = (
            16,
            512,
            0,
        )
        attention.n_local_groups, attention.o_lora_rank = 2, 1024
        attention.compress_ratio, attention.softmax_scale = 0, 1
        attention.attn_sink = torch.zeros(16, device="cuda")
        attention.wq_a = attention.q_norm = attention.wq_b = attention.wo_b = Copy()
        attention.wo_a = model.Linear(4096, 2048, dtype=torch.bfloat16).cuda()
        for layer in (indexer.weights_proj, indexer.wk, attention.wo_a):
            with torch.no_grad():
                layer.weight.normal_(std=0.02)
        slots = torch.arange(4, device="cuda")
        x = torch.randn(4, 6, 5120, dtype=torch.bfloat16, device="cuda")
        qr = torch.randn(4, 6, 1024, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(4, 3, 512, dtype=torch.bfloat16, device="cuda")
        attn_input = torch.randn(4, 6, 8192, dtype=torch.bfloat16, device="cuda")
        projected, key_shapes, head_shapes, output_shapes = [], [], [], []

        def window(values, *args, **kwargs):
            return values.new_zeros((values.size(0), 1, 512)), torch.zeros(
                (*values.shape[:2], 1), dtype=torch.int32, device="cuda"
            )

        def observed_einsum(equation, values, weight):
            if equation == "bsgd,grd->bsgr":
                output_shapes.append(tuple(values.shape))
            return original_einsum(equation, values, weight)

        def run(selected, logical):
            projected.clear()
            indexer(
                x[selected],
                qr[selected],
                latent[selected],
                20,
                1,
                cache_slots=slots[selected],
                logical_rows=logical,
            )
            result = attention(
                attn_input[selected],
                20,
                cache_slots=slots[selected],
                logical_rows=logical,
            )
            return (*projected, result)

        original_einsum = torch.einsum
        with (
            torch.inference_mode(),
            patch.object(model, "world_size", 4),
            patch.object(model, "apply_rotary_emb"),
            patch.object(model, "fp4_act_quant", wraps=model.fp4_act_quant),
            patch.object(
                model,
                "sparse_attn",
                side_effect=lambda query, *args, **kwargs: query.clone(),
            ),
            patch.object(attention, "_window_kv", side_effect=window),
            patch("torch.distributed.all_reduce"),
            patch("torch.einsum", side_effect=observed_einsum),
        ):
            handles = [
                indexer.wk.register_forward_pre_hook(
                    lambda layer, inputs: key_shapes.append(tuple(inputs[0].shape))
                ),
                indexer.weights_proj.register_forward_pre_hook(
                    lambda layer, inputs: head_shapes.append(tuple(inputs[0].shape))
                ),
                indexer.wk.register_forward_hook(
                    lambda layer, inputs, output: projected.append(output.clone())
                ),
                indexer.weights_proj.register_forward_hook(
                    lambda layer, inputs, output: projected.append(output.clone())
                ),
            ]
            try:
                run(slice(None), 1)
                self.assertEqual(key_shapes, [(1, 3, 512)] * 4)
                self.assertEqual(head_shapes, [(1, 6, 5120)] * 4)
                self.assertEqual(output_shapes, [(1, 6, 2, 4096)] * 4)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(slice(None), 1)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run(slice(None), 1)
                for _ in range(2):
                    for values in (x, qr, latent, attn_input):
                        values.normal_()
                    independent = [
                        run(slice(index, index + 1), 0) for index in range(4)
                    ]
                    graph.replay()
                    expected = (
                        [item[0] for item in independent]
                        + [item[1] for item in independent]
                        + [torch.cat([item[2] for item in independent])]
                    )
                    self.assertEqual(len(actual), len(expected))
                    for left, right in zip(actual, expected, strict=True):
                        self.assertTrue(
                            torch.equal(
                                left.contiguous().view(torch.uint8),
                                right.contiguous().view(torch.uint8),
                            )
                        )
            finally:
                for handle in handles:
                    handle.remove()
                model.shared_attn.reset_query_state()

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

            def peer_sum(value):
                reductions.append(tuple(value.shape))
                value.add_(0.25)

            with (
                patch.object(model, "world_size", 4),
                patch.object(
                    model,
                    "separate_request_math",
                    side_effect=lambda values, logical_rows: logical_rows == 1,
                )
                if device == "cpu"
                else nullcontext(),
                patch.object(model, "INDEX_SCORE_WORK_BYTES", row_bytes * rows),
                patch.object(model, "score_chunks", side_effect=observed_chunks),
                patch.object(model, "apply_rotary_emb"),
                patch.object(model, "position_rows"),
                patch.object(model, "fp4_act_quant"),
                patch.object(model.shared_attn, "index_k", keys[selection]),
                patch.object(model.shared_attn, "candidates", candidate_mask),
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
                blocks = model.shared_attn.candidates.clone()
            return result, blocks, seen, reductions

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

    def test_request_score_chunks_and_collectives_keep_independent_shapes(self):
        self._check_request_score_chunks("cpu", heads=4, dim=4, width=17)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_request_score_shapes_match_at_different_batch_sizes_and_chunk_rows(self):
        self._check_request_score_chunks("cuda")

    def test_topk_keeps_request_width_for_ties_and_short_or_empty_sources(self):
        scores = torch.full((3, 1, 16), -torch.inf)
        scores[0, 0, :10] = torch.tensor([0, 0, -1, 2, 0, -1, 0, -1, -1, -1])
        scores[1, 0, :3] = torch.tensor([2, 1, 0])
        widths = (10, 3, 0)
        actual = topk_indices(scores, 4, request_widths=widths)
        for request, width in enumerate(widths):
            expected = (
                scores[request, :, :width]
                .topk(min(4, width), sorted=False)
                .indices.sort(-1)
                .values
            )
            self.assertTrue(torch.equal(actual[request, :, : min(4, width)], expected))
            self.assertTrue(bool((actual[request, :, min(4, width) :] == 16).all()))
        for widths in ((10,), (17, 3, 0), (-1, 3, 0)):
            with self.assertRaises(ValueError):
                topk_indices(scores, 4, request_widths=widths)

    def test_candidate_blocks_keep_request_width_at_tied_cutoffs(self):
        from oh_my_vllm.models.deepseek_v41.model import select_candidate_blocks

        scores = torch.zeros(2, 1, 16)
        scores[0, :, 10:] = -torch.inf
        limits = torch.tensor([10, 16]).view(2, 1, 1)
        actual = select_candidate_blocks(scores, limits, 4, 1, request_widths=(10, 16))
        for request, width in enumerate((10, 16)):
            expected = select_candidate_blocks(
                scores[request : request + 1, :, :width], width, 4, 1
            )
            self.assertTrue(
                torch.equal(actual[request : request + 1, :, :width], expected)
            )
            self.assertFalse(bool(actual[request, :, width:].any()))

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
