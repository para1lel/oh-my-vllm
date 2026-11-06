import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import (
    kernel_native,
    model_attention,
    model_context,
    model_indexer,
)
from oh_my_vllm.models.deepseek_v41.indexing import (
    topk_indices,
)

from tests.deepseek_indexing_fixtures import DeepSeekIndexingFixture
from tests.deepseek_indexing_fixtures import manual_scores as manual_scores
from tests.deepseek_indexing_scores import (
    DeepSeekIndexingScores as DeepSeekIndexingScores,
)
from tests.deepseek_indexing_validation import (
    DeepSeekIndexingValidation as DeepSeekIndexingValidation,
)


class DeepSeekIndexing(DeepSeekIndexingFixture, unittest.TestCase):
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
                patch.object(model_context, "world_size", size),
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

        indexer, attention, slots, x, qr, latent, attn_input = (
            self._projection_shape_fixture(model)
        )
        projected, key_shapes, head_shapes, output_shapes = [], [], [], []

        def window(values, *args, **kwargs):
            return values.new_zeros((values.size(0), 1, 512)), torch.zeros(
                (*values.shape[:2], 1), dtype=torch.int32, device="cuda"
            )

        def observed_einsum(equation, values, weight):
            if equation == "bsgd,grd->bsgr":
                output_shapes.append(tuple(values.shape))
            return original_einsum(equation, values, weight)

        run = self._projection_run(
            projected, indexer, x, qr, latent, slots, attention, attn_input
        )

        original_einsum = torch.einsum
        with (
            torch.inference_mode(),
            patch.object(model_context, "world_size", 4),
            patch.object(model_indexer, "apply_rotary_emb"),
            patch.object(model_attention, "apply_rotary_emb"),
            patch.object(
                model_indexer, "fp4_act_quant", wraps=model_indexer.fp4_act_quant
            ),
            patch.object(
                model_attention,
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
                self._check_projection_graph_replay(
                    run,
                    key_shapes,
                    head_shapes,
                    output_shapes,
                    x,
                    qr,
                    latent,
                    attn_input,
                )
            finally:
                for handle in handles:
                    handle.remove()
                model_context.shared_attn.reset_query_state()

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_topk_default_keeps_torch_nan_behavior(self):
        scores = torch.zeros((1, 1, 1024), dtype=torch.bfloat16, device="cuda")
        scores[..., 0] = torch.nan
        expected = scores.topk(512, sorted=False).indices.sort(-1).values
        with patch(
            "oh_my_vllm.models.deepseek_v41.kernel_native._transform_module",
            side_effect=AssertionError(
                "default top-k must preserve Torch NaN behavior"
            ),
        ):
            actual = topk_indices(scores, 512)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool(((actual >= 0) & (actual < 1024)).all()))
        self.assertTrue(bool((actual.diff(dim=-1) > 0).all()))
        self.assertTrue(bool((actual == 0).any()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_topk_model_keeps_early_valid_indices_and_sentinels_in_graph(self):
        from oh_my_vllm.models.deepseek_v41 import model

        width, rows, offset = 1025, 6, 128
        source = torch.ones((1, rows, width), dtype=torch.bfloat16, device="cuda")
        query = source.new_zeros((1, rows, 1, 1))
        keys = source.new_zeros((1, width, 1))
        weights = source.new_zeros((1, rows, 1))
        indexer = SimpleNamespace(
            compress_ratio=2,
            index_topk=512,
            is_candidate_source=False,
            uses_candidates=False,
        )

        def chunks(*args, **kwargs):
            yield 0, rows, source.clone()

        def run():
            return model.Indexer._select_indices(
                indexer, query, keys, weights, 0, offset
            )[0]

        expected = torch.full((1, rows, 512), -1, dtype=torch.int32, device="cuda")
        for row in range(rows):
            valid = (row + 1) // 2
            expected[0, row, :valid] = torch.arange(valid, device="cuda") + offset
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with (
                patch.object(model_indexer, "score_chunks", side_effect=chunks),
                patch.object(
                    kernel_native,
                    "_transform_module",
                    wraps=kernel_native._transform_module,
                ) as native,
            ):
                with torch.cuda.stream(stream):
                    self.assertTrue(torch.equal(run(), expected))
                with torch.cuda.graph(graph, stream=stream):
                    actual = run()
                with torch.cuda.stream(stream):
                    source.fill_(2)
                    graph.replay()
                    self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(native.call_count, 2)
        finally:
            stream.synchronize()
            graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_topk_unsupported_layout_count_and_request_widths_keep_torch_behavior(self):
        values = torch.randn((1, 2, 1024), device="cuda").bfloat16()
        unaligned = torch.empty(values.numel() + 1, device="cuda", dtype=values.dtype)[
            1:
        ]
        unaligned = unaligned.view_as(values).copy_(values)
        with patch(
            "oh_my_vllm.models.deepseek_v41.kernel_native._transform_module",
            side_effect=AssertionError(
                "unsupported inputs must keep the existing path"
            ),
        ):
            for scores, count, widths in (
                (values.cpu(), 512, None),
                (values.float(), 512, None),
                (values, 256, None),
                (values[..., ::2], 512, None),
                (unaligned, 512, None),
                (values, 512, (513,)),
            ):
                actual = topk_indices(
                    scores, count, request_widths=widths, finite_scores=True
                )
                expected = scores[..., : widths[0]] if widths else scores
                expected = expected.topk(count, sorted=False).indices.sort(-1).values
                self.assertTrue(torch.equal(actual, expected))

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

    def _check_projection_graph_replay(
        self, run, key_shapes, head_shapes, output_shapes, x, qr, latent, attn_input
    ):
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
            independent = [run(slice(index, index + 1), 0) for index in range(4)]
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

    def _projection_shape_fixture(self, model):
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
        return indexer, attention, slots, x, qr, latent, attn_input

    def _projection_run(
        self, projected, indexer, x, qr, latent, slots, attention, attn_input
    ):
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

        return run


if __name__ == "__main__":
    unittest.main()
