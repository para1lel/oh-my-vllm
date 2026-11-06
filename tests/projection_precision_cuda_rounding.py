import unittest
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import (
    kernel,
    kernel_grouped,
    kernel_native,
    model_context,
    routed_projection,
)

from tests.projection_precision_fixtures import (
    fp4_literal,
    grouped_literal,
    prepared_moe_literal,
    run_grouped,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionPrecisionCuda(unittest.TestCase):
    def test_dense_prefill_graph_keeps_new_quantization_storage_and_k32_order(self):
        values = torch.zeros((129, 128), dtype=torch.bfloat16, device="cuda")
        values[:, (0, 32, 64)] = 1
        weight = torch.zeros((1280, 128), device="cuda")
        weight[:, (0, 32)] = 1
        weight[:, 64] = -1
        weight[640:, 32:] = 0
        weight = weight.to(torch.float8_e4m3fn)
        scales = torch.ones((40, 4), device="cuda")
        scales[:, 1] = 2**-25
        scales = scales.to(torch.float8_e8m0fnu)

        def run():
            encoded, input_scales = kernel.act_quant(
                values, 32, "ue8m0", torch.float8_e8m0fnu
            )
            return kernel.fp8_gemm(
                encoded,
                input_scales,
                weight,
                scales,
                block_size=32,
                output_dtype=torch.float32,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            for value in (1, 2, 0.5):
                values[:, (0, 32, 64)] = value
                graph.replay()
                expected = run()
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )
                self.assertEqual(output[:, :640].count_nonzero().item(), 0)
                self.assertTrue(
                    torch.equal(
                        output[:, 640:], torch.full_like(output[:, 640:], value)
                    )
                )
        finally:
            stream.synchronize()
            graph.reset()

    def test_routed_gate_up_share_quantization_and_replay_exactly(self):
        original = kernel.grouped_fp4_gemm

        def separate_quantization(*args, **kwargs):
            kwargs.pop("quantized", None)
            return original(*args, **kwargs)

        for strategy in ("TP4", "TEP4", "DEP4"):
            with (
                self.subTest(strategy=strategy),
                patch.object(model_context, "world_size", 1),
            ):
                moe, values, _ = prepared_moe_literal("cuda", strategy)
                with torch.no_grad():
                    for index, expert in enumerate(moe.experts):
                        expert.w2.scale.fill_(1 << index)
                values = values.repeat(3, 1)
                tokens = torch.arange(3, device="cuda")
                coefficients = torch.ones(3, device="cuda")
                counts = torch.tensor((1, 0, 2, 0), device="cuda", dtype=torch.int32)

                host_counts = None if strategy == "TP4" else [1, 0, 2, 0]

                def run(
                    moe=moe,
                    values=values,
                    tokens=tokens,
                    coefficients=coefficients,
                    counts=counts,
                    host_counts=host_counts,
                ):
                    return moe._grouped_routed(
                        values, tokens, coefficients, host_counts, counts
                    )[1]

                run()
                with patch.object(
                    kernel, "grouped_fp4_gemm", side_effect=separate_quantization
                ):
                    expected = run()
                quantize = Mock(wraps=kernel.act_quant)
                with (
                    patch.object(kernel_grouped, "act_quant", quantize),
                    patch.object(routed_projection, "act_quant", quantize),
                ):
                    actual = run()
                self.assertEqual(quantize.call_count, 2)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        replayed = run()
                    values[:, 0].fill_(2)
                    counts.copy_(
                        torch.tensor((0, 2, 0, 1), device="cuda", dtype=torch.int32)
                    )
                    if host_counts is not None:
                        host_counts[:] = (0, 2, 0, 1)
                    graph.replay()
                    with patch.object(
                        kernel, "grouped_fp4_gemm", side_effect=separate_quantization
                    ):
                        expected = run()
                    self.assertTrue(
                        torch.equal(
                            replayed.view(torch.uint8), expected.view(torch.uint8)
                        )
                    )
                finally:
                    stream.synchronize()
                    graph.reset()
                self._check_routed_quantization_shape_errors(moe, values, counts)

    def test_fp4_projection_decodes_every_packed_four_value_word(self):
        encoded = torch.zeros((4, 32), dtype=torch.float32, device="cuda")
        encoded[:, :4] = torch.eye(4, device="cuda")
        encoded = encoded.to(torch.float8_e4m3fn)
        input_scales = torch.ones((4, 1), device="cuda").to(torch.float8_e8m0fnu)
        levels = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6),
            device="cuda",
        )
        weight_scales = torch.ones((32768, 1), device="cuda").to(torch.float8_e8m0fnu)
        for first in (0, 32768):
            words = torch.arange(first, first + 32768, device="cuda")
            raw = torch.zeros((32768, 16), dtype=torch.uint8, device="cuda")
            raw[:, 0] = (words & 255).to(torch.uint8)
            raw[:, 1] = (words >> 8).to(torch.uint8)
            expected = levels[
                (words[None, :] >> (4 * torch.arange(4, device="cuda")[:, None])) & 15
            ]
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(first=first, dtype=dtype):
                    actual = kernel.fp4_gemm(
                        encoded,
                        input_scales,
                        raw.view(torch.float4_e2m1fn_x2),
                        weight_scales,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        actual, expected.to(dtype), rtol=0, atol=0
                    )
                    grouped_encoded = encoded.repeat(17, 1)
                    grouped = kernel.grouped_fp4_gemm(
                        grouped_encoded.to(torch.bfloat16),
                        torch.tensor((0, 68), device="cuda", dtype=torch.int32),
                        [(raw.view(torch.float4_e2m1fn_x2), weight_scales)] * 2,
                        output_dtype=dtype,
                        quantized=(grouped_encoded, input_scales.repeat(17, 1)),
                    )
                    torch.testing.assert_close(
                        grouped, expected.to(dtype).repeat(17, 1), rtol=0, atol=0
                    )

    def test_fp4_scales_each_k32_dot_before_accumulation(self):
        # Released FP32 and BF16 results are 646 and 648. Power-of-two scaling
        # preserves these rounded values without overflow or underflow.
        for rows in (1, 16, 17, 31, 32, 33, 63, 64, 65, 125, 126):
            encoded, scales, weight, weight_scales, factors = fp4_literal(rows)
            for dtype, value in ((torch.float32, 646), (torch.bfloat16, 648)):
                with self.subTest(rows=rows, dtype=dtype):
                    output = kernel.fp4_gemm(
                        encoded,
                        scales,
                        weight,
                        weight_scales,
                        torch.float8_e8m0fnu,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        output, (factors * value).to(dtype), rtol=0, atol=0
                    )

    def test_complete_down_k_keeps_small_terms_in_source_order(self):
        values = torch.zeros((3, 2304), dtype=torch.bfloat16, device="cuda")
        values[:, (0, 576, 1152)] = 1
        raw = torch.zeros((1280, 1152), dtype=torch.uint8, device="cuda")
        raw[:, 0] = raw[:, 288] = 2
        raw[:, 576] = 10
        weight = raw.view(torch.float4_e2m1fn_x2)
        scales = torch.ones((1280, 72), device="cuda")
        scales[:, 18] = 2**-25
        scales = scales.to(torch.float8_e8m0fnu)
        # FP32 source order is (1 + 2**-25) - 1 = 0, not 2**-25.
        actual = kernel.grouped_fp4_gemm(
            values,
            torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda"),
            [(weight, scales)] * 3,
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        encoded, input_scales = kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        actual = kernel.fp4_gemm(
            encoded, input_scales, weight, scales, act_block_size=32
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    def test_grouped_fp4_rounding_replays_with_changed_counts(self):
        for dtype, value in ((torch.bfloat16, 648), (torch.float32, 646)):
            for counts, changed in (
                ((1, 0, 33, 2), (16, 1, 0, 19)),
                ((1, 0, 3, 2), (6, 0, 0, 0)),
                ((1, 0, 13, 2), (0, 16, 0, 0)),
                ((1, 0, 14, 2), (0, 0, 17, 0)),
                ((1, 0, 257, 2), (128, 1, 0, 131)),
                ((1, 0, 1025, 2), (0, 1028, 0, 0)),
            ):
                with self.subTest(dtype=dtype, counts=counts):
                    self.check_grouped_replay(dtype, value, counts, changed)

    def check_grouped_replay(self, dtype, value, counts, changed):
        module = kernel_native._projection_module()
        case = grouped_literal(
            counts,
            max_rows=max(33, sum(counts)),
            dtype=dtype,
            width=128 if sum(counts) >= 256 else 64,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_grouped(module, case)
            eager = case.output.clone()
        stream.synchronize()
        shifts = torch.repeat_interleave(
            torch.tensor([-1, 0, 1, 2], device="cuda"),
            torch.tensor(counts, device="cuda"),
        )
        expected = (case.factors * torch.exp2(shifts[:, None].float()) * value).to(
            dtype
        )
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                run_grouped(module, case)
            for replay_counts in (counts, changed):
                with self.subTest(counts=replay_counts), torch.cuda.stream(stream):
                    case.counts.copy_(
                        torch.tensor(replay_counts, device="cuda", dtype=torch.int32)
                    )
                    case.output.fill_(-7)
                    graph.replay()
                    actual = case.output.clone()
                    finished = torch.cuda.Event()
                    finished.record()
                finished.synchronize()
                group_shifts = torch.repeat_interleave(
                    torch.tensor([-1, 0, 1, 2], device="cuda"),
                    torch.tensor(replay_counts, device="cuda"),
                )
                expected = (
                    case.factors * torch.exp2(group_shifts[:, None].float()) * value
                ).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()

    def _check_routed_quantization_shape_errors(self, moe, values, counts):
        prepared = moe._routed_state.weights.gate
        encoded, scales = kernel.act_quant(values, 32, "ue8m0", torch.float8_e8m0fnu)
        for quantized in ((encoded[:1], scales), (encoded, scales[:1])):
            with self.assertRaisesRegex(ValueError, "matching FP8 and K32"):
                kernel.grouped_fp4_gemm(values, counts, prepared, quantized=quantized)
