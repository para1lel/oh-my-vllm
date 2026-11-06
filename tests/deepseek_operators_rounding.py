import unittest
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native, kernel_projection


class DeepSeekOperatorContractsRounding(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp8_projection_scales_each_k32_dot_before_accumulation(self):
        # Raw K32 dots are -1 and 323. Their scaled FP32 sum rounds to 646,
        # then the BF16 midpoint rounds to 648, not 644.
        for rows, width, outputs in (
            (rows, width, outputs)
            for rows in (
                1,
                2,
                4,
                8,
                16,
                17,
                31,
                32,
                33,
                63,
                64,
                65,
                125,
                126,
                128,
                129,
                257,
            )
            for width, outputs in (
                (64, 160),
                (96, 32),
                (1024, 512),
                (1024, 1280),
                (1056, 1088),
                (16384, 32),
                (160, 1280),
                (6144, 1280),
                (6176, 1280),
            )
        ):
            inputs, weight, row_shifts, group_shifts, input_scales, weight_scales = (
                self._fp8_rounding_inputs(rows, width, outputs)
            )
            factors = torch.exp2(
                row_shifts[:, None].float()
                + group_shifts.repeat_interleave(32)[None, :].float()
            )
            for dtype, value in ((torch.float32, 646), (torch.bfloat16, 648)):
                with self.subTest(rows=rows, width=width, outputs=outputs, dtype=dtype):
                    actual = kernel.fp8_gemm(
                        inputs,
                        input_scales,
                        weight,
                        weight_scales,
                        torch.float8_e8m0fnu,
                        block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        actual, (factors * value).to(dtype), rtol=0, atol=0
                    )

    def test_sparse_reference_rounds_unnormalized_probabilities_before_values(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16)
        query[..., 0] = 32
        kv = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
        kv[0, :, 0] = torch.tensor([0.0, -0.5])
        kv[0, :, 1] = torch.tensor([1.0, -1.0])
        sink = torch.full((16,), -2.0)
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        actual = kernel.sparse_attn(query, kv, sink, indices, 512**-0.5)
        torch.testing.assert_close(
            actual[..., 1],
            torch.full((1, 1, 16), 0.3125, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_sparse_sink_does_not_change_probability_rounding(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
        window = torch.ones((1, 1, 512), dtype=torch.bfloat16, device="cuda")
        sink = torch.ones(16, dtype=torch.float32, device="cuda")
        indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
        actual = kernel._sparse_attn_native(query, sink, indices, window, window[:, :0])
        self.assertIsNotNone(actual)
        torch.testing.assert_close(
            actual, torch.full_like(query, 0.26953125), rtol=0, atol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_sparse_preserves_bf16_kv_without_requantization(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
        sink = torch.zeros(16, dtype=torch.float32, device="cuda")
        indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
        for compressed, value in ((False, 1.0078125), (True, 0.015625)):
            with self.subTest(compressed=compressed):
                table = torch.full_like(query[:, :, 0], value)
                empty = table[:, :0]
                window, history = (empty, table) if compressed else (table, empty)
                actual = kernel._sparse_attn_native(
                    query, sink, indices, window, history
                )
                # One zero-score key and a zero sink each contribute one unit.
                torch.testing.assert_close(
                    actual, torch.full_like(query, value / 2), rtol=0, atol=0
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_sparse_keeps_source_denominator_reduction_order(self):
        generator = torch.Generator(device="cpu").manual_seed(20261030)
        query = (
            torch.randint(-8, 9, (1, 64, 16, 512), generator=generator, device="cpu")
            .float()
            .mul(0.125)
            .to(device="cuda", dtype=torch.bfloat16)
        )
        table = (
            torch.randint(-8, 9, (1, 128, 512), generator=generator, device="cpu")
            .float()
            .mul(0.25)
            .to(device="cuda", dtype=torch.bfloat16)
        )
        sink = torch.full((16,), 0.25, dtype=torch.float32, device="cuda")
        indices = torch.arange(128, device="cuda", dtype=torch.int32)
        indices = indices.view(1, 1, 128).expand(1, 64, 128).contiguous()
        actual = kernel._sparse_attn_native(query, sink, indices, table, table[:, :0])
        # Independent released-kernel outputs for this checkpoint-free recipe.
        for row, head, column, expected in (
            (2, 4, 416, 0.025390625),
            (15, 11, 338, -0.220703125),
            (19, 2, 288, 0.314453125),
            (28, 11, 53, 0.050048828125),
            (58, 5, 392, -0.169921875),
            (62, 8, 94, -0.169921875),
        ):
            with self.subTest(row=row, head=head, column=column):
                self.assertEqual(float(actual[0, row, head, column]), expected)

    def test_sparse_attention_preserves_ordered_blocks_and_masked_slots(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        # Released reference: one block and a later maximum have different rounding.
        cases = (
            ([1, 0], -2.0, 0.3125),
            ([1] + [-1] * 63 + [0], -2.0, 0.310546875),
            ([1] + [-1] * 127 + [0], -2.0, 0.310546875),
            ([0] * 65, -2.0, 0.99609375),
            ([-1] * 65, -2.0, 0.0),
            ([-1] * 65, -1e31, 0.0),
            ([], -1e31, 0.0),
        )
        for device in devices:
            query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16, device=device)
            query[..., 0] = 32
            kv = torch.zeros((1, 2, 512), dtype=torch.bfloat16, device=device)
            kv[0, 1, 0] = -0.5
            kv[0, 0, 1], kv[0, 1, 1] = 1, -1
            for selected, sink_value, expected in cases:
                with self.subTest(device=device, indices=selected, sink=sink_value):
                    sink = torch.full((16,), sink_value, device=device)
                    indices = torch.tensor(
                        [[selected]], dtype=torch.int32, device=device
                    )
                    actual = kernel.sparse_attn(query, kv, sink, indices, 512**-0.5)
                    torch.testing.assert_close(
                        actual[..., 1],
                        torch.full_like(actual[..., 1], expected),
                        atol=0,
                        rtol=0,
                    )
                    if expected == 0:
                        torch.testing.assert_close(actual, torch.zeros_like(actual))
                    if device == "cuda":
                        native = kernel._sparse_attn_native(
                            query, sink, indices, kv, kv[:, :0]
                        )
                        self.assertIsNotNone(native)
                        torch.testing.assert_close(native, actual, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp32_projection_replays_on_caller_stream(self):
        for rows, width, outputs in (
            (1, 96, 32),
            (1, 1056, 512),
            (2, 128, 32),
            (2, 1024, 512),
            (32, 1024, 1280),
            (33, 1056, 1088),
            (65, 1056, 1088),
        ):
            encoded = torch.ones((rows, width), device="cuda").to(torch.float8_e4m3fn)
            scales = torch.ones((rows, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            weight = torch.ones((outputs, width), device="cuda").to(torch.float8_e4m3fn)
            weight_scales = torch.ones((outputs // 32, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            module = kernel_native._projection_module()
            proxy = Mock(wraps=module)
            with patch.object(
                kernel_projection, "_projection_module", return_value=proxy
            ):
                output = kernel.fp8_gemm(
                    encoded,
                    scales,
                    weight,
                    weight_scales,
                    torch.float8_e8m0fnu,
                    block_size=32,
                    output_dtype=torch.float32,
                )
            arguments = proxy.project.call_args.args
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output, torch.full_like(output, width), rtol=0, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    module.project(*arguments)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    encoded.view(torch.uint8).fill_(64)
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, 2 * width), rtol=0, atol=0
                )
            finally:
                torch.cuda.synchronize()
                graph.reset()

    def test_hyperconnection_sinkhorn_preserves_zero_probability_support(self):
        mixes = torch.zeros((1, 1, 24), dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.tensor(
            [0.0] * 8 + [0.0, -1000.0, -1000.0, -1000.0] * 4,
            dtype=torch.float32,
        )
        expected = torch.full((1, 1, 4, 4), 0.24999975, dtype=torch.float32)
        actual = kernel.hc_split_sinkhorn(mixes, scale, base)[2]
        with self.subTest(backend="reference"):
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
        if torch.cuda.is_available():
            actual = kernel.hc_prepare(
                torch.ones((1, 1, 4, 5120), dtype=torch.bfloat16, device="cuda"),
                torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device="cuda"
                ),
                torch.zeros((24, 20480), dtype=torch.float32, device="cuda"),
                scale.cuda(),
                base.cuda(),
            )[3]
            with self.subTest(backend="cuda"):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=2e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_retains_fp32_partials_before_reduction(self):
        for rows, width in ((1, 32), (33, 128), (33, 160)):
            encoded = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
            encoded[:, :2] = torch.tensor([1, 1 / 256], device="cuda")
            encoded = encoded.to(torch.float8_e4m3fn)
            input_scales = torch.ones((rows, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            for bits in (4, 8):
                with self.subTest(bits=bits, rows=rows, width=width):
                    if bits == 4:
                        weight = torch.full(
                            (32, width // 2), 0x22, dtype=torch.uint8, device="cuda"
                        ).view(torch.float4_e2m1fn_x2)
                        scales = torch.ones((32, width // 32), device="cuda").to(
                            torch.float8_e8m0fnu
                        )
                        gemm = kernel.fp4_gemm
                        kwargs = {"act_block_size": 32}
                    else:
                        weight = torch.ones((32, width), device="cuda").to(
                            torch.float8_e4m3fn
                        )
                        scales = torch.ones((1, width // 32), device="cuda").to(
                            torch.float8_e8m0fnu
                        )
                        gemm = kernel.fp8_gemm
                        kwargs = {"block_size": 32}
                    inputs = (
                        encoded,
                        input_scales,
                        weight,
                        scales,
                        torch.float8_e8m0fnu,
                    )
                    actual = gemm(*inputs, output_dtype=torch.float32, **kwargs)
                    torch.testing.assert_close(
                        actual,
                        torch.full(
                            (rows, 32), 257 / 256, dtype=torch.float32, device="cuda"
                        ),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(actual.dtype, torch.float32)
                    rounded = gemm(*inputs, **kwargs)
                    torch.testing.assert_close(
                        rounded,
                        torch.ones_like(actual, dtype=torch.bfloat16),
                        rtol=0,
                        atol=0,
                    )
                    empty = gemm(
                        encoded[:0],
                        input_scales[:0],
                        *inputs[2:],
                        output_dtype=torch.float32,
                        **kwargs,
                    )
                    self.assertEqual(empty.shape, (0, 32))
                    self.assertEqual(empty.dtype, torch.float32)
                    with self.assertRaisesRegex(ValueError, "output dtype"):
                        gemm(*inputs, output_dtype=torch.float16, **kwargs)

    def _fp8_rounding_inputs(self, rows, width, outputs):
        inputs = torch.zeros((rows, width), device="cuda")
        weight = torch.zeros((outputs, width), device="cuda")
        for column, left, right in (
            (8, -1, 1),
            (32, 16, 16),
            (48, 8, 8),
            (63, 3, 1),
        ):
            inputs[:, column] = left
            weight[:, column] = right
        inputs = inputs.to(torch.float8_e4m3fn)
        weight = weight.to(torch.float8_e4m3fn)
        row_shifts = torch.arange(rows, device="cuda") % 3 - 1
        group_shifts = torch.arange(outputs // 32, device="cuda") % 5 - 2
        input_scales = (
            (
                torch.tensor([113] + [127] * (width // 32 - 1), device="cuda")
                + row_shifts[:, None]
            )
            .to(torch.uint8)
            .view(torch.float8_e8m0fnu)
        )
        weight_scales = (
            (
                torch.tensor([126, 128] + [127] * (width // 32 - 2), device="cuda")
                + group_shifts[:, None]
            )
            .to(torch.uint8)
            .view(torch.float8_e8m0fnu)
        )
        return inputs, weight, row_shifts, group_shifts, input_scales, weight_scales
