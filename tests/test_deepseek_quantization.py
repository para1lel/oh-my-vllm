import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel


class DeepSeekQuantization(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        device = torch.device(self.device)
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)

    def test_fp8_inplace_rounds_before_dequantization(self):
        for block in (32, 128):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(block=block, dtype=dtype):
                    values = torch.tensor(
                        (448, 432, 416, 1.0625, 1.1875, -1.0625, -1.1875, 0)
                        * (block // 8),
                        dtype=dtype,
                    ).reshape(1, block)
                    expected = torch.tensor(
                        (448, 448, 416, 1, 1.25, -1, -1.25, 0) * (block // 8),
                        dtype=dtype,
                    ).reshape(1, block)

                    actual = kernel.act_quant(values, block, "ue8m0", inplace=True)

                    self.assertIs(actual, values)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_fp8_unrounded_scale_divides_the_block_maximum_by_448(self):
        values = torch.tensor((448, 224, -112, 0) * 8, dtype=torch.bfloat16).view(1, 32)

        encoded, scales = kernel.act_quant(values, 32)

        torch.testing.assert_close(scales, torch.ones((1, 1)), atol=0, rtol=0)
        torch.testing.assert_close(encoded.float(), values.float(), atol=0, rtol=0)

    def test_fp8_unrounded_normalization_keeps_released_ue8m0_storage_rounding(self):
        values = torch.tensor((336, 168, 84, -336) * 8, dtype=torch.bfloat16).view(
            1, 32
        )

        encoded, scales = kernel.act_quant(values, 32, scale_dtype=torch.float8_e8m0fnu)

        expected = torch.tensor((448, 224, 112, -448) * 8).float().view(1, 32)
        torch.testing.assert_close(encoded.float(), expected, atol=0, rtol=0)
        torch.testing.assert_close(scales.float(), torch.ones((1, 1)), atol=0, rtol=0)

    def test_fp4_inplace_uses_e4m3_scale_and_nearest_even_ties(self):
        values = torch.tensor(
            (
                6,
                5,
                3.5,
                2.5,
                1.75,
                1.25,
                0.75,
                0.25,
                -6,
                -5,
                -3.5,
                -2.5,
                -1.75,
                -1.25,
                -0.75,
                -0.25,
            ),
            dtype=torch.bfloat16,
        ).view(1, 16)
        expected = torch.tensor(
            (6, 4, 4, 2, 2, 1, 1, 0, -6, -4, -4, -2, -2, -1, -1, -0.0),
            dtype=torch.bfloat16,
        ).view(1, 16)

        actual = kernel.fp4_act_quant(
            values, 16, inplace=True, scale_dtype=torch.float8_e4m3fn
        )

        self.assertIs(actual, values)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual.signbit(), expected.signbit())

    def test_fp4_encoded_bytes_and_scale_are_released_format(self):
        values = torch.tensor(
            (
                6,
                5,
                3.5,
                2.5,
                1.75,
                1.25,
                0.75,
                0.25,
                -6,
                -5,
                -3.5,
                -2.5,
                -1.75,
                -1.25,
                -0.75,
                -0.25,
            )
            * 2,
            dtype=torch.bfloat16,
        ).view(1, 32)

        encoded, scales = kernel.fp4_act_quant(values)

        self.assertEqual(encoded.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(encoded.shape), (1, 16))
        torch.testing.assert_close(
            encoded.view(torch.uint8),
            torch.tensor(
                (0x67, 0x46, 0x24, 0x02, 0xEF, 0xCE, 0xAC, 0x8A) * 2,
                dtype=torch.uint8,
            ).view(1, 16),
        )
        torch.testing.assert_close(scales.float(), torch.ones((1, 1)), atol=0, rtol=0)

    def test_fp4_scale_rounding_precedes_value_quantization(self):
        values = torch.zeros((1, 16), dtype=torch.bfloat16)
        values[0, :2] = torch.tensor((6.375, 1.3125))

        actual = kernel.fp4_act_quant(
            values, 16, inplace=True, scale_dtype=torch.float8_e4m3fn
        )

        torch.testing.assert_close(
            actual[0, :2], torch.tensor((6, 1.5), dtype=torch.bfloat16), atol=0, rtol=0
        )

    def test_fp4_zero_groups_have_the_released_nonzero_scale(self):
        for dtype, scale in (
            (torch.float8_e4m3fn, 2.0**-9),
            (torch.float8_e8m0fnu, 2.0**-126),
        ):
            with self.subTest(dtype=dtype):
                encoded, scales = kernel.fp4_act_quant(
                    torch.zeros((1, 32), dtype=torch.bfloat16), scale_dtype=dtype
                )
                torch.testing.assert_close(
                    scales.float(), torch.full((1, 1), scale), atol=0, rtol=0
                )
                self.assertEqual(encoded.view(torch.uint8).count_nonzero(), 0)

    def test_scale_ceiling_does_not_lose_the_fraction_at_large_exponents(self):
        for bits, maximum in ((4, 6), (8, 448)):
            with self.subTest(bits=bits):
                values = torch.zeros((1, 32), dtype=torch.float32)
                values[0, 0] = torch.nextafter(
                    torch.tensor(maximum * 2.0**64), torch.tensor(torch.inf)
                )

                if bits == 4:
                    _, scales = kernel.fp4_act_quant(values)
                else:
                    _, scales = kernel.act_quant(
                        values, 32, "ue8m0", torch.float8_e8m0fnu
                    )

                torch.testing.assert_close(
                    scales.float(), torch.full((1, 1), 2.0**65), atol=0, rtol=0
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepSeekQuantizationCUDA(DeepSeekQuantization):
    device = "cuda"

    def test_fp4_projection_expands_128_element_activation_scale_groups(self):
        inputs = torch.ones((1, 512)).to(torch.float8_e4m3fn)
        weight = torch.full((128, 256), 0x22, dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        for dtype in (torch.float8_e8m0fnu, torch.float32):
            with self.subTest(dtype=dtype):
                input_scales = torch.ones((1, 4)).to(dtype)
                weight_scales = torch.ones((128, 16)).to(dtype)

                actual = kernel.fp4_gemm(
                    inputs,
                    input_scales,
                    weight,
                    weight_scales,
                    dtype,
                    act_block_size=128,
                )

                expected_dtype = (
                    torch.bfloat16 if dtype == torch.float8_e8m0fnu else torch.float32
                )
                torch.testing.assert_close(
                    actual,
                    torch.full((1, 128), 512, dtype=expected_dtype),
                    atol=0,
                    rtol=0,
                )

    def test_native_quantization_rejects_invalid_contracts_before_writes(self):
        from oh_my_vllm.operators.runtime.native import load
        from oh_my_vllm.operators.transform.api import FUNCTIONS

        module = load("transform", FUNCTIONS)[0]
        values = torch.ones((2, 32), dtype=torch.bfloat16)
        codes = torch.full((2, 16), 0xA5, dtype=torch.uint8)
        scales = torch.full((2, 1), 0xA5, dtype=torch.uint8)
        module.quantize(values, codes, scales, 4, 32, 1, False, False)
        torch.cuda.synchronize()
        torch.testing.assert_close(codes, torch.full_like(codes, 0x66))

        alias = values.view(torch.uint8).reshape(-1)[:32].view(2, 16)
        variants = (
            (values.float(), codes, scales, 4, 32, 1),
            (values, codes, scales, 7, 32, 1),
            (values, codes, scales, 4, 0, 1),
            (values, codes, scales, 4, 32, 0),
            (values, codes, scales, 8, 32, 2),
            (values, codes, scales[:1], 4, 32, 1),
            (values, alias, scales, 4, 32, 1),
            (values, codes, codes[:, :1], 4, 32, 1),
        )
        for inputs, output, scale_output, bits, block, fmt in variants:
            with self.subTest(bits=bits, block=block, format=fmt):
                saved_output, saved_scales = output.clone(), scale_output.clone()
                with self.assertRaises((RuntimeError, ValueError)):
                    module.quantize(
                        inputs, output, scale_output, bits, block, fmt, False, False
                    )
                torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
                torch.testing.assert_close(scale_output, saved_scales, atol=0, rtol=0)

    def test_quantization_uses_the_nondefault_producer_stream(self):
        values = torch.zeros((64, 32), dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000000)
            values.fill_(6)
            codes, scales = kernel.fp4_act_quant(values)
            consumed = codes.view(torch.uint8).clone()
        stream.synchronize()
        torch.testing.assert_close(consumed, torch.full_like(consumed, 0x77))
        torch.testing.assert_close(scales.float(), torch.ones((64, 1)), atol=0, rtol=0)

    def test_kv_packing_accepts_empty_history(self):
        values = torch.empty((2, 0, 512), dtype=torch.bfloat16)
        for pack, width, groups in (
            (kernel._pack_fp4_kv, 256, 32),
            (kernel._pack_fp8_kv, 512, 16),
        ):
            with self.subTest(pack=pack.__name__):
                codes, scales = pack(values)
                self.assertEqual(codes.shape, (2, 0, width))
                self.assertEqual(scales.shape, (2, 0, groups))

    def test_nonfinite_compressed_scales_fail_native_attention_closed(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16)
        window = torch.ones((1, 1, 512), dtype=torch.bfloat16)
        indices = torch.ones((1, 1, 1), dtype=torch.int32)
        sink = torch.zeros(16, dtype=torch.float32)
        for value in (torch.nan, torch.inf):
            with self.subTest(value=value):
                compressed = torch.full_like(window, value)
                self.assertIsNone(
                    kernel._sparse_attn_native(query, sink, indices, window, compressed)
                )

    def test_native_attention_preserves_released_quantized_key_values(self):
        window = torch.full((1, 1, 512), 2, dtype=torch.bfloat16)
        compressed = torch.full_like(window, 4.125)
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        for heads in (16, 64):
            with self.subTest(heads=heads):
                query = torch.zeros((1, 1, heads, 512), dtype=torch.bfloat16)

                actual = kernel._sparse_attn_native(
                    query,
                    torch.zeros(heads, dtype=torch.float32),
                    indices,
                    window,
                    compressed,
                )

                self.assertIsNotNone(actual)
                torch.testing.assert_close(
                    actual, torch.full_like(query, 2.046875), atol=0, rtol=0
                )


class DeepSeekQuantizationIntegration(unittest.TestCase):
    def test_fp8_reference_gemm_expands_weight_scales_in_both_dimensions(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for block in (32, 128):
                with self.subTest(device=device, block=block), torch.device(device):
                    inputs = torch.ones((1, block)).to(torch.float8_e4m3fn)
                    weight = torch.ones((2 * block, block)).to(torch.float8_e4m3fn)

                    actual = kernel.fp8_gemm(
                        inputs,
                        torch.ones((1, 1)),
                        weight,
                        torch.tensor([[1.0], [2.0]]),
                        block_size=block,
                    )

                    torch.testing.assert_close(
                        actual,
                        torch.tensor(
                            [(float(block),) * block + (float(2 * block),) * block]
                        ),
                        atol=0,
                        rtol=0,
                    )

    def test_fp4_reference_gemm_reads_both_values_in_each_byte(self):
        weight = torch.full((1, 16), 0x21, dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        inputs = torch.ones((1, 32)).to(torch.float8_e4m3fn)

        actual = kernel.fp4_gemm(
            inputs, torch.ones((1, 1)), weight, torch.ones((1, 1)), act_block_size=32
        )

        torch.testing.assert_close(actual, torch.tensor([[24.0]]), atol=0, rtol=0)

    def test_sparse_fallback_joins_the_separate_key_tables_when_needed(self):
        query = torch.zeros((1, 1, 1, 4))
        window = torch.ones((1, 1, 4))
        compressed = torch.full_like(window, 2)
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)

        actual = kernel.sparse_attn(
            query,
            None,
            torch.zeros(1),
            indices,
            0.5,
            native_window=window,
            native_compressed=compressed,
        )

        torch.testing.assert_close(actual, torch.ones_like(query), atol=0, rtol=0)

    def test_attention_passes_separate_tables_without_a_dense_key_copy(self):
        from oh_my_vllm.models.deepseek_v41 import model

        args = model.ModelArgs(
            dim=64,
            n_heads=4,
            q_lora_rank=32,
            head_dim=32,
            rope_head_dim=16,
            o_groups=4,
            o_lora_rank=16,
            window_size=4,
            compress_ratios=(2,),
            kv_source_layers=(0,),
            index_source_layers=(0,),
            index_n_heads=4,
            index_head_dim=32,
            index_topk=4,
            max_batch_size=1,
            max_seq_len=8,
        )

        def attend(query, combined, *arguments, **keywords):
            self.assertIsNone(combined)
            self.assertEqual(keywords["native_window"].size(1), 4)
            self.assertEqual(keywords["native_compressed"].size(1), 2)
            return torch.zeros_like(query)

        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with (
                patch.object(model, "default_dtype", torch.bfloat16),
                patch.object(model, "world_size", 1),
                patch.object(model, "rank", 0),
            ):
                attention = model.Attention(0, args)
                for parameter in attention.parameters():
                    torch.nn.init.zeros_(parameter)
                with patch.object(model, "sparse_attn", side_effect=attend):
                    attention(torch.zeros((1, 4, 64), dtype=torch.bfloat16), 0)
        finally:
            torch.set_default_dtype(previous_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepSeekQuantizationMemory(unittest.TestCase):
    def test_kv_packing_scratch_does_not_grow_with_history(self):
        for rows in (4096, 8192):
            for pack in (kernel._pack_fp4_kv, kernel._pack_fp8_kv):
                with self.subTest(rows=rows, pack=pack.__name__):
                    values = torch.zeros(
                        (2, rows, 512), device="cuda", dtype=torch.bfloat16
                    )
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()

                    codes, scales = pack(values)
                    torch.cuda.synchronize()

                    output_bytes = codes.numel() + scales.numel()
                    scratch = (
                        torch.cuda.max_memory_allocated() - baseline - output_bytes
                    )
                    self.assertLessEqual(scratch, 1 << 20)
                    del values, codes, scales
