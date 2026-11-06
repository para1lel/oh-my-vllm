import gc
import unittest
import weakref
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import (
    kernel,
    kernel_native,
    kernel_projection,
    kernel_weights,
)

from tests.deepseek_operators_fixtures import projection_reference, projection_weight


class DeepSeekProjectionStorageContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_reuses_compatible_encoded_storage(self):
        for bits, outputs, width, offset in (
            (4, 32, 128, 64),
            (4, 576, 128, 64),
            (8, 64, 128, 16),
            (8, 64, 576, 16),
        ):
            with self.subTest(bits=bits, outputs=outputs, width=width):
                weight, scales = projection_weight(bits, outputs, width, offset=offset)
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
                    packed = kernel_weights._packed_weight(weight, scales, bits)
                    self.assertIs(
                        kernel_weights._packed_weight(weight, scales, bits), packed
                    )
                    self.assertEqual(packed[0].data_ptr(), weight.data_ptr())
                    self.assertEqual(
                        packed[0].untyped_storage().data_ptr(),
                        weight.untyped_storage().data_ptr(),
                    )
                    self.assertNotEqual(packed[1].data_ptr(), scales.data_ptr())
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_alias_retains_source_storage(self):
        for bits in (4, 8):
            with (
                self.subTest(bits=bits),
                patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True),
            ):
                weight, scales = projection_weight(bits, 32, 128, offset=64)
                pointer = weight.data_ptr()
                storage_pointer = weight.untyped_storage().data_ptr()
                expected = weight.view(torch.uint8).clone()
                kernel_weights._packed_weight(weight, scales, bits)
                key = next(iter(kernel_weights._PACKED_WEIGHTS))
                del weight, scales
                gc.collect()
                retained = kernel_weights._PACKED_WEIGHTS[key].packed[0]
                self.assertEqual(retained.data_ptr(), pointer)
                self.assertEqual(retained.untyped_storage().data_ptr(), storage_pointer)
                torch.testing.assert_close(retained, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_cache_keeps_original_owners_for_padded_weights(self):
        with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
            weight, scales = projection_weight(4, 32, 64, offset=0)
            weight_reference, scales_reference = (
                weakref.ref(weight),
                weakref.ref(scales),
            )
            packed = kernel_weights._packed_weight(weight, scales, 4)
            self.assertNotEqual(packed[0].data_ptr(), weight.data_ptr())
            self.assertNotEqual(packed[1].data_ptr(), scales.data_ptr())
            torch.cuda.synchronize()
            del weight, scales
            gc.collect()
            self.assertIsNotNone(weight_reference())
            self.assertIsNotNone(scales_reference())
            self.assertIs(
                kernel_weights._packed_weight(
                    weight_reference(), scales_reference(), 4
                ),
                packed,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_cache_does_not_reuse_different_source_views(self):
        with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
            weight, scales = projection_weight(4, 64, 64, offset=0)
            small = kernel_weights._packed_weight(weight[:32], scales[:32], 4)
            full = kernel_weights._packed_weight(weight, scales, 4)
            self.assertEqual(full[0].size(0), 64)
            self.assertIsNot(small, full)
            self.assertIs(
                kernel_weights._packed_weight(weight[:32], scales[:32], 4), small
            )
            with self.assertRaisesRegex(Exception, "Check failed"):
                kernel_weights._packed_weight(weight, scales[:32], 4)
            self.assertIs(kernel_weights._packed_weight(weight, scales, 4), full)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_cache_keeps_warmed_views_for_graph_replay(self):
        from tests.projection_precision_fixtures import fp4_literal

        encoded, scales, weight, weight_scales, factors = fp4_literal(1)

        def run(outputs):
            return kernel.fp4_gemm(
                encoded,
                scales,
                weight[:outputs],
                weight_scales[:outputs],
                torch.float8_e8m0fnu,
                act_block_size=32,
            )

        with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
            run(64)
            run(128)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    output = run(64)
                run(128)
                output.fill_(-7)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, (factors[:, :64] * 648).bfloat16(), rtol=0, atol=0
                )
            finally:
                torch.cuda.synchronize()
                graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_keeps_incompatible_values_separate(self):
        for width, offset, overlap in (
            (576, 0, False),
            (128, 16, False),
            (128, 0, True),
        ):
            with self.subTest(width=width, offset=offset, overlap=overlap):
                weight, scales = projection_weight(4, 32, width, offset=offset)
                if overlap:
                    scales = (
                        weight.view(torch.uint8)
                        .flatten()[: 32 * (width // 32)]
                        .view(32, width // 32)
                        .view(torch.float8_e8m0fnu)
                    )
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
                    packed = kernel_weights._packed_weight(weight, scales, 4)
                    self.assertNotEqual(packed[0].data_ptr(), weight.data_ptr())
                    padded_width = (width + 127) // 128 * 128
                    self.assertEqual(tuple(packed[0].shape), (32, padded_width // 2))
                    torch.testing.assert_close(packed[0][:, : width // 2], raw_before)
                    torch.testing.assert_close(
                        packed[0][:, width // 2 :],
                        torch.zeros_like(packed[0][:, width // 2 :]),
                    )
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_rejects_invalid_source_views(self):
        weight, scales = projection_weight(4, 32, 128)
        unaligned, _ = projection_weight(4, 32, 128, offset=1)
        for invalid in (unaligned, weight.transpose(0, 1)):
            with (
                self.subTest(stride=invalid.stride()),
                patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True),
                self.assertRaises(RuntimeError),
            ):
                kernel_weights._packed_weight(invalid, scales, 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_alias_matches_copy_and_quantized_reference(self):
        for bits, width in ((4, 128), (4, 576), (8, 128), (8, 576)):
            with self.subTest(bits=bits, width=width):
                weight, scales = projection_weight(
                    bits, 64, width, offset=64 if bits == 4 else 16
                )
                values = torch.linspace(-2, 2, 3 * width, device="cuda").view(3, width)
                encoded, input_scales = kernel.act_quant(
                    values, 32, "ue8m0", torch.float8_e8m0fnu
                )
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
                    actual = kernel_projection._projection_gemm(
                        encoded, input_scales, weight, scales, bits
                    )
                    packed = kernel_weights._packed_weight(weight, scales, bits)
                    copy = torch.empty_like(packed[0])
                    copied_scales = torch.empty_like(packed[1])
                    kernel_native._projection_module().pack_weights(
                        weight.view(torch.uint8),
                        scales.view(torch.uint8),
                        copy,
                        copied_scales,
                        bits,
                    )
                    with patch.object(
                        kernel_projection,
                        "_packed_weight",
                        return_value=(copy, copied_scales, width),
                    ):
                        copied_output = kernel_projection._projection_gemm(
                            encoded, input_scales, weight, scales, bits
                        )
                    torch.testing.assert_close(actual, copied_output, atol=0, rtol=0)
                    expected = projection_reference(
                        encoded, input_scales, weight, scales, bits
                    )
                    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.02)
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)
