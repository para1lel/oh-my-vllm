import unittest

import numpy as np
import torch

from .qwen_fp8_reference import (
    KVPages,
    attention_bf16_vectors,
    decode_e4m3fn,
    encode_e4m3fn,
    quantize_bf16_vectors,
    reconstruct_vectors,
)


def bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    if np.any(words & 0xFFFF):
        raise ValueError("test input must already be exactly representable in BF16")
    return (words >> 16).astype(np.uint16)


def constant_rows(count, value=448.0):
    return bf16_bits(np.full((count, 2, 4, 256), value, dtype=np.float32))


class E4M3Format(unittest.TestCase):
    def test_representative_bits_and_signed_zero(self):
        codes = np.array(
            [0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0x40, 0x77, 0x78, 0x7E, 0xB8, 0xFE],
            dtype=np.uint8,
        )
        expected = np.array(
            [0.0, -0.0, 2.0**-9, 7 * 2.0**-9, 2.0**-6, 1, 2, 240, 256, 448, -1, -448]
        )
        np.testing.assert_array_equal(decode_e4m3fn(codes), expected)
        np.testing.assert_array_equal(
            np.signbit(decode_e4m3fn(codes)), np.signbit(expected)
        )
        np.testing.assert_array_equal(encode_e4m3fn(expected), codes)

    def test_nearest_even_subnormals_and_finite_saturation(self):
        values = np.array(
            [2.0**-10, 3 * 2.0**-10, 1.0625, 1.1875, 1.9375, 432, 464, 480, 1.0e300]
        )
        expected = np.array([0, 2, 0x38, 0x3A, 0x40, 0x7E, 0x7E, 0x7E, 0x7E], np.uint8)
        np.testing.assert_array_equal(encode_e4m3fn(values), expected)
        np.testing.assert_array_equal(encode_e4m3fn(-values), expected | 128)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_e4m3fn(np.array([value]))
        for code in (0x7F, 0xFF):
            with self.subTest(code=code), self.assertRaises(ValueError):
                decode_e4m3fn(np.array([code], np.uint8))

    def test_all_finite_bytes_against_independent_torch_cpu(self):
        codes = np.array([i for i in range(256) if (i & 127) != 127], np.uint8)
        expected = torch.from_numpy(codes).view(torch.float8_e4m3fn).double().numpy()
        actual = decode_e4m3fn(codes)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        midpoints = np.array([1.0625, 1.1875, 1.9375, 432, 464, 480], np.float32)
        expected_bits = (
            torch.from_numpy(midpoints).to(torch.float8_e4m3fn).view(torch.uint8)
        )
        np.testing.assert_array_equal(encode_e4m3fn(midpoints), expected_bits.numpy())
        self.assertFalse(torch.cuda.is_initialized())


class BF16Vectors(unittest.TestCase):
    def test_zero_extreme_and_smallest_bf16(self):
        bits = np.zeros((4, 256), np.uint16)
        bits[0, 1] = 0x8000
        bits[1, 0], bits[1, 1] = 0x7F7F, 0xFF7F
        bits[2, 0], bits[2, 1] = 0x0001, 0x8001
        bits[3, 0] = 0x0080
        codes, scales = quantize_bf16_vectors(bits)
        self.assertEqual(scales.dtype, np.float32)
        self.assertEqual(scales[0], 1)
        self.assertEqual(codes[0, 0], 0)
        self.assertEqual(codes[0, 1], 128)
        self.assertEqual(scales[1], np.float32(float.fromhex("0x1.fep127") / 448))
        np.testing.assert_array_equal(codes[1, :2], [0x7E, 0xFE])
        self.assertEqual(scales[2], np.float32(146 * 2.0**-149))
        np.testing.assert_array_equal(codes[2, :2], [0x7E, 0xFE])
        self.assertEqual(scales[3], np.float32(18725 * 2.0**-149))
        restored = reconstruct_vectors(codes, scales)
        self.assertEqual(restored.dtype, np.float64)
        self.assertTrue(np.isfinite(restored).all())
        self.assertEqual(restored[2, 0], 65408 * 2.0**-149)
        self.assertTrue(np.signbit(restored[0, 1]))

    def test_independent_k_v_head_scales_and_high_precision_product(self):
        magnitudes = np.array([[448, 224, 896, 112], [28, 56, 1792, 3584]], np.float32)
        values = np.broadcast_to(magnitudes[..., None], (2, 4, 256)).copy()
        values[..., 1] = -magnitudes
        codes, scales = quantize_bf16_vectors(bf16_bits(values))
        np.testing.assert_array_equal(
            scales, [[1, 0.5, 2, 0.25], [0.0625, 0.125, 4, 8]]
        )
        np.testing.assert_array_equal(codes[..., 0], np.full((2, 4), 0x7E))
        np.testing.assert_array_equal(codes[..., 1], np.full((2, 4), 0xFE))
        np.testing.assert_array_equal(reconstruct_vectors(codes, scales), values)
        codes = np.full((1, 256), 0x38, np.uint8)
        codes[0, 1] = 0x7E
        scales = np.array([np.float32(1 / 448)], np.float32)
        actual = reconstruct_vectors(codes, scales)
        self.assertEqual(actual[0, 0], float(scales[0]))
        self.assertEqual(actual[0, 1], 448 * float(scales[0]))
        self.assertNotEqual(actual[0, 1], float(np.float32(448 * scales[0])))

    def test_mixed_magnitudes_underflow_and_cancellation(self):
        values = np.zeros((1, 256), np.float32)
        values[0, :10] = [
            448,
            -448,
            1,
            -1,
            0.001953125,
            -0.001953125,
            0.0009765625,
            -0.0009765625,
            1.0625,
            -1.0625,
        ]
        codes, scales = quantize_bf16_vectors(bf16_bits(values))
        self.assertEqual(scales[0], 1)
        np.testing.assert_array_equal(
            codes[0, :10], [0x7E, 0xFE, 0x38, 0xB8, 1, 129, 0, 128, 0x38, 0xB8]
        )
        reconstructed = reconstruct_vectors(codes, scales)
        self.assertEqual(np.sum(reconstructed, dtype=np.float64), 0)
        self.assertTrue(np.signbit(reconstructed[0, 7]))

    def test_attention_bf16_rounding_and_new_replay_agreement(self):
        codes = np.full((3, 256), 0x38, np.uint8)
        scales = np.array([1.00390625, 1.01171875, 1.0], np.float32)
        expected = np.empty((3, 256), np.uint16)
        expected[0], expected[1], expected[2] = 0x3F80, 0x3F82, 0x3F80
        np.testing.assert_array_equal(attention_bf16_vectors(codes, scales), expected)
        values = np.full((2, 2, 4, 256), 448.0, np.float32)
        values[..., 1] = 1.0625
        values[..., 2] = -1.1875
        rows = bf16_bits(values)
        new_codes, new_scales = quantize_bf16_vectors(rows)
        new_attention = attention_bf16_vectors(new_codes, new_scales)
        pages = KVPages.allocate(2)
        pages.update([1, 0], 127, rows)
        replay_attention = pages.read_attention_bf16([1, 0], 127, 2)
        expected = rows.copy()
        expected[..., 1], expected[..., 2] = 0x3F80, 0xBFA0
        np.testing.assert_array_equal(new_attention, expected)
        np.testing.assert_array_equal(replay_attention, expected)
        self.assertFalse(np.array_equal(new_attention, rows))
        torch_expected = torch.from_numpy(codes).view(torch.float8_e4m3fn).float()
        torch_expected = (torch_expected * torch.from_numpy(scales[:, None])).bfloat16()
        np.testing.assert_array_equal(
            attention_bf16_vectors(codes, scales),
            torch_expected.view(torch.uint16).numpy(),
        )

    def test_nonfinite_and_invalid_shapes_are_errors(self):
        for code in (0x7F80, 0xFF80, 0x7FC0):
            bits = np.zeros((1, 256), np.uint16)
            bits[0, 42] = code
            with self.subTest(code=code), self.assertRaises(ValueError):
                quantize_bf16_vectors(bits)
        for bits in (np.zeros((1, 255), np.uint16), np.zeros((1, 256), np.float32)):
            with self.assertRaises(ValueError):
                quantize_bf16_vectors(bits)
        codes = np.zeros((1, 256), np.uint8)
        for scale in (0, -1, np.inf, np.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                reconstruct_vectors(codes, np.array([scale], np.float32))
        with self.assertRaises(ValueError):
            reconstruct_vectors(codes, np.ones((1, 1), np.float32))


class LogicalPages(unittest.TestCase):
    def test_127_128_129_valid_rows_and_guard_storage(self):
        for count in (127, 128, 129):
            with self.subTest(count=count):
                shape = (6, 128, 2, 4)
                payload = np.full((*shape, 256), 0x55, np.uint8)
                scales = np.full(shape, -123.0, np.float32)
                valid = np.zeros(shape[:2], np.bool_)
                pages = KVPages(payload[1:-1], scales[1:-1], valid[1:-1])
                rows = constant_rows(count + 2)
                rows[count:] = 0x7FC0
                pages.update([2, 0], 0, rows, valid_rows=count)
                np.testing.assert_array_equal(pages.read([2, 0], 0, count), 448)
                written = np.zeros((6, 128), np.bool_)
                written[3, : min(count, 128)] = True
                if count > 128:
                    written[1, : count - 128] = True
                np.testing.assert_array_equal(valid, written)
                np.testing.assert_array_equal(payload[written], 0x7E)
                np.testing.assert_array_equal(scales[written], 1)
                np.testing.assert_array_equal(payload[~written], 0x55)
                np.testing.assert_array_equal(scales[~written], -123)
                with self.assertRaises(ValueError):
                    pages.read([2, 0], count, 1)

    def test_partial_update_preserves_full_pages_and_separate_scales(self):
        pages = KVPages.allocate(4)
        pages.update([2, 0], 0, constant_rows(129))
        magnitudes = np.array([[224, 112, 56, 28], [896, 1792, 3584, 7168]], np.float32)
        rows = bf16_bits(np.broadcast_to(magnitudes[None, ..., None], (3, 2, 4, 256)))
        before_payload, before_scales = pages.payload.copy(), pages.scales.copy()
        pages.update([2, 0], 126, rows)
        expected = np.full((129, 2, 4, 256), 448.0)
        expected[126:] = magnitudes[..., None]
        np.testing.assert_array_equal(pages.read([2, 0], 0, 129), expected)
        expected_scales = [[0.5, 0.25, 0.125, 0.0625], [2, 4, 8, 16]]
        np.testing.assert_array_equal(pages.scales[2, 126], expected_scales)
        np.testing.assert_array_equal(pages.scales[0, 0], expected_scales)
        touched = np.zeros((4, 128), np.bool_)
        touched[2, 126:], touched[0, 0] = True, True
        np.testing.assert_array_equal(pages.payload[~touched], before_payload[~touched])
        np.testing.assert_array_equal(pages.scales[~touched], before_scales[~touched])

    def test_private_page_copy_update_and_discard_preserve_shared_tail(self):
        pages = KVPages.allocate(3)
        pages.update([0], 0, constant_rows(127))
        source_payload, source_scales = pages.payload[0].copy(), pages.scales[0].copy()
        pages.copy_page(0, 1)
        pages.update([1], 127, constant_rows(1, 896))
        np.testing.assert_array_equal(pages.read([1], 127, 1), 896)
        np.testing.assert_array_equal(pages.scales[1, 127], 2)
        pages.discard_page(1)
        with self.assertRaises(ValueError):
            pages.read([1], 0, 1)
        np.testing.assert_array_equal(pages.payload[0], source_payload)
        np.testing.assert_array_equal(pages.scales[0], source_scales)
        np.testing.assert_array_equal(pages.read([0], 0, 127), 448)
        with self.assertRaises(ValueError):
            pages.read([0], 127, 1)

    def test_rejected_update_is_atomic_and_zero_valid_rows_do_nothing(self):
        pages = KVPages.allocate(3)
        pages.update([1], 0, constant_rows(1))
        before = [pages.payload.copy(), pages.scales.copy(), pages.valid.copy()]
        bad_rows = constant_rows(2)
        bad_rows[1, 1, 3, 255] = 0x7F80
        cases = [
            ([1], 0, bad_rows, 2),
            ([1], 127, constant_rows(2), 2),
            ([1, 1], 0, constant_rows(1), 1),
            ([-1], 0, constant_rows(1), 1),
            ([3], 0, constant_rows(1), 1),
            ([1], -1, constant_rows(1), 1),
            ([1], 0, constant_rows(1), 2),
            ([1], 0, constant_rows(1), True),
        ]
        for table, start, rows, count in cases:
            with self.subTest(table=table, start=start, count=count):
                with self.assertRaises(ValueError):
                    pages.update(table, start, rows, count)
                for actual, expected in zip(
                    (pages.payload, pages.scales, pages.valid), before, strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
        pages.update([1], 128, np.full((1, 2, 4, 256), 0x7FC0, np.uint16), valid_rows=0)
        for actual, expected in zip(
            (pages.payload, pages.scales, pages.valid), before, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
