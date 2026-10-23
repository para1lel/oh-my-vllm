import unittest

import numpy as np

from development.model.deepseek_v41.quantization import decode_tensor


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.decode = decode_tensor

    def test_unscaled_encodings_and_signed_zero(self):
        cases = (
            (
                "BF16",
                np.array(
                    [0, 0x8000, 0x3F80, 0xBFC0, 1, 0x007F, 0x0080, 0x7F7F],
                    dtype=np.uint16,
                ),
                [
                    0.0,
                    -0.0,
                    1.0,
                    -1.5,
                    2.0**-133,
                    127 * 2.0**-133,
                    2.0**-126,
                    255 * 2.0**120,
                ],
            ),
            (
                "F32",
                np.array([0.0, -0.0, 1.5, -2.0, 2.0**-149], dtype=np.float32),
                [0.0, -0.0, 1.5, -2.0, 2.0**-149],
            ),
            (
                "F8_E4M3",
                np.array(
                    [0, 0x80, 1, 0x81, 7, 8, 0x38, 0xBC, 0x78, 0x7E, 0xFE],
                    dtype=np.uint8,
                ),
                [
                    0.0,
                    -0.0,
                    2.0**-9,
                    -(2.0**-9),
                    7 * 2.0**-9,
                    2.0**-6,
                    1.0,
                    -1.5,
                    256.0,
                    448.0,
                    -448.0,
                ],
            ),
            (
                "F8_E8M0",
                np.array([0, 1, 126, 127, 128, 254], dtype=np.uint8),
                [2.0**-127, 2.0**-126, 0.5, 1.0, 2.0, 2.0**127],
            ),
        )
        for dtype, raw, expected in cases:
            with self.subTest(dtype=dtype):
                raw.setflags(write=False)
                result = self.decode(raw, dtype)
                self.assertEqual(result.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
                self.assertFalse(np.shares_memory(raw, result))
                np.testing.assert_array_equal(self.decode(raw, dtype), result)
        scalar = self.decode(np.array(0x3F80, dtype=">u2"), "BF16")
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.item(), 1.0)
        np.testing.assert_array_equal(
            self.decode(np.array([1.5, -0.0], dtype=">f4"), "F32"), [1.5, -0.0]
        )

    def test_dense_and_engram_scale_placement(self):
        dense_raw = np.full((35, 33), 0x38, dtype=np.uint8).T
        dense_raw[0, 0] = 0x80
        dense_scales = np.array([[127, 128], [126, 129]], dtype=np.uint8)
        expected = np.ones((33, 35), dtype=np.float64)
        expected[0, 0] = -0.0
        expected[:32, 32:] = 2.0
        expected[32:, :32] = 0.5
        expected[32:, 32:] = 4.0
        row_raw = np.full((2, 64), 0x38, dtype=np.uint8)
        row_raw[1, 0] = 0x80
        row_scales = np.array([[127, 128], [129, 126]], dtype=np.uint8)
        expected_rows = np.array(
            [[1.0] * 32 + [2.0] * 32, [-0.0] + [4.0] * 31 + [0.5] * 32]
        )
        for raw, scales, mode, reference in (
            (dense_raw, dense_scales, "dense32", expected),
            (row_raw, row_scales, "row32", expected_rows),
            (
                np.array([[0x7E] * 32 + [1] * 32], dtype=np.uint8),
                np.array([[254, 0]], dtype=np.uint8),
                "row32",
                np.array([[448 * 2.0**127] * 32 + [2.0**-136] * 32]),
            ),
        ):
            with self.subTest(layout=mode):
                original = raw.copy()
                scale_copy = scales.copy()
                raw.setflags(write=False)
                scales.setflags(write=False)
                result = self.decode(raw, "F8_E4M3", scales=scales, scale_layout=mode)
                np.testing.assert_array_equal(result, reference)
                np.testing.assert_array_equal(np.signbit(result), np.signbit(reference))
                np.testing.assert_array_equal(raw, original)
                np.testing.assert_array_equal(scales, scale_copy)

    def test_packed_expert_low_nibble_first_and_per_row_scales(self):
        raw = np.tile(
            np.array(
                [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8
            ).view(np.int8),
            (2, 4),
        )
        group = [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ] * 2
        reference = np.array([group + group, group + group])
        reference[0, 32:] *= 2
        reference[1, :32] *= 4
        reference[1, 32:] *= 0.5
        scales = np.array([[127, 128], [129, 126]], dtype=np.uint8)
        raw.setflags(write=False)
        result = self.decode(raw, "I8", scales=scales, scale_layout="row32")
        self.assertEqual(result.shape, (2, 64))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(np.signbit(result), np.signbit(reference))
        np.testing.assert_array_equal(raw[0, :8], [16, 50, 84, 118, -104, -70, -36, -2])

    def test_activation_fp8_bf16_boundary_ties_and_group_scales(self):
        from .reference_quantization import activation_fp8

        values = np.zeros((2, 64), dtype=np.float32)
        values[0, :10] = [
            0.0,
            -0.0,
            1.0625,
            1.1875,
            -1.0625,
            -1.1875,
            1.064453125,
            2**-10,
            3 * 2**-10,
            -(2**-10),
        ]
        values[0, 31] = 448
        values[0, 32:] = values[0, :32] * 2
        original = values.copy()
        values.setflags(write=False)
        result = activation_fp8(values)
        expected = np.zeros((2, 64), dtype=np.uint8)
        expected[0, :10] = [0, 0x80, 0x38, 0x3A, 0xB8, 0xBA, 0x38, 0, 2, 0x80]
        expected[0, 31] = 0x7E
        expected[0, 32:] = expected[0, :32]
        np.testing.assert_array_equal(result.encoded, expected)
        np.testing.assert_array_equal(result.scales, [[127, 128], [105, 105]])
        reference = np.zeros((2, 64), dtype=np.float32)
        reference[0, :10] = [0, -0.0, 1, 1.25, -1, -1.25, 1, 0, 2**-8, -0.0]
        reference[0, 31] = 448
        reference[0, 32:] = reference[0, :32] * 2
        self.assertEqual(result.dequantized.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            result.dequantized.view(np.uint32), reference.view(np.uint32)
        )
        np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))

    def test_index_and_compressed_fp4_scale_rounding_and_packing(self):
        from .reference_quantization import compressed_kv_fp4, indexer_fp4

        values = np.zeros((2, 64), dtype=np.float32)
        values[0, :16] = [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            -0.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5,
            0,
        ]
        values[0, 31] = 6
        values[0, 32:] = values[0, :32] * 2
        result = indexer_fp4(values)
        expected = np.zeros((2, 32), dtype=np.uint8)
        expected[0, :8] = [0x20, 0x42, 0x64, 0x86, 0xA8, 0xCA, 0xEC, 0x0E]
        expected[0, 15] = 0x70
        expected[0, 16:] = expected[0, :16]
        np.testing.assert_array_equal(result.encoded, expected)
        np.testing.assert_array_equal(result.scales, [[127, 128], [1, 1]])
        reference = np.zeros((2, 64), dtype=np.float32)
        reference[0, :16] = [0, 1, 1, 2, 2, 4, 4, -0.0, -0.0, -1, -1, -2, -2, -4, -4, 0]
        reference[0, 31] = 6
        reference[0, 32:] = reference[0, :32] * 2
        np.testing.assert_array_equal(
            result.dequantized.view(np.uint32), reference.view(np.uint32)
        )

        cache = np.zeros((3, 32), dtype=np.float32)
        cache[0, :3] = [6.375, 1.25, -0.0]
        cache[0, 16:19] = [7.125, 4.375, -0.0]
        cache[1, 0] = 3000
        result = compressed_kv_fp4(cache)
        expected = np.zeros((3, 16), dtype=np.uint8)
        expected[0, :2] = [0x27, 0x08]
        expected[0, 8:10] = [0x67, 0x08]
        expected[1, 0] = 7
        np.testing.assert_array_equal(result.encoded, expected)
        np.testing.assert_array_equal(result.scales, [[0x38, 0x3A], [0x7E, 1], [1, 1]])
        reference = np.zeros((3, 32), dtype=np.float32)
        reference[0, :3] = [6, 1, -0.0]
        reference[0, 16:19] = [7.5, 5, -0.0]
        reference[1, 0] = 2688
        np.testing.assert_array_equal(
            result.dequantized.view(np.uint32), reference.view(np.uint32)
        )

    def test_source_quantization_rejects_invalid_input_and_bf16_overflow(self):
        from .reference_quantization import (
            activation_fp8,
            compressed_kv_fp4,
            indexer_fp4,
        )

        for operation, invalid in (
            (activation_fp8, [0.0] * 32),
            (activation_fp8, np.zeros((1, 32), dtype=np.float64)),
            (indexer_fp4, np.zeros((1, 32), dtype=np.int32)),
            (activation_fp8, np.zeros((), dtype=np.float32)),
            (indexer_fp4, np.zeros((0, 32), dtype=np.float32)),
            (activation_fp8, np.zeros((1, 31), dtype=np.float32)),
            (indexer_fp4, np.zeros((1, 16), dtype=np.float32)),
            (compressed_kv_fp4, np.zeros((1, 17), dtype=np.float32)),
            (activation_fp8, np.full((1, 32), np.nan, dtype=np.float32)),
            (indexer_fp4, np.full((1, 32), np.inf, dtype=np.float32)),
            (compressed_kv_fp4, np.full((1, 16), -np.inf, dtype=np.float32)),
            (
                activation_fp8,
                np.full((1, 32), np.finfo(np.float32).max, dtype=np.float32),
            ),
            (indexer_fp4, np.full((1, 32), 255 * 2.0**120, dtype=np.float32)),
        ):
            with (
                self.subTest(operation=operation.__name__, shape=np.shape(invalid)),
                self.assertRaises(ValueError),
            ):
                operation(invalid)

    def test_rejects_nonfinite_encodings_and_invalid_scale_contracts(self):
        cases = []
        for bits in (0x7F80, 0xFF80, 0x7FC0, 0xFFFF):
            cases.append((np.array([bits], dtype=np.uint16), "BF16", {}))
        for value in (np.inf, -np.inf, np.nan):
            cases.append((np.array([value], dtype=np.float32), "F32", {}))
        for bits in (0x7F, 0xFF):
            cases.append((np.array([bits], dtype=np.uint8), "F8_E4M3", {}))
        cases.append((np.array([255], dtype=np.uint8), "F8_E8M0", {}))
        raw = np.full((2, 32), 0x38, dtype=np.uint8)
        scales = np.full((2, 1), 127, dtype=np.uint8)
        cases.extend(
            (
                (raw, "unknown", {}),
                (raw, "F8_E4M3", {"scales": scales}),
                (raw, "F8_E4M3", {"scale_layout": "row32"}),
                (raw, "F8_E4M3", {"scales": scales, "scale_layout": "unknown"}),
                (raw, "F8_E4M3", {"scales": scales, "scale_layout": "dense32"}),
                (raw[:, :31], "F8_E4M3", {"scales": scales, "scale_layout": "row32"}),
                (raw[0], "F8_E4M3", {"scales": scales, "scale_layout": "row32"}),
                (raw[:0], "F8_E4M3", {"scales": scales[:0], "scale_layout": "row32"}),
                (raw, "F8_E4M3", {"scales": scales.ravel(), "scale_layout": "row32"}),
                (
                    raw,
                    "F8_E4M3",
                    {
                        "scales": np.array([[127], [255]], dtype=np.uint8),
                        "scale_layout": "row32",
                    },
                ),
                (
                    raw,
                    "F8_E4M3",
                    {"scales": scales.astype(np.float64), "scale_layout": "row32"},
                ),
                (raw.astype(np.int8), "I8", {}),
                (
                    raw.astype(np.int8),
                    "I8",
                    {
                        "scales": np.array([[127, 127]], dtype=np.uint8),
                        "scale_layout": "dense32",
                    },
                ),
                (
                    raw[:, :15].astype(np.int8),
                    "I8",
                    {"scales": scales, "scale_layout": "row32"},
                ),
                (
                    raw.astype(np.uint16),
                    "BF16",
                    {"scales": scales, "scale_layout": "row32"},
                ),
                (
                    raw.astype(np.float32),
                    "F32",
                    {"scales": scales, "scale_layout": "row32"},
                ),
                (raw, "F8_E8M0", {"scales": scales, "scale_layout": "row32"}),
                (raw.astype(np.float64), "F32", {}),
                (raw.astype(np.uint32), "BF16", {}),
                (raw.astype(np.int8), "F8_E4M3", {}),
                (raw, "I8", {"scales": scales, "scale_layout": "row32"}),
                ([0x38], "F8_E4M3", {}),
            )
        )
        for raw, dtype, kwargs in cases:
            with (
                self.subTest(dtype=dtype, kwargs=kwargs),
                self.assertRaises(ValueError),
            ):
                self.decode(raw, dtype, **kwargs)


if __name__ == "__main__":
    unittest.main()
