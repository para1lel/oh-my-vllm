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
