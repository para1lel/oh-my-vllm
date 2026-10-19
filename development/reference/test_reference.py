import unittest

import numpy as np

from .reference import (
    bf16,
    fp8_decode,
    fp8_encode,
    gdn,
    numerical_error,
    quantize,
    residual_rms,
)
from .sample import acceptance_reference


class Reference(unittest.TestCase):
    def test_mtp_short_tail_selects_its_final_target_and_state(self):
        target = np.zeros((4, 5, 8), np.float32)
        proposal = np.zeros((4, 4, 8), np.float32)
        drafts = np.full((4, 4), -1, np.int64)
        lengths = np.array([0, 1, 3, 4])
        for request, length in enumerate(lengths):
            for index in range(5):
                target[request, index, index] = 1
            for index in range(length):
                drafts[request, index] = index
                proposal[request, index, index] = 1
        rng = np.zeros((4, 2), np.uint64)
        slots = np.arange(20, dtype=np.int32).reshape(4, 5)
        kept, counts, selected, counters = acceptance_reference(
            target, proposal, drafts, rng, slots
        )
        np.testing.assert_array_equal(counts, lengths + 1)
        np.testing.assert_array_equal(selected, slots[np.arange(4), lengths])
        np.testing.assert_array_equal(counters[:, 1], 2 * (lengths + 1))
        for request, length in enumerate(lengths):
            np.testing.assert_array_equal(
                kept[request, : length + 1], range(length + 1)
            )
            np.testing.assert_array_equal(kept[request, length + 1 :], -1)

    def test_low_precision_rounding_and_invalid_measurements(self):
        np.testing.assert_array_equal(bf16([1 + 2**-8, 1 + 3 * 2**-8]), [1, 1 + 2**-6])
        bits = np.arange(256, dtype=np.uint8)
        finite = (bits & 127) != 127
        np.testing.assert_array_equal(
            fp8_encode(fp8_decode(bits[finite])), bits[finite]
        )
        np.testing.assert_array_equal(fp8_encode([1.0625, 1.1875, 1000]), [56, 58, 126])
        values = np.zeros((1, 128))
        values[0, :2] = [2.15625, 0.404296875]
        self.assertEqual(quantize(values)[0][0, 1], 106)
        self.assertFalse(numerical_error([0.04], [0])["passed"])
        self.assertGreater(numerical_error([0.04], [0])["max_tolerance_ratio"], 1)
        np.testing.assert_array_equal(
            residual_rms([1, 1], [2**-8, 2**-7], np.zeros(2)), [1, 1]
        )
        self.assertFalse(numerical_error([0.001], [0], state=True)["passed"])
        with self.assertRaises(ValueError):
            numerical_error([np.nan], [0])

    def test_gdn_uses_unrounded_carry_and_private_snapshots(self):
        query = np.array([[[1.0, 0]], [[1.0, 0]]])
        value = np.array([[[0.333, 0.222]], [[0.444, 0.111]]])
        state = np.zeros((1, 2, 2))
        output, snapshots = gdn(
            query, query, value, np.zeros((2, 1)), np.full((2, 1), 0.5), state
        )
        normalized = 1 / np.sqrt(1 + 1e-6)
        first = value[0, 0] * normalized * 0.5
        second = first + (value[1, 0] - first * normalized) * normalized * 0.5
        np.testing.assert_array_equal(snapshots[1, 0, :, 0], bf16(second))
        np.testing.assert_array_equal(
            output[1, 0], bf16(second * normalized / np.sqrt(2))
        )
        np.testing.assert_array_equal(state, 0)
        self.assertFalse(np.shares_memory(snapshots, state))


if __name__ == "__main__":
    unittest.main()
