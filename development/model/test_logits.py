import unittest

import numpy as np

from .compare import difference, validate_case


class LogitsReport(unittest.TestCase):
    def test_full_vocabulary_and_invalid_evidence(self):
        reference = np.linspace(-8, 8, 248320, dtype=np.float32)
        result = difference(reference, reference)
        self.assertEqual(result["nrmse"], 0)
        self.assertEqual(result["max_abs"], 0)
        self.assertAlmostEqual(result["cosine"], 1)
        self.assertTrue(result["top1_equal"])
        self.assertTrue(result["passed"])
        changed = reference.copy()
        changed[0] = 100
        result = difference(changed, reference)
        self.assertEqual(result["max_abs"], 108)
        self.assertFalse(result["top1_equal"])
        self.assertFalse(result["passed"])
        for bad in (reference[:-1], reference.reshape(1, -1), reference * np.nan):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                difference(bad, reference)
        case = {"tokens": list(range(16)), "decode_token": 42, "iterations": 1}
        validate_case(case, case, {"config": "a"}, {"config": "a"})
        for candidate, checkpoint in (
            ({**case, "decode_token": 43}, {"config": "a"}),
            ({**case, "iterations": 2}, {"config": "a"}),
            (case, {"config": "b"}),
        ):
            with self.subTest(case=candidate), self.assertRaises(ValueError):
                validate_case(candidate, case, checkpoint, {"config": "a"})
