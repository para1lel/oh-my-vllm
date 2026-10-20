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
        for case_id, error, passed in (
            ("short", 0.6, False),
            ("mtp-restored", 0.6, True),
            ("image-video", 0.6, True),
            ("context-32769", 0.6, True),
            ("mtp-restored", 1.2, True),
            ("image-video", 1.2, True),
            ("image-video", 1.4, False),
            ("context-32769", 1.0, True),
            ("context-32769", 1.2, False),
        ):
            changed = reference.copy()
            changed[0] += error
            with self.subTest(case_id=case_id, error=error):
                self.assertEqual(
                    difference(changed, reference, case_id=case_id)["passed"], passed
                )
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

        from .cases import load_cases
        from .compare import compare_endpoints

        case = next(case for case in load_cases() if case.id == "mtp-restored")
        expected = {
            name: np.broadcast_to(reference, shape).copy()
            for name, shape in case.output_shapes.items()
        }
        actual = {name: value.copy() for name, value in expected.items()}
        actual["verified"][0, 0] = 100
        metrics = compare_endpoints(case, actual, expected)
        self.assertEqual(len(metrics["verified"]), 5)
        self.assertFalse(metrics["verified"][0]["passed"])
        self.assertTrue(metrics["verified"][4]["passed"])
        for bad in (
            {**actual, "verified": actual["verified"][-1]},
            {name: value for name, value in actual.items() if name != "mtp_next"},
        ):
            with self.assertRaises(ValueError):
                compare_endpoints(case, bad, expected)
