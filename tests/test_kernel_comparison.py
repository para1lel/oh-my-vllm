"""CPU checks for the CUDA migration's formal paired performance decision."""

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "kernel_comparison",
    Path(__file__).resolve().parents[1] / "development/kernels/comparison.py",
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class ComparisonTest(unittest.TestCase):
    def test_small_consistent_gain_is_accepted_without_percentage_floor(self):
        result = comparison.compare([[(1, 0.999)] * 20] * 3, bootstrap_samples=1000)
        self.assertTrue(result["passed"])

    def test_tie_and_round_regression_fail(self):
        for rounds in (
            [[(1, 1)] * 20] * 3,
            [[(1, 0.8)] * 20, [(1, 0.8)] * 20, [(1, 1.01)] * 20],
        ):
            self.assertFalse(
                comparison.compare(rounds, bootstrap_samples=1000)["passed"]
            )

    def test_noisy_positive_median_is_not_sufficient(self):
        pairs = [(1, 0.99)] * 11 + [(1, 1.2)] * 9
        self.assertFalse(
            comparison.compare([pairs] * 3, bootstrap_samples=1000)["passed"]
        )

    def test_incomplete_or_invalid_measurements_are_rejected(self):
        for rounds in ([[(1, 0.9)] * 20] * 2, [[(1, float("nan"))] * 20] * 3):
            with self.assertRaises(ValueError):
                comparison.compare(rounds)
