import math
import unittest
from dataclasses import replace

from oh_my_vllm.kernels.specification import MODEL, OPERATORS

from .roofline import ExpectedCase, Observation, Work, expected_efficiencies
from .work import work_for
from .workload import PROFILES, SERVING_OPERATORS, modeled_cases, stage_cases


class Contracts(unittest.TestCase):
    def test_serving_distribution_covers_operator_paths(self):
        cases = modeled_cases()
        self.assertAlmostEqual(sum(profile.probability for profile in PROFILES), 1)
        self.assertEqual(
            {case.shape.operator for case in cases},
            set(SERVING_OPERATORS),
        )
        stages = stage_cases()
        self.assertEqual(
            {c.shape.operator for c in stages}, {o.name for o in OPERATORS}
        )
        self.assertTrue(all(case.calls > 0 for case in cases))
        self.assertTrue(
            all(work_for(case.shape).lower_bound_seconds() > 0 for case in cases)
        )
        self.assertTrue(all(case.shape.rows <= MODEL["step_tokens"] for case in cases))
        self.assertTrue(
            any(
                case.shape.operator == "mtp_context" and case.shape.rows > 784
                for case in cases
            )
        )
        self.assertTrue(any(case.shape.variant == "kv_only" for case in stages))
        self.assertTrue(any(case.shape.variant == "query_only" for case in cases))
        self.assertTrue(any(case.shape.variant == "embedding_media" for case in cases))
        self.assertTrue(
            any(
                case.shape.operator == "sample" and case.shape.rows == 20
                for case in cases
            )
        )

    def test_expected_efficiency_uses_fixed_weights_and_complete_coverage(self):
        work = Work(7700)
        lower = work.lower_bound_seconds()
        expected = {"a": ExpectedCase("op", 1, work), "b": ExpectedCase("op", 3, work)}
        observations = [
            Observation("op", "a", 1, lower, lower * 2, True, "source"),
            Observation("op", "b", 3, lower, lower * 5, True, "source"),
        ]
        efficiency = expected_efficiencies(observations, expected, "source")["op"]
        self.assertAlmostEqual(efficiency, 0.275)
        for invalid in (
            observations[:1],
            [*observations, observations[0]],
            [replace(observations[0], calls=0.01), observations[1]],
            [replace(observations[0], operator="other"), observations[1]],
            [replace(observations[0], lower_bound_seconds=lower * 2), observations[1]],
            [replace(observations[0], source_id="other"), observations[1]],
            [replace(observations[0], correct=False), observations[1]],
            [replace(observations[0], measured_seconds=math.nan), observations[1]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_efficiencies(invalid, expected, "source")
        with self.assertRaises(ValueError):
            expected_efficiencies([], {}, "source")

    def test_tensor_core_work_uses_one_shared_resource(self):
        combined = Work(0, fp8_flops=int(4.5e15), bf16_flops=int(2.25e15))
        self.assertEqual(combined.lower_bound_seconds(), 2)
        with self.assertRaises(ValueError):
            Work(-1).lower_bound_seconds()
        with self.assertRaises(ValueError):
            Work(math.nan).lower_bound_seconds()


if __name__ == "__main__":
    unittest.main()
