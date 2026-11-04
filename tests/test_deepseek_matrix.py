import unittest

from development.model.check_deepseek_batch import (
    argument_parser,
    require_multiround_gates,
    target_signature,
    update_boundaries,
)
from development.serving.deepseek_matrix import AGENTX_CASES, HTTP_CASES


class DeepSeekMatrix(unittest.TestCase):
    def test_http_matrix_matches_the_approved_38_case_scope(self):
        self.assertEqual(len(HTTP_CASES), 38)
        dspark = [case for case in HTTP_CASES if case["mode"] == "dspark5"]
        ordinary = [case for case in HTTP_CASES if case["mode"] == "ordinary"]
        self.assertEqual(len(dspark), 36)
        self.assertEqual(len(ordinary), 2)
        self.assertEqual({case["strategy"] for case in dspark}, {"tp4", "tep4", "dep4"})
        self.assertEqual(
            {
                (
                    case["batch"],
                    case["input_tokens"],
                    case["output_tokens"],
                )
                for case in dspark
            },
            {
                (batch, input_tokens, output_tokens)
                for batch in (4, 16, 64)
                for input_tokens in (400000, 800000)
                for output_tokens in (64, 8192)
            },
        )
        self.assertEqual(
            ordinary,
            [
                {
                    "mode": "ordinary",
                    "strategy": "tp4",
                    "batch": 4,
                    "input_tokens": 400000,
                    "output_tokens": 64,
                },
                {
                    "mode": "ordinary",
                    "strategy": "tp4",
                    "batch": 4,
                    "input_tokens": 400000,
                    "output_tokens": 8192,
                },
            ],
        )

    def test_batch_checker_requires_verify_continuation_by_default(self):
        parser = argument_parser()
        values = ["--model", "/unused-model", "--output", "/unused-output"]
        self.assertTrue(parser.parse_args(values).require_verify_continuation)
        self.assertFalse(
            parser.parse_args(
                [*values, "--no-require-verify-continuation"]
            ).require_verify_continuation
        )
        with self.assertRaisesRegex(AssertionError, "preceding proposal"):
            require_multiround_gates(
                "dspark", "eager", continued=0, max_batch=4, crossings={}
            )
        require_multiround_gates(
            "dspark",
            "eager",
            continued=0,
            max_batch=4,
            crossings={},
            require_verify_continuation=False,
        )

    def test_batch_checker_rejects_candidate_runs_with_only_singletons(self):
        for execution in ("eager", "graph", "graph_fallback"):
            with (
                self.subTest(execution=execution),
                self.assertRaisesRegex(AssertionError, "multi-request"),
            ):
                require_multiround_gates(
                    "dspark_aligned", execution, continued=1, max_batch=1, crossings={}
                )
            require_multiround_gates(
                "dspark_aligned", execution, continued=1, max_batch=2, crossings={}
            )

    def test_batch_checker_boundaries_require_actual_calls_on_both_sides(self):
        crossings = dict(bucket=False, window=False, **{"compress/4": False})
        observed = {}
        buckets = {128: 128, 129: 256}

        def call(request, position):
            return dict(
                requests=[request],
                batch_size=1,
                signatures=[
                    target_signature(position, 1, 4, 128, (4,), buckets.__getitem__)
                ],
            )

        update_boundaries([call(1, 127)], observed, crossings)
        self.assertFalse(any(crossings.values()))
        update_boundaries([call(2, 128)], observed, crossings)
        self.assertFalse(any(crossings.values()))
        with self.assertRaisesRegex(AssertionError, "actual target boundary"):
            require_multiround_gates(
                "ordinary",
                "graph",
                continued=0,
                max_batch=4,
                crossings=crossings,
                require_boundaries=True,
            )
        update_boundaries([call(1, 128)], observed, crossings)
        self.assertTrue(all(crossings.values()))
        require_multiround_gates(
            "ordinary",
            "graph",
            continued=0,
            max_batch=4,
            crossings=crossings,
            require_boundaries=True,
        )

    def test_agentx_matrix_matches_the_approved_strategy_scope(self):
        self.assertEqual(
            AGENTX_CASES,
            (
                {"concurrency": 4, "strategy": "tp4"},
                {"concurrency": 16, "strategy": "tep4"},
                {"concurrency": 64, "strategy": "dep4"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
