import unittest

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
