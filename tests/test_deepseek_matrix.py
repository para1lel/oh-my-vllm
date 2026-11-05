import unittest
from types import SimpleNamespace

from development.model.check_deepseek_batch import (
    TargetCapture,
    argument_parser,
    execute_single_round,
    require_multiround_gates,
    require_scheduled_requests,
    require_single_execution,
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

    def test_batch_decode_flag_preserves_default_and_selects_real_scheduling(self):
        parser = argument_parser()
        values = ["--model", "/unused-model", "--output", "/unused-output"]
        self.assertTrue(parser.parse_args(values).batch_decode)
        self.assertFalse(parser.parse_args([*values, "--no-batch-decode"]).batch_decode)
        rows = [dict(request_id=key) for key in range(1, 5)]
        calls = []

        def execute(message):
            calls.append(message)
            return dict(
                outputs=[
                    dict(request_id=row["request_id"]) for row in message["scheduled"]
                ]
            )

        worker = SimpleNamespace(execute=execute)
        outputs = execute_single_round(
            worker, rows, reference=False, batch_decode=False
        )
        self.assertEqual(set(outputs), {1, 2, 3, 4})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["scheduled"], rows)
        worker.execute = lambda _message: dict(outputs=[*rows, rows[0]])
        with self.assertRaisesRegex(AssertionError, "request count"):
            execute_single_round(worker, rows, reference=False, batch_decode=False)

    def test_batch_off_requires_all_scheduled_outputs_and_target_calls(self):
        rows = [dict(request_id=key) for key in range(1, 5)]
        targets = {key: [{}] for key in range(1, 5)}
        require_scheduled_requests(rows, rows, targets)
        for count in (0, 2):
            with (
                self.subTest(target_count=count),
                self.assertRaisesRegex(AssertionError, "scheduled requests"),
            ):
                require_scheduled_requests(
                    rows, rows, {key: [{}] * count for key in range(1, 5)}
                )
        for outputs, observed in (
            (rows[:-1], targets),
            ([*rows, rows[0]], targets),
            (rows, {1: []}),
        ):
            with (
                self.subTest(outputs=outputs, targets=observed),
                self.assertRaisesRegex(AssertionError, "scheduled requests"),
            ):
                require_scheduled_requests(rows, outputs, observed)
        require_multiround_gates(
            "dspark",
            "eager",
            continued=1,
            max_batch=1,
            crossings={},
            require_batch=False,
            scheduled_requests=64,
            required_requests=64,
        )
        with self.assertRaisesRegex(AssertionError, "full request batch"):
            require_multiround_gates(
                "dspark",
                "eager",
                continued=1,
                max_batch=1,
                crossings={},
                require_batch=False,
                scheduled_requests=4,
                required_requests=64,
            )
        with self.assertRaisesRegex(AssertionError, "preceding proposal"):
            require_multiround_gates(
                "dspark",
                "eager",
                continued=0,
                max_batch=1,
                crossings={},
                require_batch=False,
                scheduled_requests=64,
                required_requests=64,
            )

    def test_single_batch_off_rejects_eager_calls_requested_as_graph(self):
        import torch
        from oh_my_vllm.models.deepseek_v41.inference import Worker

        class CpuModel:
            layers = ()

            def __call__(self, _inputs, **_options):
                logits = torch.tensor([[0.0, 1.0]])
                return logits.argmax(-1), logits, None

        worker = Worker.__new__(Worker)
        worker.device, worker.prefill_rows, worker.dspark = (
            torch.device("cpu"),
            16,
            False,
        )
        worker.model, worker.decode_graphs = CpuModel(), {}
        worker.args, worker.decode_cycle = SimpleNamespace(window_size=128), 2
        worker.decode_graph_enabled, worker.decode_graph_limit = True, 0
        worker._input_tokens = lambda values: torch.tensor(values, dtype=torch.long)
        worker._decode_graph_step = lambda *_args, **_options: None
        worker.requests = {
            key: SimpleNamespace(cache_slot=key - 1) for key in range(1, 5)
        }
        rows = [dict(request_id=key, token_ids=[key]) for key in range(1, 5)]

        def execute(message):
            for row in message["scheduled"]:
                worker._target(row["token_ids"], 1, row["request_id"] - 1)
            return dict(outputs=message["scheduled"])

        worker.execute = execute
        batch = SimpleNamespace(
            target_decode_batch=None, context_token_bucket=lambda _end: 128
        )
        with TargetCapture(worker, torch, batch) as captured:
            outputs = execute_single_round(
                worker, rows, reference=False, batch_decode=False
            )
        require_scheduled_requests(rows, list(outputs.values()), captured.targets)
        require_single_execution("eager", captured.calls)
        with self.assertRaisesRegex(AssertionError, "requested graph"):
            require_single_execution("graph", captured.calls)
        with self.assertRaisesRegex(AssertionError, "captured calls"):
            require_single_execution("eager", [])

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
