import copy
import json
import unittest

from . import agentx, agentx_data


def raw_record(response_id, phase="profiling"):
    chunk = {
        "id": response_id,
        "choices": [{"finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 75},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return {
        "metadata": {
            "x_request_id": response_id,
            "benchmark_phase": phase,
            "request_start_ns": 101_000_000_000,
            "request_end_ns": 104_000_000_000,
        },
        "status": 200,
        "start_perf_ns": 10_000_000_000,
        "payload": {"max_tokens": 20, "ignore_eos": True},
        "responses": [
            {
                "perf_ns": 13_000_000_000,
                "packets": [{"name": "data", "value": json.dumps(chunk)}],
            },
            {
                "perf_ns": 13_000_000_000,
                "packets": [{"name": "data", "value": "[DONE]"}],
            },
        ],
    }


def server_metric(response_id):
    return {
        "response_id": response_id,
        "status": "completed",
        "processing_started_unix_s": 102.0,
        "processing_seconds": 2.0,
        "output_tokens": 20,
        "prefix_hit_tokens": 75,
        "queue_seconds": 1.0,
    }


class AgentXTests(unittest.TestCase):
    def test_phase_and_full_response_identity_control_selection(self):
        records = [
            raw_record("chatcmpl_old_1", "warmup"),
            raw_record("chatcmpl_new_1"),
        ]
        overflow = raw_record("overflow")
        overflow["metadata"]["context_overflow_skip"] = True
        overflow.update(status=400, error={"message": "context too long"})
        records.append(overflow)
        metrics = {
            name: server_metric(name) for name in ("chatcmpl_old_1", "chatcmpl_new_1")
        }
        metrics["chatcmpl_old_1"]["processing_seconds"] = 200.0
        rows, counts = agentx_data.collect_rows(records, metrics, 100.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["response_id"], "chatcmpl_new_1")
        self.assertEqual(rows[0]["processing_start_s"], 2.0)
        self.assertEqual(rows[0]["output_tokens_per_second"], 10.0)
        self.assertEqual(rows[0]["cache_hit_ratio"], 0.75)
        self.assertEqual(rows[0]["client_seconds"], 3.0)
        self.assertEqual(counts["warmup"], 1)
        self.assertEqual(counts["skipped_context_overflow"], 1)
        self.assertEqual(counts["failed"], 0)

    def test_missing_usage_identity_and_incomplete_stream_are_rejected(self):
        base = raw_record("chatcmpl_1")
        metric = server_metric("chatcmpl_1")
        cases = []
        missing = copy.deepcopy(base)
        chunk = json.loads(missing["responses"][0]["packets"][0]["value"])
        del chunk["usage"]["prompt_tokens_details"]
        missing["responses"][0]["packets"][0]["value"] = json.dumps(chunk)
        cases.append(([missing], {"chatcmpl_1": metric}))
        for field, value in (
            ("total_tokens", None),
            ("total_tokens", -1),
            ("total_tokens", True),
            ("total_tokens", 999),
            ("completion_tokens_details", {"reasoning_tokens": -10}),
            ("completion_tokens_details", {"reasoning_tokens": 1}),
        ):
            invalid = copy.deepcopy(base)
            chunk = json.loads(invalid["responses"][0]["packets"][0]["value"])
            chunk["usage"][field] = value
            invalid["responses"][0]["packets"][0]["value"] = json.dumps(chunk)
            cases.append(([invalid], {"chatcmpl_1": metric}))
        cases.append(([base], {"chatcmpl_other_1": metric}))
        cases.append(([base, base], {"chatcmpl_1": metric}))
        bad_time = metric | {"processing_seconds": float("nan")}
        cases.append(([base], {"chatcmpl_1": bad_time}))
        for records, metrics in cases:
            with (
                self.subTest(records=records, metrics=metrics),
                self.assertRaises(ValueError),
            ):
                agentx_data.collect_rows(records, metrics, 100.0)
        incomplete = copy.deepcopy(base)
        incomplete["responses"].pop()
        rows, counts = agentx_data.collect_rows(
            [incomplete], {"chatcmpl_1": metric}, 100.0
        )
        self.assertEqual(rows, [])
        self.assertEqual(counts["incomplete"], 1)

    def test_profile_command_keeps_scenario_and_rejects_short_measurement(self):
        command = agentx.profile_command("/env/python", "/model", "/run", 1234, 900, 42)
        for flag, value in (
            ("--scenario", "inferencex-agentx-mvp"),
            ("--concurrency", "6"),
            ("--public-dataset", agentx.DATASET),
            ("--benchmark-duration", "900"),
            ("--random-seed", "42"),
            ("--export-level", "raw"),
            ("--max-context-length", "262144"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--unsafe-override", command)
        self.assertNotIn("--ignore-trace-delays", command)
        with self.assertRaises(ValueError):
            agentx.profile_command("/env/python", "/model", "/run", 1234, 60, 42)


if __name__ == "__main__":
    unittest.main()
