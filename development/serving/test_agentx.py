import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import agentx, agentx_data, deepseek_agentx
from .deepseek_agentx import profile_command as deepseek_profile_command
from .deepseek_agentx import workload_identity


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


def export_fixture(root, records):
    artifacts = root / "agentx"
    artifacts.mkdir()
    summary = {
        "aiperf_version": "1.0.7",
        "metadata": {"submission_valid": True},
        "was_cancelled": False,
        "start_time": "1970-01-01T00:01:40",
        "end_time": "1970-01-01T00:18:40",
        "request_count": {"avg": 2},
    }
    (artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
    (artifacts / "profile_export_raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (root / "mtp4-service.log").write_text(
        "".join(
            "SERVING_REQUEST_METRIC " + json.dumps(server_metric(name)) + "\n"
            for name in ("first", "second")
        )
    )
    logs = artifacts / "logs"
    logs.mkdir()
    log = (
        "1970-01-01 00:01:40.000 - PhaseRunner - NOTICE - "
        "Phase profiling (profiling) started | phase_index=0 | profiling_index=0 | "
        "target: 900.0s duration\n"
        "1970-01-01 00:16:40.000 - PhaseRunner - NOTICE - "
        "Phase profiling (profiling) sending complete | sent=14, completed=1, "
        "in_flight=13 | sessions: sent=9, completed=0 | timeout_triggered=True\n"
        "1970-01-01 00:18:40.000 - PhaseRunner - NOTICE - "
        "Phase profiling (profiling) complete | completed=2, cancelled=12, errors=0 | "
        "sessions: completed=0, cancelled=3 | elapsed=1020.00s | "
        "grace_period_timeout=True\n"
    )
    (logs / "aiperf.log").write_text(log)
    return artifacts, summary, log


class AgentXTests(unittest.TestCase):
    def test_export_accounts_for_drain_cancellations_without_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, _ = export_fixture(
                root, [raw_record("first"), raw_record("second")]
            )
            output = root / "requests.csv"
            rows, counts = agentx_data.export_run(artifacts, output)

            self.assertEqual(counts["cancelled"], 12)
            self.assertEqual(counts["sent"], 14)
            self.assertEqual(counts["cancelled_without_raw"], 12)
            self.assertTrue(counts["accounting_complete"])
            self.assertEqual(counts["profiling_seconds"], 900)
            self.assertEqual(counts["drain_seconds"], 120)
            self.assertTrue(counts["grace_period_timeout"])
            self.assertEqual(len(rows), 2)
            with output.open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)
            self.assertEqual(
                json.loads((root / "request-counts.json").read_text()), counts
            )

    def test_export_rejects_missing_or_inconsistent_phase_totals(self):
        for case in (
            "missing",
            "truncated",
            "duplicate",
            "malformed",
            "unreturned",
            "missing_result",
            "errors",
            "version",
            "stale",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts, summary, log = export_fixture(
                    root, [raw_record("first"), raw_record("second")]
                )
                path = artifacts / "logs/aiperf.log"
                if case == "missing":
                    path.unlink()
                elif case == "truncated":
                    path.write_text(log.rsplit("\n", 2)[0] + "\n")
                elif case == "duplicate":
                    path.write_text(log + log)
                elif case == "malformed":
                    path.write_text(log.replace("sent=14", "sent=unknown"))
                elif case == "unreturned":
                    path.write_text(log.replace("cancelled=12", "cancelled=11"))
                elif case == "missing_result":
                    path.write_text(
                        log.replace("completed=2", "completed=3").replace(
                            "cancelled=12", "cancelled=11"
                        )
                    )
                elif case == "errors":
                    path.write_text(log.replace("errors=0", "errors=1"))
                elif case == "version":
                    summary["aiperf_version"] = "1.0.8"
                elif case == "stale":
                    summary["start_time"] = "1970-01-02T00:01:40"
                (artifacts / "profile_export_aiperf.json").write_text(
                    json.dumps(summary)
                )
                with self.assertRaises(ValueError):
                    agentx_data.export_run(artifacts, root / "requests.csv")
                self.assertFalse((root / "requests.csv").exists())
                self.assertFalse((root / "request-counts.json").exists())

    def test_export_reconciles_errors_skips_and_exported_cancellations(self):
        failed = raw_record("failed")
        failed.update(status=503, error={"message": "worker unavailable"})
        incomplete = raw_record("incomplete")
        incomplete["responses"].pop()
        cancelled = raw_record("cancelled")
        cancelled["metadata"]["was_cancelled"] = True
        overflow = raw_record("overflow")
        overflow["metadata"]["context_overflow_skip"] = True
        records = [raw_record("first"), raw_record("second")]
        records.extend([failed, incomplete, cancelled, overflow])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, summary, log = export_fixture(root, records)
            summary["error_request_count"] = {"avg": 3}
            summary["skipped_context_overflow_count"] = {"avg": 1}
            (artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
            log = (
                log.replace("sent=14", "sent=8")
                .replace("in_flight=13", "in_flight=7")
                .replace("completed=2", "completed=5")
                .replace("cancelled=12", "cancelled=3")
                .replace("errors=0", "errors=2")
            )
            explicit_log = root / "harness.log"
            (artifacts / "logs/aiperf.log").unlink()
            explicit_log.write_text(log)
            rows, counts = agentx_data.export_run(
                artifacts, root / "requests.csv", harness_log=explicit_log
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(counts["failed"], 1)
            self.assertEqual(counts["incomplete"], 1)
            self.assertEqual(counts["cancelled"], 3)
            self.assertEqual(counts["cancelled_without_raw"], 2)
            self.assertEqual(counts["skipped_context_overflow"], 1)
            self.assertEqual(counts["sent"], 8)

    def test_export_keeps_complete_runs_without_cancellations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, log = export_fixture(
                root, [raw_record("first"), raw_record("second")]
            )
            log = (
                log.replace("sent=14", "sent=2")
                .replace("in_flight=13", "in_flight=1")
                .replace("cancelled=12", "cancelled=0")
                .replace(" | grace_period_timeout=True", "")
            )
            (artifacts / "logs/aiperf.log").write_text(log)
            _, counts = agentx_data.export_run(artifacts, root / "requests.csv")

            self.assertEqual(counts["sent"], 2)
            self.assertEqual(counts["cancelled"], 0)
            self.assertEqual(counts["cancelled_without_raw"], 0)
            self.assertFalse(counts["grace_period_timeout"])

    def test_deepseek_workload_identity_binds_dataset_and_grace_period(self):
        arguments = type(
            "Arguments",
            (),
            {
                "seed": 42,
                "strategy": "tep4",
                "concurrency": 16,
                "duration": 900,
                "grace_period": 45.5,
                "prompt_seed": "agentx-deepseek-42",
                "dataset": "semianalysis_cc_traces_weka_062126",
            },
        )()

        self.assertEqual(workload_identity(arguments)["grace_period"], 45.5)
        self.assertEqual(
            workload_identity(arguments)["dataset"],
            "semianalysis_cc_traces_weka_062126",
        )

    def test_deepseek_dataset_selection_keeps_the_approved_default(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "deepseek_agentx",
                "--model",
                "/model",
                "--artifacts",
                directory,
                "--agentx-python",
                "/env/python",
                "--csv",
                f"{directory}/result.csv",
                "--figure",
                f"{directory}/result.png",
                "--strategy",
                "tep4",
                "--concurrency",
                "16",
            ]
            for selection, expected in (
                ([], agentx.DATASET),
                (
                    ["--dataset", "semianalysis_cc_traces_weka_062126"],
                    "semianalysis_cc_traces_weka_062126",
                ),
            ):
                with (
                    self.subTest(dataset=expected),
                    patch("sys.argv", [*argv, *selection]),
                    patch.object(
                        deepseek_agentx,
                        "selected_uuids",
                        return_value=("a", "b", "c", "d"),
                    ),
                ):
                    arguments = deepseek_agentx.arguments()
                    self.assertEqual(getattr(arguments, "dataset", None), expected)

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
            ("--concurrency", "4"),
            ("--public-dataset", agentx.DATASET),
            ("--benchmark-duration", "900"),
            ("--benchmark-grace-period", "120"),
            ("--agentic-warmup-grace-period", "1800"),
            ("--random-seed", "42"),
            ("--export-level", "raw"),
            ("--max-context-length", "262144"),
            ("--request-timeout-seconds", "1800"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--unsafe-override", command)
        self.assertNotIn("--ignore-trace-delays", command)
        with self.assertRaises(ValueError):
            agentx.profile_command("/env/python", "/model", "/run", 1234, 60, 42)
        argv = [
            "agentx",
            "--model",
            "/model",
            "--artifacts",
            "/run/agentx-test-artifacts",
            "--agentx-python",
            "/env/python",
            "--csv",
            "/run/agentx-test.csv",
            "--figure",
            "/run/agentx-test.png",
        ]
        with (
            patch("sys.argv", argv),
            patch.object(agentx, "selected_uuid", return_value="gpu"),
        ):
            self.assertEqual(agentx.arguments().grace_period, 120)
        for value in ("0", "30.5", "600", "10000"):
            with (
                self.subTest(grace_period=value),
                patch("sys.argv", [*argv, "--grace-period", value]),
                patch.object(agentx, "selected_uuid", return_value="gpu"),
            ):
                args = agentx.arguments()
                self.assertEqual(args.grace_period, float(value))
                command = agentx.profile_command(
                    "/env/python", "/model", "/run", 1234, 900, 42, args.grace_period
                )
                self.assertEqual(
                    float(command[command.index("--benchmark-grace-period") + 1]),
                    float(value),
                )
        for value in ("-1", "nan", "inf", "-inf", "bad"):
            with self.subTest(grace_period=value):
                with self.assertRaises(ValueError):
                    agentx.profile_command(
                        "/env/python", "/model", "/run", 1234, 900, 42, value
                    )
                with (
                    patch("sys.argv", [*argv, f"--grace-period={value}"]),
                    patch.object(agentx, "selected_uuid", return_value="gpu"),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    agentx.arguments()
                self.assertEqual(error.exception.code, 2)

    def test_deepseek_profile_command_supports_a_million_token_context(self):
        command = deepseek_profile_command(
            "/env/python",
            "/model",
            "/run",
            1234,
            900,
            42,
            max_context=1048576,
            concurrency=16,
            dataset="semianalysis_cc_traces_weka_062126",
        )

        self.assertEqual(
            command[command.index("--max-context-length") + 1],
            "1048576",
        )
        self.assertEqual(
            command[command.index("--concurrency") + 1],
            "16",
        )
        self.assertEqual(
            command[command.index("--public-dataset") + 1],
            agentx.FULL_CONTEXT_DATASET,
        )


if __name__ == "__main__":
    unittest.main()
