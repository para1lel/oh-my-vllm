import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import deepseek_agentx
from .agentx_completion import write_tree_completion
from .deepseek_agentx import export_deepseek_run
from .test_agentx import export_fixture, raw_record


def log_line(message, logger="PhaseRunner", time="00:18:40.000"):
    level = "NOTICE" if message.startswith("Phase ") else "INFO"
    return f"1970-01-01 {time} - {logger} - {level} - {message}\n"


def teardown(phase="profiling", released=0):
    return log_line(
        f"Session-tree slots for phase {phase}: peak_open=2 (target concurrency 4); "
        f"released {released} still-open at teardown; late_events=0"
    )


def clean_fixture(root):
    records = [raw_record("first"), raw_record("second")]
    for index, record in enumerate(records):
        record["metadata"].update(
            root_correlation_id=f"root-{index}", phase_index=0, profiling_index=0
        )
    artifacts, summary, log = export_fixture(root, records)
    log = (
        log.replace("sent=14", "sent=2")
        .replace("completed=1, in_flight=13", "completed=2, in_flight=0")
        .replace("cancelled=12", "cancelled=0")
        .replace("cancelled=3", "cancelled=0")
        .replace(" | grace_period_timeout=True", "")
    )
    (artifacts / "logs/aiperf.log").write_text(log)
    return artifacts, summary, log


def assess(artifacts, root):
    return write_tree_completion(artifacts, root / "session-tree-status.json")


class AgentXCompletionTests(unittest.TestCase):
    def test_zero_cancellation_and_complete_request_accounting_leave_trees_unknown(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, log = clean_fixture(root)
            (artifacts / "logs/aiperf.log").write_text(log + teardown())
            rows, summary = export_deepseek_run(artifacts, root / "requests.csv")
            trees = summary["session_trees"]
            self.assertEqual(len(rows), 2)
            self.assertTrue(summary["accounting_complete"])
            self.assertEqual(
                summary["request_accounting_scope"], "issued_http_requests"
            )
            self.assertEqual(trees["status"], "unknown")
            self.assertFalse(trees["strict_target_passed"])
            self.assertIn("MISSING_COMPLETE_TREE_INVENTORY", trees["reasons"])
            self.assertEqual(trees["tree_counts"], dict(admitted=None, completed=None))
            self.assertEqual(trees["observed_root_sessions"]["sent"], 9)
            self.assertEqual(trees["observed_root_sessions"]["distinct_ids_in_raw"], 2)
            self.assertIsNone(trees["pending_children"])
            self.assertEqual(trees["pending_joins"], dict(active=None, future=None))
            self.assertEqual(json.loads((root / "summary.json").read_text()), summary)
            self.assertEqual(
                json.loads((root / "session-tree-status.json").read_text()), trees
            )
            self.assertNotIn(
                "session_trees", json.loads((root / "request-counts.json").read_text())
            )

    def test_no_http_in_flight_does_not_hide_pending_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, log = clean_fixture(root)
            branch = log_line(
                "BranchOrchestrator leaked state at cleanup: "
                "0 active_joins, 0 future_joins, "
                "1 tracked children, 1 parents with descendants",
                "aiperf.timing.branch_orchestrator",
            )
            (artifacts / "logs/aiperf.log").write_text(log + branch + teardown())
            trees = assess(artifacts, root)
            self.assertEqual(trees["status"], "incomplete")
            self.assertEqual(trees["pending_children"], 1)
            self.assertEqual(trees["pending_joins"], dict(active=0, future=0))
            self.assertIn("PENDING_BRANCH_STATE", trees["reasons"])

    def test_warmup_teardown_and_other_run_are_outside_profiling_scope(self):
        branch = log_line(
            "BranchOrchestrator leaked state at cleanup: "
            "8 active_joins, 9 future_joins, "
            "10 tracked children, 11 parents with descendants",
            "aiperf.timing.branch_orchestrator",
        )
        for suffix in (
            teardown("warmup", 7) + branch,
            log_line("Starting AIPerf System", "aiperf.cli_runner._single_run")
            + branch
            + teardown(released=7),
            (branch + teardown(released=7)).replace("1970-01-01", "1970-01-02"),
        ):
            with (
                self.subTest(suffix=suffix),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts, _, log = clean_fixture(root)
                warmup_record = raw_record("warmup", "warmup")
                warmup_record["metadata"].update(
                    root_correlation_id="warmup-root", was_cancelled=True
                )
                with (artifacts / "profile_export_raw.jsonl").open("a") as source:
                    source.write(json.dumps(warmup_record) + "\n")
                warmup = branch + teardown("warmup", 7)
                (artifacts / "logs/aiperf.log").write_text(warmup + log + suffix)
                trees = assess(artifacts, root)
                self.assertEqual(trees["status"], "unknown")
                self.assertIsNone(trees["pending_children"])
                self.assertIsNone(trees["forced_tree_slots"])
                self.assertFalse(trees["scope"]["teardown_bound"])
                self.assertEqual(
                    trees["observed_root_sessions"]["distinct_ids_in_raw"], 2
                )

    def test_intervening_phase_start_makes_log_attribution_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, log = clean_fixture(root)
            lines = log.splitlines(keepends=True)
            lines.insert(
                1, log_line("Phase warmup (warmup) started | target: 1 requests")
            )
            (artifacts / "logs/aiperf.log").write_text(
                "".join(lines) + teardown(released=9)
            )
            trees = assess(artifacts, root)
            self.assertEqual(trees["status"], "unknown")
            self.assertIn("AMBIGUOUS_PHASE_BOUNDARY", trees["reasons"])
            self.assertIsNone(trees["forced_tree_slots"])

    def test_forced_completion_and_teardown_retain_facts_and_successful_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, _, log = clean_fixture(root)
            lines = log.splitlines(keepends=True)
            lines.insert(
                2,
                log_line(
                    "Timeout waiting 10.0s for cancelled credits to return. "
                    "Some credits may be stuck. Forcing phase completion."
                ),
            )
            (artifacts / "logs/aiperf.log").write_text(
                "".join(lines) + teardown(released=2)
            )
            _, summary = export_deepseek_run(artifacts, root / "requests.csv")
            trees = summary["session_trees"]
            self.assertEqual(trees["status"], "incomplete")
            self.assertEqual(trees["forced_tree_slots"], 2)
            self.assertIn("FORCED_PHASE_COMPLETION", trees["reasons"])
            with (root / "requests.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)
            digest = hashlib.sha256(
                (artifacts / "logs/aiperf.log").read_bytes()
            ).hexdigest()
            self.assertEqual(trees["source_digests"]["harness_log"]["sha256"], digest)
            forced = next(
                fact
                for fact in trees["facts"]
                if fact["kind"] == "FORCED_PHASE_COMPLETION"
            )
            self.assertEqual(forced["line"], 3)

    def test_missing_log_or_missing_teardown_never_implies_completion(self):
        for missing_log in (False, True):
            with (
                self.subTest(missing_log=missing_log),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts, _, log = clean_fixture(root)
                path = artifacts / "logs/aiperf.log"
                if missing_log:
                    path.unlink()
                else:
                    path.write_text(
                        log
                        + log_line(
                            "BranchOrchestrator leaked state at cleanup: "
                            "1 active_joins, 0 future_joins, 1 tracked children, "
                            "1 parents with descendants",
                            "aiperf.timing.branch_orchestrator",
                        )
                    )
                trees = assess(artifacts, root)
                self.assertEqual(trees["status"], "unknown")
                self.assertIsNone(trees["pending_children"])
                self.assertIsNone(trees["forced_tree_slots"])
                if missing_log:
                    self.assertIsNone(trees["source_digests"]["harness_log"]["sha256"])
                    with self.assertRaises(ValueError):
                        export_deepseek_run(artifacts, root / "requests.csv")

    def test_stale_or_duplicate_profiling_log_has_no_assigned_teardown(self):
        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts, _, log = clean_fixture(root)
                content = (
                    log.replace("1970-01-01", "1970-01-02") if stale else log + log
                )
                (artifacts / "logs/aiperf.log").write_text(
                    content + teardown(released=5)
                )
                trees = assess(artifacts, root)
                self.assertEqual(trees["status"], "unknown")
                self.assertIn("UNBOUND_PROFILING_LOG", trees["reasons"])
                self.assertIsNone(trees["observed_root_sessions"]["sent"])
                self.assertIsNone(trees["forced_tree_slots"])

    def test_truncation_suppression_abandoned_join_and_cancellation_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, summary, log = clean_fixture(root)
            summary["branch_stats"] = dict(children_truncated=3, joins_suppressed=2)
            (artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
            log = log.replace(
                "sent=2, completed=2, in_flight=0", "sent=3, completed=2, in_flight=1"
            )
            log = log.replace(
                "complete | completed=2, cancelled=0",
                "complete | completed=2, cancelled=1",
            )
            abandoned = log_line(
                "Abandoned pending join for parent root-1 "
                "(outstanding=1, gated_turn_index=7, age_ms=10)",
                "aiperf.timing.branch_orchestrator",
            )
            (artifacts / "logs/aiperf.log").write_text(log + abandoned + teardown())
            rows, result = export_deepseek_run(artifacts, root / "requests.csv")
            trees = result["session_trees"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(result["cancelled_without_raw"], 1)
            self.assertEqual(trees["status"], "incomplete")
            self.assertEqual(trees["truncated_children"], 3)
            self.assertEqual(trees["suppressed_joins"], 2)
            for reason in (
                "TRUNCATED_CHILDREN",
                "SUPPRESSED_JOINS",
                "ABANDONED_JOIN",
                "CANCELLED_REQUESTS",
            ):
                self.assertIn(reason, trees["reasons"])
            self.assertIsNone(trees["pending_joins"]["active"])

    def test_failed_request_export_retains_tree_report_and_original_error(self):
        for case in (
            "cancelled",
            "missing_metric",
            "bad_raw",
            "missing_raw",
            "missing_summary",
            "bad_summary",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts, summary, _ = clean_fixture(root)
                if case == "cancelled":
                    summary["was_cancelled"] = True
                    (artifacts / "profile_export_aiperf.json").write_text(
                        json.dumps(summary)
                    )
                elif case == "missing_metric":
                    (root / "mtp4-service.log").write_text("")
                elif case == "bad_raw":
                    (artifacts / "profile_export_raw.jsonl").write_text("{bad\n")
                elif case == "missing_raw":
                    (artifacts / "profile_export_raw.jsonl").unlink()
                elif case == "missing_summary":
                    (artifacts / "profile_export_aiperf.json").unlink()
                else:
                    (artifacts / "profile_export_aiperf.json").write_text("{bad")
                expected_error = (
                    FileNotFoundError
                    if case.startswith("missing_") and case != "missing_metric"
                    else ValueError
                )
                with self.assertRaises(expected_error):
                    export_deepseek_run(artifacts, root / "requests.csv")
                trees = json.loads((root / "session-tree-status.json").read_text())
                self.assertEqual(
                    trees["status"], "incomplete" if case == "cancelled" else "unknown"
                )
                if case == "cancelled":
                    self.assertIn("RUN_CANCELLED", trees["reasons"])
                self.assertFalse((root / "requests.csv").exists())
                self.assertFalse((root / "request-counts.json").exists())

    def test_tree_report_write_error_preserves_request_result(self):
        for cancelled in (False, True):
            with (
                self.subTest(cancelled=cancelled),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts, summary, _ = clean_fixture(root)
                (root / "session-tree-status.json").mkdir()
                summary["was_cancelled"] = cancelled
                (artifacts / "profile_export_aiperf.json").write_text(
                    json.dumps(summary)
                )
                with contextlib.redirect_stderr(io.StringIO()) as diagnostic:
                    if cancelled:
                        with self.assertRaisesRegex(
                            ValueError, "valid completed measurement"
                        ):
                            export_deepseek_run(artifacts, root / "requests.csv")
                    else:
                        rows, result = export_deepseek_run(
                            artifacts, root / "requests.csv"
                        )
                        self.assertEqual(len(rows), 2)
                        self.assertEqual(result["session_trees"]["status"], "unknown")
                        self.assertIn(
                            "TREE_REPORT_WRITE_FAILED",
                            result["session_trees"]["reasons"],
                        )
                self.assertTrue(diagnostic.getvalue())
                self.assertEqual((root / "requests.csv").exists(), not cancelled)

    def test_harness_error_retains_exit_code_and_available_tree_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            args = type(
                "Arguments",
                (),
                dict(
                    artifacts=run,
                    model=root,
                    agentx_python=root / "python",
                    strategy="tp4",
                    gpu_uuids=(),
                    csv=root / "requests.csv",
                    duration=900,
                    seed=42,
                    grace_period=120,
                    cache_context=262144,
                    concurrency=4,
                    dataset="fixture",
                ),
            )()
            failure = deepseek_agentx.subprocess.CalledProcessError(23, ["harness"])

            def failed_harness(*_):
                artifacts, summary, _ = clean_fixture(run)
                summary["was_cancelled"] = True
                (artifacts / "profile_export_aiperf.json").write_text(
                    json.dumps(summary)
                )
                raise failure

            with (
                patch.object(deepseek_agentx, "arguments", return_value=args),
                patch.object(deepseek_agentx, "validate_agentx_python"),
                patch.object(deepseek_agentx, "_validate_harness_version"),
                patch.object(deepseek_agentx, "Tokenization"),
                patch.object(deepseek_agentx, "workload_identity", return_value={}),
                patch.object(deepseek_agentx, "artifact_identity", return_value={}),
                patch.object(deepseek_agentx, "Service"),
                patch.object(deepseek_agentx, "profile_command", return_value=[]),
                patch.object(
                    deepseek_agentx, "run_harness", side_effect=failed_harness
                ),
                self.assertRaises(
                    deepseek_agentx.subprocess.CalledProcessError
                ) as caught,
            ):
                deepseek_agentx.main()
            self.assertIs(caught.exception, failure)
            self.assertEqual(caught.exception.returncode, 23)
            trees = json.loads((run / "session-tree-status.json").read_text())
            self.assertEqual(trees["status"], "incomplete")
            self.assertIn("RUN_CANCELLED", trees["reasons"])
            self.assertFalse((root / "requests.csv").exists())

    def test_cleanup_requires_current_timestamp_and_explicit_teardown(self):
        branch = log_line(
            "BranchOrchestrator leaked state at cleanup: "
            "2 active_joins, 3 future_joins, 9 tracked children, "
            "4 parents with descendants",
            "aiperf.timing.branch_orchestrator",
        )
        for foreign in (False, True):
            with (
                self.subTest(foreign=foreign),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                artifacts, _, log = clean_fixture(root)
                lines = log.splitlines(keepends=True)
                lines.insert(
                    1, branch.replace("1970-01-01", "1970-01-02") if foreign else branch
                )
                (artifacts / "logs/aiperf.log").write_text(
                    "".join(lines) + (teardown() if foreign else "")
                )
                _, summary = export_deepseek_run(artifacts, root / "requests.csv")
                trees = summary["session_trees"]
                self.assertEqual(trees["status"], "unknown")
                self.assertIsNone(trees["pending_children"])
                self.assertEqual(trees["pending_joins"], dict(active=None, future=None))
                self.assertIn(
                    "UNBOUND_LOG_FACT" if foreign else "UNBOUND_BRANCH_CLEANUP",
                    trees["reasons"],
                )

    def test_raw_roots_require_explicit_indices_and_current_request_times(self):
        for case in ("indices", "timestamp", "partial"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts, _, _ = clean_fixture(root)
                path = artifacts / "profile_export_raw.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                for record in records[:1] if case == "partial" else records:
                    metadata = record["metadata"]
                    if case in ("indices", "partial"):
                        metadata.pop("phase_index")
                        metadata.pop("profiling_index")
                    else:
                        for key in ("request_start_ns", "request_end_ns"):
                            metadata[key] += 86_400_000_000_000
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                rows, summary = export_deepseek_run(artifacts, root / "requests.csv")
                trees = summary["session_trees"]
                self.assertEqual(len(rows), 2)
                self.assertTrue(summary["accounting_complete"])
                self.assertIsNone(
                    trees["observed_root_sessions"]["distinct_ids_in_raw"]
                )
                self.assertIn(
                    "UNBOUND_RAW_TIME" if case == "timestamp" else "UNBOUND_RAW_PHASE",
                    trees["reasons"],
                )

    def test_cleanup_uses_only_the_post_completion_phaserunner_teardown(self):
        logger = "aiperf.timing.branch_orchestrator"
        branch = log_line(
            "BranchOrchestrator leaked state at cleanup: 2 active_joins, "
            "3 future_joins, 9 tracked children, 4 parents with descendants",
            logger,
        )
        stats = log_line("BranchOrchestrator stats: joins_suppressed=7", logger)
        abandoned = log_line("Abandoned pending join for parent root-1", logger)
        early = teardown(released=9).replace("00:18:40.000", "00:01:41.000")
        for case in ("early_slot", "wrong_logger", "early_cleanup"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifacts, _, log = clean_fixture(root)
                lines = log.splitlines(keepends=True)
                if case == "wrong_logger":
                    log += branch + teardown(released=9).replace(
                        " - PhaseRunner - ", " - other.logger - "
                    )
                else:
                    lines.insert(1, early + branch + stats + abandoned)
                    log = "".join(lines) + (
                        teardown() if case == "early_cleanup" else ""
                    )
                (artifacts / "logs/aiperf.log").write_text(log)
                trees = assess(artifacts, root)
                self.assertEqual(trees["status"], "unknown")
                self.assertFalse(trees["strict_target_passed"])
                self.assertEqual(
                    trees["scope"]["teardown_bound"], case == "early_cleanup"
                )
                self.assertIsNone(trees["pending_children"])
                self.assertEqual(trees["pending_joins"], dict(active=None, future=None))
                self.assertIsNone(trees["suppressed_joins"])
                self.assertEqual(
                    trees["forced_tree_slots"], 0 if case == "early_cleanup" else None
                )


if __name__ == "__main__":
    unittest.main()
