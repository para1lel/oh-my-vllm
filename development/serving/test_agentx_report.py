import contextlib
import csv
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import agentx
from .test_agentx_completion import clean_fixture, teardown


@contextlib.contextmanager
def qwen_main_fixture(root, harness):
    args = SimpleNamespace(
        artifacts=root / "run",
        model=root / "model",
        agentx_python=root / "python",
        csv=root / "requests.csv",
        figure=root / "figure.png",
        duration=900,
        seed=42,
        grace_period=120,
    )
    identity = dict(device=dict(name="NVIDIA B200"), checkpoint={}, tokenizer={})
    with (
        patch.object(agentx, "arguments", return_value=args),
        patch.object(agentx, "validate_agentx_python"),
        patch.object(agentx.subprocess, "check_output", return_value="1.0.7\n"),
        patch.object(agentx, "artifact_identity", return_value=identity),
        patch.object(agentx, "verify_artifact_identity"),
        patch.object(agentx, "Service") as service,
        patch.object(agentx, "run_harness", side_effect=harness),
        patch.object(agentx.subprocess, "run"),
        contextlib.redirect_stdout(io.StringIO()) as output,
        contextlib.redirect_stderr(io.StringIO()) as diagnostics,
    ):
        yield args, output, diagnostics, service.return_value


def fixture_harness(
    run, *, cancelled=False, blocked_report=False, released=0, failure=None
):
    def harness(*_):
        artifacts, summary, log = clean_fixture(run)
        summary["was_cancelled"] = cancelled
        (artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
        (artifacts / "logs/aiperf.log").write_text(log + teardown(released=released))
        if blocked_report:
            (run / "session-tree-status.json").mkdir()
        if failure is not None:
            raise failure

    return harness


class AgentXReportTests(unittest.TestCase):
    def test_success_reports_request_scope_and_same_trees_in_summary_and_stdout(self):
        for released, status in ((0, "unknown"), (1, "incomplete")):
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                harness = fixture_harness(root / "run", released=released)
                with qwen_main_fixture(root, harness) as (args, output, _, service):
                    agentx.main()
                summary = json.loads((args.artifacts / "summary.json").read_text())
                trees = json.loads(
                    (args.artifacts / "session-tree-status.json").read_text()
                )
                self.assertEqual(
                    json.loads(output.getvalue().splitlines()[-1]), summary
                )
                self.assertEqual(summary["session_trees"], trees)
                self.assertEqual(
                    summary["request_accounting_scope"], "issued_http_requests"
                )
                self.assertTrue(summary["accounting_complete"])
                self.assertEqual(summary["completed"], 2)
                self.assertEqual(trees["status"], status)
                self.assertFalse(trees["strict_target_passed"])
                with args.csv.open() as source:
                    self.assertEqual(len(list(csv.DictReader(source))), 2)
                service.close.assert_called_once()

    def test_harness_and_export_errors_save_available_trees_without_success_outputs(
        self,
    ):
        for stage in ("harness", "export"):
            for blocked_report in (False, True):
                with (
                    self.subTest(stage=stage, blocked_report=blocked_report),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    failure = subprocess.CalledProcessError(23, ["fixture-harness"])

                    harness = fixture_harness(
                        root / "run",
                        cancelled=True,
                        blocked_report=blocked_report,
                        failure=failure if stage == "harness" else None,
                    )

                    with qwen_main_fixture(root, harness) as (
                        args,
                        output,
                        diagnostics,
                        service,
                    ):
                        expected = type(failure) if stage == "harness" else ValueError
                        with self.assertRaises(expected) as caught:
                            agentx.main()
                    if stage == "harness":
                        self.assertIs(caught.exception, failure)
                        self.assertEqual(caught.exception.returncode, 23)
                    report = args.artifacts / "session-tree-status.json"
                    if blocked_report:
                        self.assertTrue(diagnostics.getvalue())
                        self.assertTrue(report.is_dir())
                    else:
                        trees = json.loads(report.read_text())
                        self.assertEqual(trees["status"], "incomplete")
                        self.assertIn("RUN_CANCELLED", trees["reasons"])
                    self.assertFalse(args.csv.exists())
                    self.assertFalse(args.figure.exists())
                    self.assertFalse((args.artifacts / "summary.json").exists())
                    self.assertFalse((args.artifacts / "request-counts.json").exists())
                    self.assertEqual(len(output.getvalue().splitlines()), 1)
                    service.close.assert_called_once()

    def test_report_write_error_keeps_valid_request_result_and_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = fixture_harness(root / "run", blocked_report=True)
            with qwen_main_fixture(root, harness) as (args, output, diagnostics, _):
                agentx.main()
            summary = json.loads((args.artifacts / "summary.json").read_text())
            self.assertEqual(summary, json.loads(output.getvalue().splitlines()[-1]))
            self.assertEqual(summary["session_trees"]["status"], "unknown")
            self.assertIn(
                "TREE_REPORT_WRITE_FAILED", summary["session_trees"]["reasons"]
            )
            self.assertTrue(summary["accounting_complete"])
            self.assertTrue(diagnostics.getvalue())
            with args.csv.open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)

    def test_unexpected_report_error_does_not_replace_original_exception(self):
        for stage in ("harness", "export"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                failure = (
                    subprocess.CalledProcessError(23, ["fixture-harness"])
                    if stage == "harness"
                    else ValueError("fixture export failure")
                )

                harness = fixture_harness(
                    root / "run", failure=failure if stage == "harness" else None
                )

                with (
                    qwen_main_fixture(root, harness) as (args, _, diagnostics, service),
                    patch.object(agentx, "export_run", side_effect=failure),
                    patch.object(
                        agentx,
                        "write_tree_completion",
                        side_effect=RuntimeError("fixture report failure"),
                    ),
                    self.assertRaises(type(failure)) as caught,
                ):
                    agentx.main()
                self.assertIs(caught.exception, failure)
                self.assertTrue(diagnostics.getvalue())
                self.assertFalse(args.csv.exists())
                self.assertFalse((args.artifacts / "summary.json").exists())
                service.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
