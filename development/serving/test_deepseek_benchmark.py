import contextlib
import csv
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import deepseek_accounting as accounting
from . import deepseek_accounting_floor as accounting_floor
from .deepseek_accounting import (
    VERSION,
    BindingError,
    collective_profile_digest,
    ideal_trace,
    reference_trace,
)
from .deepseek_benchmark import (
    DEEPSEEK_NATIVE_MODULES,
    Service,
    argument_parser,
    artifact_identity,
    reference_configuration,
    runtime_reference,
    verify_artifact_identity,
    write_csv,
)
from .deepseek_runtime_trace import assemble_trace, rank_paths


def runtime_fixture(case, *, reference_version=VERSION):
    trace = ideal_trace(case)
    harness = {
        "reference": reference_configuration(reference_version),
        "source": "test-source-digest",
        "source_commit": "test-commit",
        "checkpoint": {"config": "config-digest", "index": "index-digest"},
        "tokenizer": {"tokenizer.json": "tokenizer-digest"},
        "native": {
            name: {"source_id": f"{name}-source", "binary_id": f"{name}-binary"}
            for name in DEEPSEEK_NATIVE_MODULES
        },
        "topology": [{"uuid": f"GPU-{rank}"} for rank in range(4)],
    }
    events = []
    for rank in range(4):
        identity = {
            "operation": "identity",
            "checkpoint": deepcopy(harness["checkpoint"]),
            "tokenizer": deepcopy(harness["tokenizer"]),
            "config": {"num_speculative_tokens": 5 if case["mode"] == "dspark5" else 0},
            "cache_memory": {
                "rank": rank,
                "strategy": case["strategy"],
                "effective_strategy": case["strategy"],
            },
            "execution": {
                "runtime": {
                    "rank": rank,
                    "gpu_uuid": f"GPU-{rank}",
                    "source_commit": harness["source_commit"],
                    "source_digest": harness["source"],
                    "native_modules": {
                        name: {
                            "source": value["source_id"],
                            "binary": value["binary_id"],
                        }
                        for name, value in harness["native"].items()
                    },
                }
            },
        }
        calls = [
            {"operation": "accounting_call", **deepcopy(record)}
            for record in trace["calls"]
            if record["rank"] == rank
        ]
        for record in calls:
            for row in record["request_rows"]:
                row["id"] += 100
        events.append([identity, *calls])
    responses = [
        {
            "id": f"chatcmpl_test_{ordinal + 101}",
            "usage": {
                "prompt_tokens": case["input_tokens"],
                "completion_tokens": case["output_tokens"],
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        for ordinal in range(case["batch"])
    ]
    return harness, events, responses


def write_runtime_events(path, events):
    for source, rows in zip(rank_paths(path), events, strict=True):
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_fixture(case, harness, events, responses, *, version):
    service = object.__new__(Service)
    service.arguments = SimpleNamespace(reference_version=version)
    service.port = 0
    service.request_timeout_seconds = 7200
    with tempfile.TemporaryDirectory() as directory:
        service.trace = Path(directory) / "execution.jsonl"
        write_runtime_events(service.trace, events)
        with (
            patch(
                "development.serving.deepseek_benchmark_service.exact_prompt",
                return_value=[],
            ),
            patch(
                "development.serving.deepseek_benchmark_service.post",
                return_value=responses[0],
            ),
            patch(
                "development.serving.deepseek_benchmark_service.time.perf_counter",
                side_effect=[100.0, 102.0],
            ),
        ):
            result = service.run(object(), case, "test", identity=harness)
        path = Path(directory) / "result.csv"
        write_csv(path, [result])
        with path.open() as source:
            return result, next(csv.DictReader(source))


class DeepSeekBenchmark(unittest.TestCase):
    def test_reference_cli_and_manifest_bind_the_selected_accountant(self):
        required = [
            "--model",
            "/model",
            "--artifacts",
            "/artifacts",
            "--csv",
            "/out.csv",
        ]
        parser = argument_parser()
        self.assertEqual(parser.parse_args(required).reference_version, VERSION)
        arguments = parser.parse_args(
            [*required, "--reference-version", accounting_floor.VERSION]
        )
        for version in ("wire", "unknown", "deepseek-http-reference-v3"):
            with (
                self.subTest(version=version),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([*required, "--reference-version", version])
        with tempfile.TemporaryDirectory() as directory:
            arguments.binary = Path(directory) / "service"
            arguments.binary.write_bytes(b"CPU test executable identity")
            with (
                patch(
                    "development.serving.deepseek_benchmark.source_identity",
                    return_value="source",
                ),
                patch(
                    "development.serving.deepseek_benchmark.native_identity",
                    return_value={},
                ),
                patch(
                    "development.serving.deepseek_benchmark.numeric_runtime_identity",
                    return_value={},
                ),
                patch(
                    "development.serving.deepseek_benchmark.checkpoint_descriptor_identity",
                    return_value={},
                ),
                patch(
                    "development.serving.deepseek_benchmark.topology_identity",
                    return_value=[],
                ),
            ):
                tokenizer = SimpleNamespace(identity={})
                expected = artifact_identity(
                    arguments, tokenizer, (), workload={"kind": "http"}
                )
                self.assertEqual(
                    expected["reference"],
                    reference_configuration(accounting_floor.VERSION),
                )
                arguments.reference_version = VERSION
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    verify_artifact_identity(
                        arguments, tokenizer, (), expected, workload={"kind": "http"}
                    )

    def test_optional_v4_dispatch_and_csv_preserve_runtime_source_identity(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                case = dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=1,
                    input_tokens=20,
                    output_tokens=4,
                )
                fixture = runtime_fixture(
                    case, reference_version=accounting_floor.VERSION
                )
                for rows in fixture[1]:
                    for row in rows[-1].get("request_rows", []):
                        row["proposed"] = 5
                result, csv_row = run_fixture(
                    case, *fixture, version=accounting_floor.VERSION
                )
                self.assertEqual(result["accounting_version"], accounting_floor.VERSION)
                self.assertEqual(result["source_trace_version"], VERSION)
                self.assertEqual(result["trace"]["binding"]["version"], VERSION)
                self.assertEqual(
                    result["source_trace_rate_profile"],
                    accounting.rate_profile_digest(),
                )
                self.assertEqual(
                    result["rate_profile"], accounting_floor.rate_profile_digest()
                )
                self.assertEqual(
                    result["assumptions_sha256"], accounting_floor.assumptions_digest()
                )
                self.assertEqual(
                    result["reference_scope"], "conditional partial resource floor"
                )
                self.assertEqual(result["reference_acceptance"], "trace-resource-floor")
                self.assertEqual((result["accepted"], result["proposed"]), (2, 5))
                self.assertIsNone(result["profile_digest"])
                self.assertFalse(result["dep4_communication_modeled"])
                self.assertEqual(
                    result["reference_seconds"],
                    accounting_floor.reference_trace(
                        result["trace"], strategy=strategy
                    ).seconds,
                )
                self.assertEqual(csv_row["source_trace_version"], VERSION)
                self.assertEqual(
                    csv_row["accounting_version"], accounting_floor.VERSION
                )
                for field in (
                    "reference_conditions",
                    "reference_zero_credit",
                    "reference_diagnostics",
                ):
                    self.assertEqual(json.loads(csv_row[field]), result[field])
                for rank in result["reference_resources"]["ranks"]:
                    self.assertEqual(
                        (
                            rank["hbm_bytes"],
                            rank["host_h2d_bytes"],
                            rank["host_d2h_bytes"],
                            rank["wire_sent_bytes"],
                            rank["wire_received_bytes"],
                        ),
                        (0, 0, 0, 0, 0),
                    )

    def test_v4_selection_rejects_missing_or_changed_runtime_and_manifest_identity(
        self,
    ):
        case = dict(
            mode="dspark5", strategy="tp4", batch=1, input_tokens=20, output_tokens=4
        )
        fixture = runtime_fixture(case, reference_version=accounting_floor.VERSION)
        for change in (
            "missing_runtime",
            "changed_gpu",
            "missing_reference",
            "changed_rate",
            "changed_assumptions",
        ):
            harness, events, responses = deepcopy(fixture)
            if change == "missing_runtime":
                del events[2][0]["execution"]["runtime"]
            elif change == "changed_gpu":
                events[2][0]["execution"]["runtime"]["gpu_uuid"] = "GPU-changed"
            elif change == "missing_reference":
                del harness["reference"]
            else:
                field = (
                    "rate_profile" if change == "changed_rate" else "assumptions_sha256"
                )
                harness["reference"][field] = "changed"
            with (
                self.subTest(change=change),
                self.assertRaises(BindingError),
            ):
                run_fixture(
                    case, harness, events, responses, version=accounting_floor.VERSION
                )

    def test_v4_dispatch_rejects_missing_or_relabelled_source_binding(self):
        case = dict(
            mode="dspark5", strategy="tep4", batch=1, input_tokens=20, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        trace = assemble_trace(events, responses, harness=harness, case=case)
        with self.assertRaises(BindingError):
            runtime_reference(
                {},
                strategy="tep4",
                expected_binding=trace["binding"],
                version=accounting_floor.VERSION,
            )
        for field in ("source", "version", "rate_profile"):
            for missing in (False, True):
                broken = deepcopy(trace)
                if missing:
                    del broken["binding"][field]
                else:
                    broken["binding"][field] = "changed"
                with (
                    self.subTest(field=field, missing=missing),
                    self.assertRaises(BindingError),
                ):
                    runtime_reference(
                        broken,
                        strategy="tep4",
                        expected_binding=trace["binding"],
                        version=accounting_floor.VERSION,
                    )

    def test_result_and_csv_hold_measured_accounting(self):
        case = dict(
            mode="dspark5", strategy="tp4", batch=1, input_tokens=2, output_tokens=4
        )
        harness, events, responses = runtime_fixture(case)
        # Rejected proposals must come from runtime counters, not ideal acceptance.
        for rows in events:
            rows[-1]["request_rows"][0]["proposed"] = 5
        service = object.__new__(Service)
        service.arguments = SimpleNamespace()
        service.port = 0
        service.request_timeout_seconds = 14400
        with tempfile.TemporaryDirectory() as directory:
            service.trace = Path(directory) / "execution.jsonl"
            write_runtime_events(service.trace, events)
            with (
                patch(
                    "development.serving.deepseek_benchmark_service.exact_prompt",
                    return_value=[],
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.post",
                    return_value=responses[0],
                ) as send,
                patch(
                    "development.serving.deepseek_benchmark_service.time.perf_counter",
                    side_effect=[100.0, 102.0],
                ),
            ):
                result = service.run(object(), case, "test", identity=harness)
                self.assertEqual(send.call_args.kwargs["timeout"], 14400)
            path = Path(directory) / "result.csv"
            write_csv(path, [result])
            with path.open() as source:
                row = next(csv.DictReader(source))
        self.assertEqual(result["accounting_version"], VERSION)
        self.assertEqual(result["communication_model"], "lower_bound")
        self.assertEqual(result["profile_digest"], collective_profile_digest())
        self.assertFalse(result["dep4_communication_measured"])
        self.assertEqual(result["reference_acceptance"], "trace")
        self.assertEqual((result["accepted"], result["proposed"]), (2, 5))
        self.assertEqual(result["trace"]["manifest"]["requests"][0]["id"], 101)
        self.assertEqual(
            result["trace"]["binding"]["source"],
            {"commit": harness["source_commit"], "digest": harness["source"]},
        )
        self.assertEqual(
            result["reference_seconds"],
            reference_trace(result["trace"], strategy="tp4").seconds,
        )
        self.assertEqual(result["efficiency_percent"], 50 * result["reference_seconds"])
        self.assertNotIn("ideal_efficiency_percent", row)
        self.assertEqual(row["reference_acceptance"], "trace")
        self.assertEqual(float(row["reference_seconds"]), result["reference_seconds"])
        self.assertEqual(float(row["efficiency_percent"]), result["efficiency_percent"])
        self.assertEqual(row["accounting_version"], VERSION)
        self.assertEqual(row["communication_model"], "lower_bound")
        self.assertEqual(row["profile_digest"], collective_profile_digest())
        self.assertEqual(
            row["communication_rule_version"], result["communication_rule_version"]
        )


if __name__ == "__main__":
    unittest.main()
