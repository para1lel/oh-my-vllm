import contextlib
import csv
import hashlib
import io
import json
import os
import socket
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

from . import deepseek_accounting as accounting
from . import deepseek_accounting_v4 as accounting_v4
from .deepseek_accounting import (
    VERSION,
    BindingError,
    TraceError,
    collective_profile_digest,
    digest,
    ideal_trace,
    reference_trace,
)
from .deepseek_agentx import service_max_num_seqs
from .deepseek_benchmark import (
    DEEPSEEK_NATIVE_MODULES,
    Service,
    argument_parser,
    artifact_identity,
    checkpoint_descriptor_identity,
    effective_cache_configuration,
    effective_service_configuration,
    native_identity,
    reference_configuration,
    request_body,
    runtime_reference,
    verify_artifact_identity,
    write_csv,
)
from .deepseek_runtime_trace import assemble_trace, rank_paths, read_rank_events


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
                "development.serving.deepseek_benchmark.exact_prompt", return_value=[]
            ),
            patch(
                "development.serving.deepseek_benchmark.post", return_value=responses[0]
            ),
            patch(
                "development.serving.deepseek_benchmark.time.perf_counter",
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
            [*required, "--reference-version", accounting_v4.VERSION]
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
                    reference_configuration(accounting_v4.VERSION),
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
                fixture = runtime_fixture(case, reference_version=accounting_v4.VERSION)
                for rows in fixture[1]:
                    for row in rows[-1].get("request_rows", []):
                        row["proposed"] = 5
                with (
                    patch.object(
                        accounting_v4,
                        "reference_trace",
                        wraps=accounting_v4.reference_trace,
                    ) as selected,
                    patch.object(
                        accounting_v4,
                        "reference_case",
                        side_effect=AssertionError("static reference used"),
                    ),
                    patch(
                        "development.serving.deepseek_benchmark.reference_trace",
                        side_effect=AssertionError("legacy dispatch used"),
                    ),
                ):
                    result, csv_row = run_fixture(
                        case, *fixture, version=accounting_v4.VERSION
                    )
                selected.assert_called_once()
                self.assertEqual(result["accounting_version"], accounting_v4.VERSION)
                self.assertEqual(result["source_trace_version"], VERSION)
                self.assertEqual(result["trace"]["binding"]["version"], VERSION)
                self.assertEqual(
                    result["source_trace_rate_profile"],
                    accounting.rate_profile_digest(),
                )
                self.assertEqual(
                    result["rate_profile"], accounting_v4.rate_profile_digest()
                )
                self.assertEqual(
                    result["assumptions_sha256"], accounting_v4.assumptions_digest()
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
                    accounting_v4.reference_trace(
                        result["trace"], strategy=strategy
                    ).seconds,
                )
                self.assertEqual(csv_row["source_trace_version"], VERSION)
                self.assertEqual(csv_row["accounting_version"], accounting_v4.VERSION)
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
        fixture = runtime_fixture(case, reference_version=accounting_v4.VERSION)
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
                patch.object(
                    accounting_v4,
                    "reference_trace",
                    side_effect=AssertionError("invalid identity was priced"),
                ),
                self.assertRaises(BindingError),
            ):
                run_fixture(
                    case, harness, events, responses, version=accounting_v4.VERSION
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
                version=accounting_v4.VERSION,
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
                        version=accounting_v4.VERSION,
                    )

    def test_request_timeout_parses_and_reaches_service_and_http_client(self):
        from .deepseek_benchmark import post

        parser = argument_parser()
        required = [
            "--model",
            "/model",
            "--artifacts",
            "/artifacts",
            "--csv",
            "/result.csv",
        ]
        self.assertEqual(parser.parse_args(required).request_timeout_seconds, 7200)
        for value in ("0", "-1", "nan", "1.5"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args([*required, "--request-timeout-seconds", value])
        arguments = parser.parse_args([*required, "--request-timeout-seconds", "14400"])
        response = Mock(status=200)
        response.read.return_value = b"{}"
        connection = Mock()
        connection.getresponse.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            arguments.artifacts = Path(directory)
            with patch(
                "development.serving.deepseek_benchmark.subprocess.Popen"
            ) as launch:
                launch.return_value.poll.return_value = 0
                service = Service(arguments, "ordinary", "tp4")
            try:
                command = launch.call_args.args[0]
                self.assertEqual(
                    command[command.index("--request-timeout-seconds") + 1], "14400"
                )
                with patch(
                    "development.serving.deepseek_benchmark.http.client.HTTPConnection",
                    return_value=connection,
                ) as connect:
                    post(1234, {}, timeout=service.request_timeout_seconds)
                self.assertEqual(connect.call_args.kwargs["timeout"], 14400)
                connection.close.assert_called_once()
            finally:
                service.close()

    def test_stale_service_socket_reports_failure_after_directory_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                artifacts=root,
                model=root,
                binary=root / "service",
                cache_context=1048576,
                blocks=32,
                mamba_blocks=8,
            )

            class FinishedProcess:
                def poll(self):
                    return 0

            with patch(
                "development.serving.deepseek_benchmark.subprocess.Popen",
                return_value=FinishedProcess(),
            ):
                service = Service(arguments, "ordinary", "tp4")
            directory = service.ipc.parent
            with socket.socket(socket.AF_UNIX) as stale:
                stale.bind(str(service.ipc))
            with self.assertRaisesRegex(RuntimeError, "did not remove.*IPC"):
                service.close()
            self.assertFalse(directory.exists())

    def test_failed_service_launch_removes_its_temporary_ipc_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                artifacts=root,
                model=root,
                binary=root / "service",
                cache_context=1048576,
                blocks=32,
                mamba_blocks=8,
            )
            directories = []

            def fail(command, **kwargs):
                directories.append(Path(command[command.index("--socket") + 1]).parent)
                raise OSError("cannot start test worker")

            with (
                patch("tempfile.tempdir", str(root)),
                patch(
                    "development.serving.deepseek_benchmark.subprocess.Popen",
                    side_effect=fail,
                ),
                self.assertRaisesRegex(OSError, "cannot start"),
            ):
                Service(arguments, "ordinary", "tp4")
            self.assertEqual(len(directories), 1)
            self.assertFalse(directories[0].exists())

    def test_services_use_distinct_temporary_ipc_directories_and_remove_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                artifacts=root,
                model=root,
                binary=root / "service",
                cache_context=1048576,
                blocks=32,
                mamba_blocks=8,
            )

            class FinishedProcess:
                def poll(self):
                    return 0

            with (
                patch("tempfile.tempdir", str(root)),
                patch(
                    "development.serving.deepseek_benchmark.subprocess.Popen",
                    return_value=FinishedProcess(),
                ),
            ):
                services = [
                    Service(arguments, "ordinary", "tp4"),
                    Service(arguments, "dspark5", "tep4"),
                ]
            try:
                self.assertTrue(
                    all(service.ipc.is_relative_to(root) for service in services)
                )
                self.assertNotEqual(services[0].ipc, services[1].ipc)
                directories = [service.ipc.parent for service in services]
                self.assertTrue(all(directory.is_dir() for directory in directories))
            finally:
                for service in services:
                    service.close()
            self.assertTrue(all(not directory.exists() for directory in directories))

    def test_effective_strategy_reads_the_execution_trace_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                artifacts=root,
                model=root,
                binary=root / "service",
                cache_context=1048576,
                blocks=32,
                mamba_blocks=8,
            )

            class FinishedProcess:
                def poll(self):
                    return 0

            with patch(
                "development.serving.deepseek_benchmark.subprocess.Popen",
                return_value=FinishedProcess(),
            ):
                service = Service(arguments, "dspark5", "dep4")
            service.trace.write_text(
                json.dumps(
                    {
                        "operation": "identity",
                        "cache_memory": {"effective_strategy": "tep4"},
                    }
                )
                + "\n"
            )
            try:
                self.assertEqual(service.effective_strategy(), "tep4")
            finally:
                service.close()

    def test_numeric_runtime_changes_invalidate_the_measurement_identity(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "service"
            binary.write_bytes(b"fixed service executable")
            arguments = SimpleNamespace(
                binary=binary,
                model=Path(directory),
                cache_context=1048576,
                blocks=16,
                mamba_blocks=8,
            )
            tokenizer = SimpleNamespace(identity={"tokenizer": "fixed"})
            precision = torch.get_float32_matmul_precision()
            with (
                patch.dict("os.environ"),
                patch(
                    "development.serving.deepseek_benchmark.native_identity",
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
                os.environ.pop("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", None)
                expected = artifact_identity(arguments, tokenizer, (), workload={})
                self.assertEqual(expected["configuration"]["native_sparse"], "1")
                for override in ("0", "1"):
                    os.environ["OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE"] = override
                    actual = artifact_identity(arguments, tokenizer, (), workload={})
                    self.assertEqual(actual["configuration"]["native_sparse"], override)
                    if override == "0":
                        with self.assertRaisesRegex(RuntimeError, "changed"):
                            verify_artifact_identity(
                                arguments, tokenizer, (), expected, workload={}
                            )
                try:
                    torch.set_float32_matmul_precision(
                        "high" if precision != "high" else "highest"
                    )
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        verify_artifact_identity(
                            arguments, tokenizer, (), expected, workload={}
                        )
                finally:
                    torch.set_float32_matmul_precision(precision)

    def test_native_identity_reads_the_registered_paged_allocator_manifest(self):
        from oh_my_vllm.operators.runtime.native import (
            MODULE_SOURCES,
            OPTIONAL_MODULE_SOURCES,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "native.so"
            binary.write_bytes(b"immutable native test image")
            metadata = {
                "library": str(binary),
                "binary_id": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
            for name in MODULE_SOURCES | OPTIONAL_MODULE_SOURCES:
                (root / f"{name}.json").write_text(json.dumps(metadata))
            with patch(
                "development.serving.deepseek_benchmark.cache_directory",
                return_value=root,
            ):
                identity = native_identity()
            self.assertEqual(identity["paged"], metadata)

    def test_agentx_default_sequence_capacity_follows_concurrency(self):
        self.assertEqual(service_max_num_seqs(16), 16)
        self.assertEqual(service_max_num_seqs(64), 64)
        self.assertEqual(service_max_num_seqs(16, 32), 32)
        with self.assertRaises(ValueError):
            service_max_num_seqs(16, 65)

    def test_identity_uses_environment_cache_capacity_when_no_cli_override_exists(self):
        arguments = SimpleNamespace(blocks=None, mamba_blocks=None)
        with patch.dict("os.environ", {"OH_MY_VLLM_NUM_GPU_BLOCKS": "17"}):
            configuration = effective_cache_configuration(arguments)

        self.assertEqual(configuration, {"blocks": 17, "mamba_blocks": 128})

    def test_service_identity_uses_the_effective_sequence_capacity(self):
        self.assertEqual(
            effective_service_configuration(SimpleNamespace(max_num_seqs=16)),
            {
                "max_num_batched_tokens": 32768,
                "max_num_seqs": 16,
                "prefill_rows": 8192,
            },
        )
        self.assertEqual(
            effective_service_configuration(SimpleNamespace()),
            {
                "max_num_batched_tokens": 32768,
                "max_num_seqs": 64,
                "prefill_rows": 8192,
            },
        )
        self.assertEqual(
            effective_service_configuration(
                SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=1)
            ),
            {
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 1,
                "prefill_rows": 8192,
            },
        )

    def test_service_identity_records_the_prefill_row_override(self):
        with patch.dict("os.environ", {"OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS": "1024"}):
            configuration = effective_service_configuration(SimpleNamespace())

        self.assertEqual(configuration["prefill_rows"], 1024)

    def test_request_body_uses_the_service_reasoning_field(self):
        body = request_body([{"role": "user", "content": "hello"}], 4, 7)

        self.assertNotIn("effort", body)
        self.assertEqual(body["reasoning_effort"], "off")
        self.assertTrue(body["ignore_eos"])

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
                    "development.serving.deepseek_benchmark.exact_prompt",
                    return_value=[],
                ),
                patch(
                    "development.serving.deepseek_benchmark.post",
                    return_value=responses[0],
                ) as send,
                patch(
                    "development.serving.deepseek_benchmark.time.perf_counter",
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

    def test_checkpoint_descriptor_does_not_read_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_bytes(b"config")
            (root / "model.safetensors.index.json").write_bytes(b"index")
            self.assertEqual(
                checkpoint_descriptor_identity(root),
                {
                    "config": hashlib.sha256(b"config").hexdigest(),
                    "index": hashlib.sha256(b"index").hexdigest(),
                },
            )

    def test_manifest_uses_response_ids_and_http_usage(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=2, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        responses.reverse()
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(
            [row["id"] for row in trace["manifest"]["requests"]], [102, 101]
        )
        self.assertEqual(trace["binding"]["manifest_sha256"], digest(trace["manifest"]))
        for field, value in (("completion_tokens", 3), ("prompt_tokens", True)):
            broken = deepcopy(responses)
            broken[0]["usage"][field] = value
            with self.subTest(field=field), self.assertRaises(TraceError):
                assemble_trace(events, broken, harness=harness, case=case)
        for response_id in (None, "chatcmpl_test_0", "chatcmpl_test_102x"):
            broken = deepcopy(responses)
            broken[0]["id"] = response_id
            with self.subTest(response_id=response_id), self.assertRaises(TraceError):
                assemble_trace(events, broken, harness=harness, case=case)
        responses[0]["id"] = responses[1]["id"]
        with self.assertRaisesRegex(TraceError, "repeat"):
            assemble_trace(events, responses, harness=harness, case=case)

    def test_runtime_identities_reject_a_different_gpu_uuid_for_each_strategy(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                case = dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=1,
                    input_tokens=2,
                    output_tokens=2,
                )
                harness, events, responses = runtime_fixture(case)
                for rank in range(4):
                    uuid = f"GPU-{UUID(int=rank)}"
                    harness["topology"][rank]["uuid"] = uuid
                    events[rank][0]["execution"]["runtime"]["gpu_uuid"] = uuid
                events[0][0]["execution"]["runtime"]["gpu_uuid"] = f"GPU-{UUID(int=99)}"
                with self.assertRaisesRegex(
                    BindingError, "rank 0 runtime identity differs"
                ):
                    assemble_trace(events, responses, harness=harness, case=case)

    def test_runtime_identities_reject_changed_evidence_and_fallback(self):
        case = dict(
            mode="dspark5", strategy="tep4", batch=1, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        for path in (
            ("execution", "runtime", "source_commit"),
            ("execution", "runtime", "source_digest"),
            ("execution", "runtime", "native_modules", "projection", "binary"),
            ("execution", "runtime", "gpu_uuid"),
            ("checkpoint", "config"),
            ("checkpoint", "index"),
            ("tokenizer", "tokenizer.json"),
            ("cache_memory", "effective_strategy"),
        ):
            broken = deepcopy(events)
            parent = broken[2][0]
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = "changed"
            with self.subTest(path=path), self.assertRaises(BindingError):
                assemble_trace(broken, responses, harness=harness, case=case)
        broken = deepcopy(events)
        broken[1].append(deepcopy(broken[1][0]))
        with self.assertRaisesRegex(BindingError, "one initial identity"):
            assemble_trace(broken, responses, harness=harness, case=case)
        case["strategy"] = "dep4"
        for rows in events:
            rows[0]["cache_memory"]["strategy"] = "dep4"
        with self.assertRaisesRegex(BindingError, "fallback"):
            assemble_trace(events, responses, harness=harness, case=case)

    def test_mixed_target_call_records_an_error_and_rejects_the_complete_trace(self):
        import torch
        from oh_my_vllm.models.deepseek_v41.trace import TargetTrace

        worker = SimpleNamespace(
            rank=0,
            strategy="tp4",
            device="cpu",
            routing_trace_enabled=True,
            model=SimpleNamespace(layers=[None]),
            target_trace_frontiers={101: 3, 102: 3},
            trace=SimpleNamespace(record=Mock()),
        )
        collector = TargetTrace.for_requests(worker, [(101, 3), (102, 4)])
        routes = collector.routes(1)
        routes.record(0, torch.arange(6, dtype=torch.int32).expand(2, -1))
        collector.append(routes, 3, 1, True)
        worker.trace.record.assert_called_once()
        operation = worker.trace.record.call_args.args[0]
        error = worker.trace.record.call_args.kwargs
        self.assertEqual(operation, "accounting_error")
        self.assertEqual(error["call"], 1)
        self.assertIn("mixed", error["reason"])
        self.assertEqual(worker.target_trace_frontiers, {101: 3, 102: 3})
        for start, frontier, verification in ((0, 0, False), (2, 2, True)):
            with self.subTest(start=start, verification=verification):
                worker.trace.record.reset_mock()
                worker.target_trace_frontiers = {101: frontier}
                collector = TargetTrace.for_requests(worker, [(101, 3)])
                routes = collector.routes(6)
                routes.record(0, torch.arange(6, dtype=torch.int32).expand(6, -1))
                collector.append(routes, start, 6, True, verification=verification)
                collector.finish(accepted=2, proposed=5)
                worker.trace.record.assert_called_once()
                self.assertEqual(
                    worker.trace.record.call_args.args[0], "accounting_error"
                )
                self.assertIn(
                    "prompt boundary", worker.trace.record.call_args.kwargs["reason"]
                )
                self.assertEqual(worker.target_trace_frontiers, {101: frontier})

        case = dict(
            mode="ordinary", strategy="tp4", batch=2, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        events[0].append(dict(operation=operation, **error))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_runtime_events(path, events)
            loaded = read_rank_events(path)
            self.assertEqual(loaded[0][-1]["operation"], "accounting_error")
            with self.assertRaisesRegex(TraceError, "accounting_error.*call 1"):
                assemble_trace(loaded, responses, harness=harness, case=case)

    def test_call_selection_keeps_complete_calls_and_rejects_cross_case_work(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=1, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        for rows in events:
            other = deepcopy(rows[1])
            other["call"] = 90
            other["request_rows"][0]["id"] = 999
            rows.append(other)
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual({row["call"] for row in trace["calls"]}, {0, 1})
        for rank in (0, 3):
            broken = deepcopy(events)
            foreign = deepcopy(broken[rank][1]["request_rows"][0])
            foreign["id"] = 999
            broken[rank][1]["request_rows"].append(foreign)
            with self.subTest(rank=rank), self.assertRaisesRegex(TraceError, "mixes"):
                assemble_trace(broken, responses, harness=harness, case=case)
        for change in ("missing_rank", "duplicate", "missing_layer", "gap"):
            broken = deepcopy(events)
            if change == "missing_rank":
                broken[3].pop(1)
            elif change == "duplicate":
                broken[1].append(deepcopy(broken[1][1]))
            elif change == "missing_layer":
                broken[0][1]["expert_assignments"].pop()
            else:
                broken[0][2]["request_rows"][0]["start"] += 1
            with self.subTest(change=change), self.assertRaises(TraceError):
                assemble_trace(broken, responses, harness=harness, case=case)

    def test_dep4_keeps_empty_ranks_and_observed_owner(self):
        case = dict(
            mode="dspark5", strategy="dep4", batch=1, input_tokens=2, output_tokens=4
        )
        harness, events, responses = runtime_fixture(case)
        # Move the actual owner away from both ordinal and request-ID modulo four.
        for left, right in zip(events[0][1:], events[2][1:], strict=True):
            left["request_rows"], right["request_rows"] = (
                right["request_rows"],
                left["request_rows"],
            )
            left["expert_assignments"], right["expert_assignments"] = (
                right["expert_assignments"],
                left["expert_assignments"],
            )
            left["dispatch_rows"], right["dispatch_rows"] = (
                right["dispatch_rows"],
                left["dispatch_rows"],
            )
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(trace["manifest"]["requests"][0]["owner_rank"], 2)
        self.assertEqual(len(trace["calls"]), 8)
        self.assertEqual(trace["binding"]["accepted"], 2)
        self.assertTrue(
            all(not row["request_rows"] for row in trace["calls"] if row["rank"] != 2)
        )
        self.assertTrue(
            any(row["touched_experts"][0] for row in trace["calls"] if row["rank"] == 0)
        )
        broken = deepcopy(events)
        broken[1][2]["request_rows"] = broken[2][2]["request_rows"]
        broken[2][2]["request_rows"] = []
        with self.assertRaisesRegex(TraceError, "owner changed"):
            assemble_trace(broken, responses, harness=harness, case=case)
        broken = deepcopy(events)
        broken[0][1]["request_rows"] = deepcopy(broken[2][1]["request_rows"])
        with self.assertRaisesRegex(TraceError, "owner changed"):
            assemble_trace(broken, responses, harness=harness, case=case)

    def test_reader_rejects_missing_rank_and_unflushed_records(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=1, input_tokens=2, output_tokens=1
        )
        _, events, _ = runtime_fixture(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.jsonl"
            write_runtime_events(path, events)
            self.assertEqual(read_rank_events(path), events)
            rank_paths(path)[3].unlink()
            with self.assertRaisesRegex(TraceError, "rank 3"):
                read_rank_events(path)
            rank_paths(path)[3].write_text('{"operation":')
            with self.assertRaisesRegex(TraceError, "rank 3"):
                read_rank_events(path)

    def test_verify_artifact_identity_rejects_changed_measurement_inputs(self):
        with (
            patch(
                "development.serving.deepseek_benchmark.artifact_identity",
                return_value={"checkpoint": "changed"},
            ),
            self.assertRaisesRegex(RuntimeError, "changed"),
        ):
            verify_artifact_identity(
                object(),
                object(),
                (),
                {"checkpoint": "expected"},
                workload={},
            )


if __name__ == "__main__":
    unittest.main()
