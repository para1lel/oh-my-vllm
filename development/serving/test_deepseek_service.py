import contextlib
import io
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .deepseek_agentx import service_max_num_seqs
from .deepseek_benchmark import (
    Service,
    argument_parser,
    effective_cache_configuration,
    effective_service_configuration,
    request_body,
    runtime_reference,
    workload_cache_requirements,
)
from .deepseek_runtime_trace import assemble_trace
from .test_deepseek_benchmark import runtime_fixture, write_runtime_events


class DeepSeekService(unittest.TestCase):
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
                "development.serving.deepseek_benchmark_service.subprocess.Popen"
            ) as launch:
                launch.return_value.poll.return_value = 0
                service = Service(arguments, "ordinary", "tp4")
            try:
                command = launch.call_args.args[0]
                self.assertEqual(
                    command[command.index("--request-timeout-seconds") + 1], "14400"
                )
                with patch(
                    "development.serving.deepseek_benchmark_service.http.client.HTTPConnection",
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
                "development.serving.deepseek_benchmark_service.subprocess.Popen",
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
                    "development.serving.deepseek_benchmark_service.subprocess.Popen",
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
                    "development.serving.deepseek_benchmark_service.subprocess.Popen",
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
                "development.serving.deepseek_benchmark_service.subprocess.Popen",
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

    def test_agentx_default_sequence_capacity_follows_concurrency(self):
        self.assertEqual(service_max_num_seqs(16), 16)
        self.assertEqual(service_max_num_seqs(64), 64)
        self.assertEqual(service_max_num_seqs(16, 32), 32)
        with self.assertRaises(ValueError):
            service_max_num_seqs(16, 65)

    def test_capacity_covers_resident_owner_rows_output_and_private_state(self):
        from .deepseek_matrix import HTTP_CASES

        requirements = workload_cache_requirements(HTTP_CASES)
        for strategy, local in (("tp4", 64), ("tep4", 64), ("dep4", 16)):
            with self.subTest(strategy=strategy):
                plan = requirements[strategy]
                self.assertEqual(plan["block_tokens"], 128)
                self.assertEqual(plan["maximum_total_tokens"], 808192)
                self.assertEqual(plan["resident_requests"], local)
                self.assertEqual(plan["blocks"], 1 + local * (6315 + 1))
                self.assertEqual(plan["mamba_blocks"], 1 + local * 7)
                self.assertEqual(
                    plan["placement"],
                    "request_owner" if strategy == "dep4" else "replicated",
                )
        uneven = workload_cache_requirements(
            [
                dict(
                    strategy="dep4",
                    mode="dspark5",
                    batch=5,
                    input_tokens=128,
                    output_tokens=1,
                )
            ]
        )["dep4"]
        self.assertEqual(uneven["resident_requests"], 2)
        self.assertEqual(uneven["blocks"], 7)
        self.assertEqual(uneven["mamba_blocks"], 15)

    def test_service_passes_workload_capacity_and_rejects_short_overrides(self):
        requirements = workload_cache_requirements(
            [
                dict(
                    strategy="dep4",
                    mode="dspark5",
                    batch=16,
                    input_tokens=800000,
                    output_tokens=8192,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                artifacts=root,
                model=root,
                binary=root / "service",
                cache_context=1048576,
                blocks=None,
                mamba_blocks=None,
                cache_requirements=requirements,
            )
            process = Mock()
            process.poll.return_value = 0
            with (
                patch.dict(os.environ),
                patch(
                    "development.serving.deepseek_benchmark_service.subprocess.Popen",
                    return_value=process,
                ) as launch,
            ):
                os.environ.pop("OH_MY_VLLM_NUM_GPU_BLOCKS", None)
                service = Service(arguments, "dspark5", "dep4")
                try:
                    command = launch.call_args.args[0]
                    self.assertEqual(
                        int(command[command.index("--num-gpu-blocks") + 1]),
                        1 + 4 * 6316,
                    )
                    self.assertEqual(
                        int(command[command.index("--mamba-blocks") + 1]), 128
                    )
                finally:
                    service.close()
                for name, value in (("blocks", 2048), ("mamba_blocks", 28)):
                    setattr(arguments, name, value)
                    with (
                        self.subTest(override=name),
                        self.assertRaisesRegex(
                            ValueError, "cannot cover the selected resident workload"
                        ),
                    ):
                        Service(arguments, "dspark5", "dep4")
                    setattr(arguments, name, None)
                with patch.dict(os.environ, {"OH_MY_VLLM_NUM_GPU_BLOCKS": "2048"}):
                    with self.assertRaisesRegex(ValueError, "cannot cover"):
                        effective_cache_configuration(arguments, "dep4")
                    arguments.blocks = 30000
                    self.assertEqual(
                        effective_cache_configuration(arguments, "dep4")["blocks"],
                        30000,
                    )

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

    def test_http_seconds_include_preparation_and_exclude_trace_work(self):
        case = dict(
            mode="dspark5", strategy="tp4", batch=1, input_tokens=2, output_tokens=4
        )
        harness, events, responses = runtime_fixture(case)
        service = object.__new__(Service)
        service.arguments = SimpleNamespace()
        service.port = 0
        service.request_timeout_seconds = 7200
        clock = [100.0]

        def prepare_prompt(*args):
            clock[0] += 7.0
            return []

        def prepare_body(*args):
            clock[0] += 5.0
            return request_body(*args)

        def receive_response(*args, **kwargs):
            clock[0] += 11.0
            return responses[0]

        def build_trace(*args, **kwargs):
            clock[0] += 13.0
            return assemble_trace(*args, **kwargs)

        def calculate_reference(*args, **kwargs):
            clock[0] += 17.0
            return runtime_reference(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            service.trace = Path(directory) / "execution.jsonl"
            write_runtime_events(service.trace, events)
            with (
                patch(
                    "development.serving.deepseek_benchmark_service.exact_prompt",
                    side_effect=prepare_prompt,
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.request_body",
                    side_effect=prepare_body,
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.post",
                    side_effect=receive_response,
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.assemble_trace",
                    side_effect=build_trace,
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.runtime_reference",
                    side_effect=calculate_reference,
                ),
                patch(
                    "development.serving.deepseek_benchmark_service.time.perf_counter",
                    side_effect=lambda: clock[0],
                ),
            ):
                result = service.run(object(), case, "test", identity=harness)
        self.assertEqual(result["seconds"], 23.0)
        self.assertEqual(clock[0], 153.0)
