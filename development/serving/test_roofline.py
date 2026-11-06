import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import benchmark
from .benchmark import CONFIGURATIONS, efficiency, select_configurations
from .roofline import lower_bound
from .roofline_test_support import _invoke_benchmark as _invoke_benchmark
from .roofline_test_support import _run_http_fixture as _run_http_fixture
from .roofline_test_support import _verification_trace as _verification_trace
from .roofline_test_support import (
    _verify_checkpoint_identity as _verify_checkpoint_identity,
)
from .roofline_test_support import (
    _verify_tokenizer_identity as _verify_tokenizer_identity,
)
from .roofline_test_support import _write_benchmark_inputs as _write_benchmark_inputs
from .roofline_test_support import prefill_trace as prefill_trace
from .roofline_test_support import request_manifest as request_manifest
from .roofline_test_support import tokenizer_fixture as tokenizer_fixture


def _assert_benchmark_results(self, root, calls, services):
    measured = [case for case in calls if case[2] != 1024]
    self.assertEqual(measured[:24], CONFIGURATIONS)
    self.assertEqual(measured[24:], CONFIGURATIONS[:2])
    for mode, batch, length, output in CONFIGURATIONS:
        result = json.loads(
            (root / "complete" / f"{mode}-{batch}-{length}-{output}.json").read_text()
        )
        self.assertEqual(result["output_tokens"], output)
    self.assertEqual(
        json.loads((root / "complete/summary.json").read_text()),
        dict(configurations=24),
    )
    self.assertEqual(
        json.loads((root / "partial/summary.json").read_text()),
        dict(configurations=2),
    )
    for instance in services:
        instance.ready.assert_called_once()
        instance.close.assert_called_once()
        self.assertEqual(instance.verify_identity.call_count, 2)


class BenchmarkMeasurements(unittest.TestCase):
    def test_service_close_allows_the_server_shutdown_window(self):
        service = benchmark.Service.__new__(benchmark.Service)
        service.process = Mock(pid=123)
        service.process.poll.return_value = None
        service.log = Mock()
        service.ipc = Mock()
        service.ipc.exists.return_value = False

        def complete_shutdown(timeout=None):
            if timeout is not None and timeout <= 35:
                raise benchmark.subprocess.TimeoutExpired("service", timeout)
            return 0

        service.process.wait.side_effect = complete_shutdown
        with patch.object(benchmark.os, "killpg") as kill:
            service.close()
        kill.assert_called_once_with(123, signal.SIGTERM)
        service.log.close.assert_called_once()

    def test_case_selection_and_timing(self):
        self.assertEqual(len(CONFIGURATIONS), 24)
        self.assertEqual(select_configurations(None, (64, 8192)), CONFIGURATIONS)
        self.assertEqual(select_configurations(None, (8192,)), CONFIGURATIONS[1::2])
        self.assertEqual(
            select_configurations(["mtp4:4:131072"], (64, 8192)),
            [("mtp4", 4, 131072, 64), ("mtp4", 4, 131072, 8192)],
        )
        self.assertEqual(
            select_configurations(["ordinary:1:32768:64"], (64, 8192)),
            [("ordinary", 1, 32768, 64)],
        )
        self.assertEqual(efficiency(100, 1), 0.01)
        self.assertEqual(efficiency(1, 1.06), 1.06)
        for times in ((0, 1), (-1, 1), (1, float("nan")), (float("inf"), 1)):
            with self.assertRaises(ValueError):
                efficiency(*times)
        for selected, outputs in (
            (None, (64, 64)),
            (None, ()),
            (None, (0,)),
            (["ordinary:1:32768"] * 2, (64, 8192)),
            (["ordinary:1:32768:8192", "mtp4:4:131072"], (64,)),
            (["ordinary:3:32768"], (64, 8192)),
        ):
            with self.assertRaises(ValueError):
                select_configurations(selected, outputs)

    def test_command_reports_selected_cases_and_rejects_changed_artifacts(self):
        calls, services = [], []
        mutation = None

        def service(arguments, mode):
            def run(tokenizer, batch, length, output):
                calls.append((mode, batch, length, output))
                return dict(
                    mode=mode,
                    batch=batch,
                    input_tokens=length,
                    output_tokens=output,
                    seconds=100.0,
                    efficiency=0.01,
                )

            def close():
                if mutation is not None:
                    mutation.write_bytes(b"changed CPU mock")

            instance = Mock(run=Mock(side_effect=run), close=Mock(side_effect=close))
            services.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, vendor_files, shard = _write_benchmark_inputs(root)

            self.assertEqual(
                _invoke_benchmark(root, binary, service, root / "complete"), 0
            )
            self.assertEqual(
                _invoke_benchmark(
                    root,
                    binary,
                    service,
                    root / "partial",
                    "--case",
                    "ordinary:1:32768",
                ),
                0,
            )
            _assert_benchmark_results(self, root, calls, services)
            for label, target in (
                ("binary", binary),
                ("native", root / "ffn.so"),
                ("vision-vendor", vendor_files["vision"][0]),
                ("vision-header", vendor_files["vision"][1]),
                ("checkpoint", shard),
                ("tokenizer", root / "tokenizer.json"),
                ("generation", root / "generation_config.json"),
                ("template", root / "chat_template.jinja"),
            ):
                original = target.read_bytes()
                mutation = target
                with self.subTest(artifact=label), self.assertRaises(RuntimeError):
                    _invoke_benchmark(root, binary, service, root / f"changed-{label}")
                target.write_bytes(original)
                mutation = None
                self.assertFalse((root / f"changed-{label}/summary.json").exists())
            with (
                patch.object(
                    benchmark, "cache_directory", return_value=root / "missing"
                ),
                self.assertRaises(FileNotFoundError),
            ):
                benchmark.native_identity()

    def test_http_measurement_checks_trace_identity_lengths_and_request_ids(self):
        events = [
            dict(operation="identity", config=dict(num_speculative_tokens=0)),
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=1, start=0, count=16, prompt=16)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=1, accepted=0, proposed=0, position=16)],
            ),
        ]
        response = dict(
            id="chatcmpl_744872_6ad586dc_1",
            choices=[dict(message=dict(content="x"), finish_reason="length")],
            usage=dict(
                prompt_tokens=16,
                completion_tokens=1,
                prompt_tokens_details=dict(cached_tokens=0),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            service = object.__new__(benchmark.Service)
            service.arguments = SimpleNamespace(
                artifact_id="test-artifact", prompt_seed="shared-input"
            )
            service.mode, service.port = "ordinary", 0
            service.sampling_defaults = dict(
                presence_penalty=0.5, repetition_guard=True
            )
            service.trace = Path(directory) / "trace.jsonl"

            result = _run_http_fixture(self, service, events, response)
            self.assertEqual(result["seconds"], 2.0)
            self.assertEqual(
                result["lower_bound"], lower_bound(events, result["manifest"])
            )
            self.assertEqual(
                result["manifest_id"], benchmark.json_digest(result["manifest"])
            )
            self.assertEqual(result["manifest"]["artifact_id"], "test-artifact")
            self.assertEqual(result["manifest"]["sampling"]["temperature"], 1)
            self.assertEqual(result["manifest"]["sampling"]["presence_penalty"], 0.5)
            self.assertTrue(result["manifest"]["sampling"]["repetition_guard"])
            self.assertEqual(
                result["efficiency"], result["lower_bound"]["seconds"] / 2.0
            )
            with self.assertRaisesRegex(ValueError, "distinct request IDs"):
                _run_http_fixture(self, service, events, response, batch=2)
            events[0]["config"]["num_speculative_tokens"] = 4
            with self.assertRaisesRegex(ValueError, "service mode"):
                _run_http_fixture(self, service, events, response)
            events[0]["config"]["num_speculative_tokens"] = 0
            events[1]["requests"][0]["prompt"] = 17
            with self.assertRaisesRegex(ValueError, "prompt length"):
                _run_http_fixture(self, service, events, response)
            events[1]["requests"][0]["prompt"] = 16
            response["usage"]["completion_tokens"] = 2
            with self.assertRaisesRegex(ValueError, "token count"):
                _run_http_fixture(self, service, events, response)
            checkpoint, identity, expected = _verify_checkpoint_identity(self, events)

            _verify_tokenizer_identity(
                self, directory, events, identity, checkpoint, expected
            )
