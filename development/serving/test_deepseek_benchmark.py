import hashlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .deepseek_agentx import service_max_num_seqs
from .deepseek_benchmark import (
    Service,
    artifact_identity,
    effective_cache_configuration,
    effective_service_configuration,
    native_identity,
    request_body,
    verify_artifact_identity,
)


class DeepSeekBenchmark(unittest.TestCase):
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
                patch(
                    "development.serving.deepseek_benchmark.native_identity",
                    return_value={},
                ),
                patch(
                    "development.serving.deepseek_benchmark.checkpoint_identity",
                    return_value={},
                ),
                patch(
                    "development.serving.deepseek_benchmark.topology_identity",
                    return_value=[],
                ),
            ):
                expected = artifact_identity(arguments, tokenizer, (), workload={})
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
