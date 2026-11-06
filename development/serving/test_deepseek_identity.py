import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import deepseek_accounting as accounting
from . import deepseek_accounting_floor as accounting_floor
from . import deepseek_accounting_packed as accounting_packed
from . import deepseek_accounting_roofline as accounting_roofline
from .deepseek_benchmark import (
    artifact_identity,
    checkpoint_descriptor_identity,
    native_identity,
    verify_artifact_identity,
)
from .deepseek_benchmark_reference import runtime_reference


class DeepSeekIdentity(unittest.TestCase):
    def test_reference_metadata_distinguishes_dep4_models_from_diagnostics(self):
        calculators = (
            (accounting_floor.VERSION, "direct_protocol_sent_bytes", False),
            (accounting_packed.VERSION, "sent_bytes", True),
            (accounting_roofline.VERSION, "wire_sent_bytes", True),
        )
        for strategy in ("dep4", "tp4", "tep4"):
            trace = accounting.ideal_trace(
                dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=3,
                    input_tokens=20,
                    output_tokens=2,
                )
            )
            for version, byte_field, modeled in calculators:
                with self.subTest(version=version, strategy=strategy):
                    reference, metadata = runtime_reference(
                        trace,
                        strategy=strategy,
                        expected_binding=trace["binding"],
                        version=version,
                    )
                    if strategy == "dep4":
                        self.assertGreater(
                            sum(getattr(rank, byte_field) for rank in reference.ranks),
                            0,
                        )
                    self.assertIs(
                        metadata["dep4_communication_modeled"],
                        modeled if strategy == "dep4" else False,
                    )
                    self.assertIs(metadata["dep4_communication_measured"], False)

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
                "development.serving.deepseek_benchmark_identity.cache_directory",
                return_value=root,
            ):
                identity = native_identity()
            self.assertEqual(identity["paged"], metadata)

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
