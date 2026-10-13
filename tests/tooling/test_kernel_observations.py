"""Profiler evidence cannot turn missing counters into measured observations."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from development.kernels.observations import (
    _profile_run,
    profile_provenance,
    read_nsight,
    validate_profile_loads,
    validate_profile_worker,
)


class ObservationTest(unittest.TestCase):
    def read(self, values, required=("a", "b")):
        header = (
            '"ID","Process ID","Kernel Name",'
            '"Metric Name","Metric Unit","Metric Value"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            path.write_text(header + "\n".join(values))
            return read_nsight(path, required)

    def test_each_launch_requires_all_requested_counters(self):
        complete = ['"0","1","kernel","a","%","1.5"', '"0","1","kernel","b","","2"']
        self.assertEqual(self.read(complete)["kind"], "measured")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.read([*complete, '"1","1","kernel","a","%","3"'])

    def test_unavailable_and_nonfinite_counters_are_rejected(self):
        for value in ("N/A", "", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read([f'"0","1","kernel","a","%","{value}"'], ("a",))

    def test_cancellation_reaps_the_owned_process_group(self):
        process = MagicMock(pid=12345)
        process.communicate.side_effect = KeyboardInterrupt
        with (
            patch("subprocess.Popen", return_value=process) as launch,
            patch("os.killpg") as kill,
            self.assertRaises(KeyboardInterrupt),
        ):
            _profile_run(["ncu"])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 12345])
        process.wait.assert_called()

    def test_profile_load_identity_comes_from_its_own_worker(self):
        import json

        gemm = {"source_sha256": "source", "project_headers": {"accum.cuh": "header"}}
        marker = "CUDA_GEMM_BUILD_PROVENANCE " + json.dumps(gemm)
        self.assertEqual(profile_provenance("NCU banner\n" + marker), {"gemm": gemm})
        self.assertEqual(profile_provenance("NCU banner", marker), {"gemm": gemm})
        self.assertEqual(profile_provenance("no owned provider loaded"), {})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            profile_provenance(marker + "\n" + marker)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            profile_provenance(marker, marker)
        with self.assertRaises(ValueError):
            profile_provenance("CUDA_GEMM_BUILD_PROVENANCE []")
        with self.assertRaises(ValueError):
            profile_provenance("CUDA_GEMM_BUILD_PROVENANCE invalid")

    def test_all_counter_launches_match_the_requested_worker(self):
        worker = {"pid": 784, "case": "gated", "backend": "cuda"}
        metrics = [{"Process ID": "784"}, {"Process ID": "784"}]
        validate_profile_worker(metrics, worker, "gated", "cuda")
        for changed in (
            {},
            {**worker, "pid": 785},
            {**worker, "pid": True},
            {**worker, "case": "another"},
            {**worker, "backend": "tilelang"},
        ):
            with self.subTest(worker=changed), self.assertRaises(ValueError):
                validate_profile_worker(metrics, changed, "gated", "cuda")
        with self.assertRaises(ValueError):
            validate_profile_worker(
                [*metrics, {"Process ID": "785"}], worker, "gated", "cuda"
            )

    def test_gated_projection_requires_current_pointwise_and_gemm_libraries(self):
        import copy
        import hashlib
        import json

        config = {"operation": "gated_norm_fp8_linear", "tokens": 624}
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "provider.so"
            library.write_bytes(b"loaded test provider")
            record = {
                "source_sha256": "1" * 64,
                "build_input_sha256": "2" * 64,
                "so_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "so_path": str(library),
                "nvcc_path": "nvcc",
                "nvcc_version": "tested",
                "compiler": "nvcc",
                "compiler_version": "tested",
                "compiler_identity_source": "live",
                "pdl": True,
                "project_headers": {"groupwise_accum.cuh": "3" * 64},
            }
            records = {"pointwise": record, "gemm": copy.deepcopy(record)}
            record["project_headers"] = {"operators/common.cuh": "5" * 64}
            identity = {
                k: v
                for k, v in records["gemm"].items()
                if k not in {"so_path", "so_sha256", "build_input_sha256"}
            }
            records["gemm"]["build_input_sha256"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            inputs = {"project_headers": records["gemm"]["project_headers"]}
            sources = {
                "python/oh_my_vllm/kernels/cuda_backend/" + name: "1" * 64
                for name in ("kernels.cu", "groupwise_fp8.cu")
            }
            sources["python/oh_my_vllm/kernels/cuda_backend/operators/common.cuh"] = (
                "5" * 64
            )
            validate_profile_loads(records, "cuda", config, sources, inputs)
            validate_profile_loads({}, "tilelang", config, sources, inputs)
            validate_profile_loads(
                {"pointwise": record}, "cuda", {**config, "tokens": 8}, sources, inputs
            )
            for operation in (
                "recurrent",
                "convolution",
                "gdn_prefill",
                "attention",
                "dspark_attention",
                "prepare_context",
                "prepare_query",
            ):
                other = {"operation": operation, "batch": 4}
                if operation.startswith("prepare_"):
                    other["tokens"] = 624
                with self.subTest(operation=operation):
                    validate_profile_loads(
                        {"pointwise": record}, "cuda", other, sources, inputs
                    )
            for key in ("pointwise", "gemm"):
                for field in (
                    None,
                    "so_sha256",
                    "build_input_sha256",
                    "source_sha256",
                    "so_path",
                    "project_headers" if key == "gemm" else "nvcc_version",
                ):
                    changed = copy.deepcopy(records)
                    if field is None:
                        del changed[key]
                    else:
                        del changed[key][field]
                    with (
                        self.subTest(key=key, field=field),
                        self.assertRaises(ValueError),
                    ):
                        validate_profile_loads(changed, "cuda", config, sources, inputs)
            changed = copy.deepcopy(records)
            changed["gemm"]["source_sha256"] = "4" * 64
            with self.assertRaisesRegex(ValueError, "source"):
                validate_profile_loads(changed, "cuda", config, sources, inputs)
            for headers in (None, {}, {"operators/common.cuh": "6" * 64}):
                changed = copy.deepcopy(records)
                changed["pointwise"]["project_headers"] = headers
                with (
                    self.subTest(headers=headers),
                    self.assertRaisesRegex(ValueError, "CUDA operator headers differ"),
                ):
                    validate_profile_loads(changed, "cuda", config, sources, inputs)
            library.write_bytes(b"changed after load")
            with self.assertRaisesRegex(ValueError, "library bytes changed"):
                validate_profile_loads(records, "cuda", config, sources, inputs)
