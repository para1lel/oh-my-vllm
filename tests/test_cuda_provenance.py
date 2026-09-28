"""The build identity must describe the loaded module, not a newer cache entry."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

BACKEND = (
    Path(__file__).resolve().parents[1]
    / "python/oh_my_vllm/kernels/cuda_backend/__init__.py"
)


def isolated_backend():
    """Give every test a fresh compiled() cache without touching production state."""
    spec = importlib.util.spec_from_file_location("isolated_cuda_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProvenanceTest(unittest.TestCase):
    def build_cached_fixture(self, root):
        backend = isolated_backend()
        library = root / "oh_my_vllm_cuda_fixture" / "loaded.so"
        library.parent.mkdir(exist_ok=True)
        library.write_bytes(b"trusted compiled module")
        with (
            patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": str(root)}),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("tvm_ffi.cpp.build_inline", return_value=str(library)),
            patch("tvm_ffi.load_module", return_value=object()),
            patch.object(
                backend,
                "_nvcc_identity",
                return_value=(Path("/cuda/bin/nvcc"), "V12.8.1"),
            ),
            redirect_stderr(io.StringIO()),
        ):
            backend.compiled()
        manifest = library.parent / "oh_my_vllm_cuda.provenance.json"
        self.assertTrue(manifest.is_file())
        return library, manifest

    def load_offline(self, root, *, should_load, compiler_path=None):
        backend = isolated_backend()
        with (
            patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": str(root)}),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            patch("tvm_ffi.cpp.build_inline") as build,
            patch("tvm_ffi.load_module", return_value=object()) as load,
            patch.object(
                backend,
                "_nvcc_identity",
                return_value=(compiler_path, "nvcc-unavailable: missing"),
            ),
            redirect_stderr(io.StringIO()),
        ):
            if should_load:
                backend.compiled()
            else:
                with self.assertRaises(RuntimeError):
                    backend.compiled()
        build.assert_not_called()
        return backend, load

    def test_compiled_records_exact_library_and_actual_compiler(self):
        backend = isolated_backend()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "oh_my_vllm_cuda_fixture"
            cache.mkdir()
            loaded = cache / "loaded.so"
            loaded.write_bytes(b"loaded CUDA module")
            newer = Path(directory) / "newer.so"
            newer.write_bytes(b"not loaded")
            expected = hashlib.sha256(loaded.read_bytes()).hexdigest()
            captured = {}

            def build(*args, **kwargs):
                captured.update(kwargs)
                return str(loaded)

            output = io.StringIO()
            with (
                patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": directory}),
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("tvm_ffi.cpp.build_inline", side_effect=build),
                patch("tvm_ffi.load_module", return_value=object()),
                patch.object(
                    backend,
                    "_nvcc_identity",
                    return_value=(Path("/cuda/bin/nvcc"), "V12.8.1"),
                ),
                redirect_stderr(output),
            ):
                backend.compiled()
            reported = backend.provenance(require_loaded=True)
            self.assertEqual(reported["so_path"], str(loaded))
            self.assertEqual(reported["so_sha256"], expected)
            self.assertEqual(reported["nvcc_path"], "/cuda/bin/nvcc")
            self.assertEqual(reported["nvcc_version"], "V12.8.1")
            self.assertTrue(
                any(
                    flag.startswith("-DOH_MY_VLLM_NVCC_ID_")
                    for flag in captured["extra_cuda_cflags"]
                )
            )
            marker = output.getvalue().strip().removeprefix("CUDA_BUILD_PROVENANCE ")
            self.assertEqual(json.loads(marker), reported)

    def test_cuda_home_compiler_wins_over_path(self):
        backend = isolated_backend()
        with (
            patch("tvm_ffi.cpp.extension._find_cuda_home", return_value="/cuda/home"),
            patch(
                "subprocess.check_output", return_value="nvcc release 12.8, V12.8.1\n"
            ) as check,
        ):
            path, version = backend._nvcc_identity()
        self.assertEqual(path, Path("/cuda/home/bin/nvcc"))
        self.assertEqual(version, "nvcc release 12.8, V12.8.1")
        check.assert_called_once_with(["/cuda/home/bin/nvcc", "--version"], text=True)

    def test_compiler_version_changes_build_cache_key(self):
        flags = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "oh_my_vllm_cuda_fixture"
            cache.mkdir()
            loaded = cache / "loaded.so"
            loaded.write_bytes(b"module")

            def build(*args, **kwargs):
                flags.append(kwargs["extra_cuda_cflags"])
                return str(loaded)

            for version in ("V12.8.1", "V12.9.0"):
                backend = isolated_backend()
                with (
                    patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": directory}),
                    patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                    patch("tvm_ffi.cpp.build_inline", side_effect=build),
                    patch("tvm_ffi.load_module", return_value=object()),
                    patch.object(
                        backend,
                        "_nvcc_identity",
                        return_value=(Path("/cuda/bin/nvcc"), version),
                    ),
                    redirect_stderr(io.StringIO()),
                ):
                    backend.compiled()
        self.assertNotEqual(flags[0], flags[1])

    def test_build_input_digest_covers_source_flags_and_toolkit(self):
        backend = isolated_backend()
        with patch.dict("os.environ"):
            original = backend._build_input_digest("source A", "torch A")
            self.assertNotEqual(
                original, backend._build_input_digest("source B", "torch A")
            )
            self.assertNotEqual(
                original, backend._build_input_digest("source A", "torch B")
            )
            with patch.object(backend, "_BASE_CUDA_FLAGS", ("-O0",)):
                self.assertNotEqual(
                    original, backend._build_input_digest("source A", "torch A")
                )
            with patch.object(backend, "_effective_cuda_target", return_value="sm_90"):
                self.assertNotEqual(
                    original, backend._build_input_digest("source A", "torch A")
                )
            with patch.dict("os.environ", {"CXX": "different-cxx"}):
                self.assertNotEqual(
                    original, backend._build_input_digest("source A", "torch A")
                )
            with patch("importlib.metadata.version", return_value="different-tvm-ffi"):
                self.assertNotEqual(
                    original, backend._build_input_digest("source A", "torch A")
                )

    def test_cuda_target_is_fixed_even_on_multigpu_host(self):
        backend = isolated_backend()
        with patch.dict("os.environ", {"TVM_FFI_CUDA_ARCH_LIST": "10.0a"}):
            self.assertEqual(
                backend._effective_cuda_target(),
                "-gencode=arch=compute_100a,code=sm_100a",
            )
        with (
            patch.dict("os.environ", {"TVM_FFI_CUDA_ARCH_LIST": "9.0"}),
            self.assertRaisesRegex(RuntimeError, "requires.*10.0a"),
        ):
            backend._effective_cuda_target()
        with patch.dict("os.environ") as environment:
            environment.pop("TVM_FFI_CUDA_ARCH_LIST", None)
            with self.assertRaisesRegex(RuntimeError, "requires.*10.0a"):
                backend._effective_cuda_target()
            self.assertNotIn("TVM_FFI_CUDA_ARCH_LIST", environment)

    def test_unset_cache_root_matches_tvm_ffi_default(self):
        backend = isolated_backend()
        with patch.dict("os.environ") as environment:
            environment.pop("TVM_FFI_CACHE_DIR", None)
            self.assertEqual(
                backend._cache_root(), Path("~/.cache/tvm-ffi").expanduser().resolve()
            )

    def test_offline_process_reuses_only_matching_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, manifest = self.build_cached_fixture(root)
            record = json.loads(manifest.read_text())
            self.assertEqual(
                record["so_sha256"], hashlib.sha256(library.read_bytes()).hexdigest()
            )
            backend, load = self.load_offline(root, should_load=True)
            load.assert_called_once_with(library)
            identity = backend.provenance(require_loaded=True, require_compiler=True)
            self.assertEqual(identity["compiler_identity_source"], "manifest")
            self.assertEqual(identity["so_sha256"], record["so_sha256"])
            self.assertEqual(identity["nvcc_version"], "V12.8.1")

    def test_offline_process_rejects_changed_cuda_home(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_cached_fixture(root)
            _, load = self.load_offline(
                root,
                should_load=False,
                compiler_path=Path("/different-cuda/bin/nvcc"),
            )
            load.assert_not_called()

    def test_known_cuda_home_selects_unique_matching_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, manifest = self.build_cached_fixture(root)
            second = root / "oh_my_vllm_cuda_other"
            second.mkdir()
            second_library = second / "loaded.so"
            second_library.write_bytes(library.read_bytes())
            record = json.loads(manifest.read_text())
            record["so_path"] = str(second_library)
            record["nvcc_path"] = "/other-cuda/bin/nvcc"
            (second / manifest.name).write_text(json.dumps(record))
            backend, load = self.load_offline(
                root, should_load=True, compiler_path=Path("/cuda/bin/nvcc")
            )
            load.assert_called_once_with(library)
            self.assertEqual(
                backend.provenance(require_loaded=True)["nvcc_path"],
                "/cuda/bin/nvcc",
            )
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()

    def test_offline_process_rejects_stale_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self.build_cached_fixture(root)
            record = json.loads(manifest.read_text())
            record["build_input_sha256"] = "0" * 64
            manifest.write_text(json.dumps(record))
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()
            # Restore the genuine manifest, then change the library bytes.
            library, manifest = self.build_cached_fixture(root)
            library.write_bytes(b"tampered module")
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()

    def test_offline_process_rejects_missing_and_outside_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, manifest = self.build_cached_fixture(root)
            manifest.unlink()
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()
            library, manifest = self.build_cached_fixture(root)
            with tempfile.TemporaryDirectory() as other:
                record = json.loads(manifest.read_text())
                outside = Path(other) / "outside.so"
                outside.write_bytes(library.read_bytes())
                record["so_path"] = str(outside)
                manifest.write_text(json.dumps(record))
                _, load = self.load_offline(root, should_load=False)
                load.assert_not_called()

    def test_offline_process_rejects_ambiguous_and_symlinked_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, manifest = self.build_cached_fixture(root)
            second = root / "oh_my_vllm_cuda_second"
            second.mkdir()
            second_library = second / "loaded.so"
            second_library.write_bytes(library.read_bytes())
            record = json.loads(manifest.read_text())
            record["so_path"] = str(second_library)
            (second / manifest.name).write_text(json.dumps(record))
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()
            (second / manifest.name).unlink()
            saved = manifest.with_name("saved.json")
            manifest.replace(saved)
            manifest.symlink_to(saved)
            _, load = self.load_offline(root, should_load=False)
            load.assert_not_called()

    def test_offline_process_detects_library_change_during_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, _ = self.build_cached_fixture(root)
            backend = isolated_backend()

            def mutate(path):
                self.assertEqual(path, library)
                library.write_bytes(b"changed during load")
                return object()

            with (
                patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": str(root)}),
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("tvm_ffi.cpp.build_inline") as build,
                patch("tvm_ffi.load_module", side_effect=mutate),
                patch.object(
                    backend,
                    "_nvcc_identity",
                    return_value=(None, "nvcc-unavailable: missing"),
                ),
                self.assertRaisesRegex(RuntimeError, "changed while loading"),
            ):
                backend.compiled()
            build.assert_not_called()

    def test_nvcc_disappearing_after_build_does_not_break_runtime(self):
        backend = isolated_backend()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "oh_my_vllm_cuda_fixture"
            cache.mkdir()
            loaded = cache / "cached.so"
            loaded.write_bytes(b"cached module")
            with (
                patch.dict("os.environ", {"TVM_FFI_CACHE_DIR": directory}),
                patch("torch.cuda.get_device_capability", return_value=(10, 0)),
                patch("tvm_ffi.cpp.build_inline", return_value=str(loaded)),
                patch("tvm_ffi.load_module", return_value=object()),
                patch.object(
                    backend,
                    "_nvcc_identity",
                    side_effect=[
                        (Path("/cuda/bin/nvcc"), "V12.8.1"),
                        (Path("/cuda/bin/nvcc"), "nvcc-unavailable: missing"),
                    ],
                ),
                redirect_stderr(io.StringIO()),
            ):
                backend.compiled()
            self.assertEqual(
                backend.provenance(require_loaded=True)["nvcc_version"],
                "nvcc-changed-during-build",
            )
            with self.assertRaisesRegex(RuntimeError, "compiler identity"):
                backend.provenance(require_loaded=True, require_compiler=True)

    def test_unloaded_process_never_claims_a_cached_shared_library(self):
        backend = isolated_backend()
        self.assertIsNone(backend.provenance()["so_sha256"])
        with self.assertRaisesRegex(RuntimeError, "loaded CUDA module"):
            backend.provenance(require_loaded=True)


if __name__ == "__main__":
    unittest.main()
