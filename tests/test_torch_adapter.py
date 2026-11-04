import unittest
from unittest.mock import patch

import torch
import tvm_ffi
from oh_my_vllm.operators.runtime import torch_adapter


class TorchAdapterTests(unittest.TestCase):
    def test_native_tensor_keeps_a_contiguous_torch_buffer_without_copying(self):
        value = torch.arange(4, dtype=torch.float32)

        native = torch_adapter.native_tensor(value)

        self.assertIsInstance(native, tvm_ffi.Tensor)
        value[2] = 17
        self.assertEqual(float(torch.from_dlpack(native)[2]), 17)

    def test_native_tensor_rejects_non_contiguous_values(self):
        value = torch.ones((2, 4), dtype=torch.float32)[:, ::2]

        with self.assertRaisesRegex(ValueError, "contiguous"):
            torch_adapter.native_tensor(value)

    def test_native_tensor_rejects_non_torch_and_unsupported_device_values(self):
        with self.assertRaises(TypeError):
            torch_adapter.native_tensor([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "device"):
            torch_adapter.native_tensor(torch.empty(2, device="meta"))

    def test_torch_tensor_consumes_a_native_dlpack_producer(self):
        source = torch.arange(4, dtype=torch.float32)
        native = tvm_ffi.from_dlpack(source)

        value = torch_adapter.torch_tensor(native)

        self.assertIsInstance(value, torch.Tensor)
        native_from_value = torch.from_dlpack(value)
        torch.testing.assert_close(native_from_value, source)

    def test_cpu_stream_is_the_zero_stream(self):
        self.assertEqual(torch_adapter.current_stream("cpu"), 0)

    def test_stream_scope_restores_the_previous_raw_stream_after_failure(self):
        device = tvm_ffi.device("cuda:0")
        previous = tvm_ffi.get_raw_stream(device)
        with torch_adapter.use_stream("cuda:0", 7):
            self.assertEqual(tvm_ffi.get_raw_stream(device), 7)
            with (
                self.assertRaisesRegex(RuntimeError, "submit failed"),
                torch_adapter.use_stream("cuda:0", 11),
            ):
                self.assertEqual(tvm_ffi.get_raw_stream(device), 11)
                raise RuntimeError("submit failed")
            self.assertEqual(tvm_ffi.get_raw_stream(device), 7)
        self.assertEqual(tvm_ffi.get_raw_stream(device), previous)

    def test_native_shape_is_created_inside_the_adapter(self):
        shape = torch_adapter.native_shape((2, 3))

        self.assertIsInstance(shape, tvm_ffi.Shape)
        self.assertEqual(tuple(shape), (2, 3))

    def test_native_tensor_list_keeps_torch_owners(self):
        values = (torch.arange(2, dtype=torch.float32),)

        native = torch_adapter.native_tensors(values)

        self.assertEqual(len(native), 1)
        self.assertIsInstance(native[0], tvm_ffi.Tensor)

    def test_native_dtypes_and_cpu_aliases_keep_bits(self):
        expected = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.bool: "bool",
            torch.uint8: "uint8",
            torch.uint16: "uint16",
            torch.uint32: "uint32",
            torch.uint64: "uint64",
        }
        for dtype, name in expected.items():
            with self.subTest(dtype=name):
                self.assertEqual(torch_adapter._native_dtype(dtype), name)
                value = torch.tensor([0, 1, 2, 3], dtype=dtype)
                native = torch_adapter.native_tensor(value)
                alias = torch.from_dlpack(native)
                probe = 1 if dtype is torch.bool else 17
                value[2] = probe
                self.assertEqual(alias[2].item(), probe)
                self.assertEqual(alias.dtype, dtype)

    def test_torch_library_schemas_are_registered(self):
        namespace = torch.ops.oh_my_vllm
        for name in (
            "native_empty",
            "native_view",
            "native_copy",
            "native_zero",
            "native_upload",
            "native_download",
        ):
            self.assertTrue(hasattr(namespace, name), name)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaTorchAdapterTests(unittest.TestCase):
    def test_current_stream_returns_the_torch_cuda_stream(self):
        stream = torch.cuda.current_stream()
        self.assertEqual(torch_adapter.current_stream("cuda:0"), stream.cuda_stream)

    def test_upload_routes_pinned_snapshots_and_ordinary_tensors(self):
        from oh_my_vllm.operators.runtime.device import Device

        device = Device()
        uploads = device.uploads()
        stream = torch.cuda.Stream(device="cuda:0")
        other_stream = torch.cuda.Stream(device="cuda:0")
        calls = []

        def raw_stream():
            return tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))

        class SpyModule:
            def __init__(self, module):
                self.module = module

            def __getattr__(self, name):
                return getattr(self.module, name)

            def upload_bytes(self, destination, source):
                calls.append(("upload_bytes", raw_stream()))
                return self.module.upload_bytes(destination, source)

            def copy_bytes(self, destination, source):
                calls.append(("copy_bytes", raw_stream()))
                return self.module.copy_bytes(destination, source)

        source = torch.arange(32, dtype=torch.uint8) * 7
        upload_output = device.empty((32,), torch.uint8)
        range_output = device.zeros((64,), torch.uint8)
        try:
            with (
                patch.object(
                    torch_adapter,
                    "_device_module",
                    return_value=SpyModule(device.module),
                ),
                torch.cuda.stream(stream),
            ):
                snapshot = uploads.prepare(upload_output, source)
                uploads.update(snapshot.copy_, source)
                source.add_(1)
                ranged = uploads.prepare_range(range_output, capacity_bytes=32)
                ranged.copy(source[:8], offset=16)
                ranged.copy(source[:8] + 1, offset=16)
                ranged.copy(source[8:12], offset=32)
                setup_output = device.upload(source)
                with torch.cuda.stream(other_stream):
                    uploads.update(snapshot.copy_, source + 1)
            torch.cuda.synchronize()
            self.assertEqual(
                [name for name, _ in calls],
                ["upload_bytes"] * 4 + ["copy_bytes", "upload_bytes"],
            )
            self.assertEqual({value for _, value in calls[:-1]}, {stream.cuda_stream})
            self.assertEqual(calls[-1][1], other_stream.cuda_stream)
            self.assertTrue(
                torch.equal(device.download(upload_output, None), source + 1)
            )
            expected = torch.zeros(64, dtype=torch.uint8)
            expected[16:24] = source[:8] + 1
            expected[32:36] = source[8:12]
            self.assertTrue(torch.equal(device.download(range_output, None), expected))
            self.assertTrue(torch.equal(device.download(setup_output, None), source))
        finally:
            uploads.close()
            device.close()


if __name__ == "__main__":
    unittest.main()
