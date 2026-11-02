import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.operators.runtime.device import Device


class DownloadDtypeContract(unittest.TestCase):
    def test_bf16_bits_reinterpret_as_uint16_not_numeric(self):
        device = Device.__new__(Device)
        source = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
        with patch("torch.ops.oh_my_vllm.native_download", return_value=source):
            actual = device.download(object(), torch.uint16)
        self.assertEqual(actual.dtype, torch.uint16)
        self.assertTrue(torch.equal(actual, source.view(torch.uint16)))
        self.assertFalse(torch.equal(actual, source.to(torch.uint16)))

    def test_download_rejects_a_different_element_width(self):
        device = Device.__new__(Device)
        source = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        with (
            patch("torch.ops.oh_my_vllm.native_download", return_value=source),
            self.assertRaisesRegex(ValueError, "element width"),
        ):
            device.download(object(), torch.uint8)

    def test_stream_checks_use_the_torch_current_stream(self):
        device = Device.__new__(Device)
        calls = []
        device.runtime = SimpleNamespace(
            cudaStreamSynchronize=lambda stream: calls.append(("sync", stream)) or 0,
            cudaStreamQuery=lambda stream: calls.append(("query", stream)) or 0,
        )
        device.checked = lambda status: None
        device.transfer_engines = set()
        with patch(
            "oh_my_vllm.operators.runtime.device.torch_adapter.current_stream",
            return_value=77,
        ):
            device.synchronize_compute()
            self.assertTrue(device.pinned_quiescent())
        self.assertEqual(calls, [("sync", 77), ("query", 77)])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeviceTensors(unittest.TestCase):
    def setUp(self):
        self.device = Device()

    def tearDown(self):
        self.device.close()

    def test_allocations_and_views_are_torch_tensors_with_aliasing(self):
        value = self.device.empty((4,), "float32")
        scratch = self.device.scratch((4,), "float32")
        zeros = self.device.zeros((4,), "float32")

        for result in (value, scratch, zeros):
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.device.type, "cuda")
            self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(zeros, torch.zeros(4, device="cuda"))

        value.copy_(torch.arange(4, dtype=torch.float32, device="cuda"))
        view = self.device.view(value, (2,), 1)
        self.assertIsInstance(view, torch.Tensor)
        torch.testing.assert_close(view, torch.tensor((1.0, 2.0), device="cuda"))
        view[0] = 19
        self.assertEqual(float(value[1]), 19)

    def test_upload_and_download_use_torch_tensors(self):
        source = torch.arange(4, dtype=torch.float32)
        uploaded = self.device.upload(source)

        self.assertIsInstance(uploaded, torch.Tensor)
        self.assertEqual(uploaded.device.type, "cuda")
        torch.testing.assert_close(
            self.device.download(uploaded, torch.float32), source
        )

    def test_unsigned_uploads_keep_dtype_and_bits(self):
        for dtype in (torch.uint16, torch.uint32, torch.uint64):
            with self.subTest(dtype=dtype):
                source = torch.tensor([0, 1, 17, 255], dtype=dtype)
                uploaded = self.device.upload(source)
                self.assertEqual(uploaded.dtype, dtype)
                downloaded = self.device.download(uploaded, dtype)
                self.assertEqual(downloaded.dtype, dtype)
                torch.testing.assert_close(downloaded, source, rtol=0, atol=0)

    def test_upload_rejects_non_contiguous_torch_tensors(self):
        source = torch.ones((2, 4), dtype=torch.float32)[:, ::2]

        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.device.upload(source)


if __name__ == "__main__":
    unittest.main()
