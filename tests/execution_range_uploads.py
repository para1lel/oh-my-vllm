import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


class ExecutionRangeUploads(unittest.TestCase):
    def test_range_upload_bounds_use_tensor_numel(self):
        from oh_my_vllm.operators.runtime.uploads import RangeUpload, UploadPool

        device = self._range_upload_device()
        device.upload_pool = UploadPool(device)
        destination = torch.full((8,), -1, dtype=torch.int32)
        upload = RangeUpload(device, destination, 32)
        from oh_my_vllm.operators.runtime import torch_adapter

        def submit(destination, source):
            torch.from_dlpack(destination).copy_(torch.from_dlpack(source))

        try:
            with (
                patch.object(
                    torch_adapter,
                    "_device_module",
                    return_value=SimpleNamespace(upload_bytes=submit),
                ),
                patch.object(
                    torch_adapter, "native_tensor", wraps=torch_adapter.native_tensor
                ) as bind,
                patch.object(torch_adapter, "current_stream", return_value=7),
            ):
                upload.copy(torch.arange(6, dtype=torch.int32).reshape(2, 3))
                torch.testing.assert_close(
                    destination,
                    torch.tensor([0, 1, 2, 3, 4, 5, -1, -1], dtype=torch.int32),
                )
                upload.copy(torch.full((2, 3), 9, dtype=torch.int32))
                self.assertEqual(bind.call_count, 2)
                upload.copy(torch.tensor([20, 21], dtype=torch.int32), offset=6)
                torch.testing.assert_close(
                    destination,
                    torch.tensor([9, 9, 9, 9, 9, 9, 20, 21], dtype=torch.int32),
                )
                self.assertEqual(bind.call_count, 4)
                upload.copy(torch.tensor([30, 31], dtype=torch.int32), offset=1)
                torch.testing.assert_close(
                    destination,
                    torch.tensor([9, 30, 31, 9, 9, 9, 20, 21], dtype=torch.int32),
                )
                self.assertEqual(bind.call_count, 6)
                replacement = torch.full((8,), -2, dtype=torch.int32)
                destination.set_(replacement)
                upload.copy(torch.tensor([40, 41], dtype=torch.int32), offset=1)
                torch.testing.assert_close(
                    replacement,
                    torch.tensor([-2, 40, 41, -2, -2, -2, -2, -2], dtype=torch.int32),
                )
                self.assertEqual(bind.call_count, 8)
                upload.copy(torch.empty((0,), dtype=torch.int32), offset=8)
                self.assertEqual(bind.call_count, 8)
                for source, offset in (
                    (torch.ones(3, dtype=torch.int32), 6),
                    (torch.ones(9, dtype=torch.int32), 0),
                    (torch.ones(2, dtype=torch.int64), 0),
                    (torch.ones(2, dtype=torch.int32), -1),
                    (torch.ones(2, dtype=torch.int32), True),
                    (torch.ones(4, dtype=torch.int32)[::2], 0),
                ):
                    with self.assertRaisesRegex(ValueError, "prepared bounds or dtype"):
                        upload.copy(source, offset=offset)
                self.assertEqual(bind.call_count, 8)
        finally:
            upload.close()
            device.upload_pool.close()

    def test_uploads_reject_numpy_runtime_buffers(self):
        from oh_my_vllm.operators.runtime.uploads import Uploads

        device = SimpleNamespace()
        uploads = Uploads(device)
        with self.assertRaisesRegex(ValueError, "CPU Torch"):
            uploads.copy(torch.empty(1), np.zeros(1, np.int32))

    def _range_upload_device(self):
        allocations = {}

        def allocate(pointer, size, flags):
            storage = ctypes.create_string_buffer(size)
            allocations[ctypes.addressof(storage)] = storage
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(
                storage
            )
            return 0

        def free(pointer):
            allocations.pop(pointer.value, None)
            return 0

        def capture(stream, status):
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

        device = SimpleNamespace(
            runtime=SimpleNamespace(
                cudaHostAlloc=allocate,
                cudaFreeHost=free,
                cudaEventCreateWithFlags=lambda *_: 0,
                cudaEventDestroy=lambda *_: 0,
                cudaEventRecord=lambda *_: 0,
                cudaEventSynchronize=lambda *_: 0,
                cudaStreamIsCapturing=capture,
            ),
            checked=lambda status: self.assertEqual(status, 0),
            wait_pinned_safe=lambda: None,
            synchronize=lambda: None,
            view=lambda value, shape, offset: value.reshape(-1)[
                offset : offset + int(np.prod(shape))
            ].reshape(shape),
        )
        return device
