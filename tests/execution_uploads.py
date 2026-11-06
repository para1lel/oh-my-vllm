import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch


class ExecutionUploads(unittest.TestCase):
    def test_upload_snapshot_uses_numpy_and_preserves_torch_fallback(self):
        from oh_my_vllm.operators.runtime.uploads import UploadBuffer

        pool = Mock()
        device = SimpleNamespace(
            upload_pool=pool,
            runtime=SimpleNamespace(cudaEventCreateWithFlags=lambda *_: 0),
            checked=lambda status: self.assertEqual(status, 0),
        )
        sources = [
            torch.tensor([0, 1], dtype=dtype)
            for dtype in (
                torch.bool,
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint16,
                torch.uint32,
                torch.uint64,
                torch.float16,
                torch.float32,
                torch.float64,
                torch.complex64,
                torch.complex128,
            )
        ]
        sources.extend((torch.tensor(3), torch.empty((0, 3), dtype=torch.int32)))
        fallbacks = (
            torch.tensor([1.25, 2.5], dtype=torch.bfloat16),
            torch.tensor([3.0, 4.0], requires_grad=True),
            torch.tensor([1 + 2j, 3 + 4j]).conj(),
            torch.tensor([5, 6], dtype=torch.int32)._neg_view(),
        )
        for source in (*sources, *fallbacks):
            with self.subTest(
                dtype=source.dtype,
                shape=source.shape,
                fallback=any(source is item for item in fallbacks),
            ):
                pool.acquire.return_value = SimpleNamespace(
                    tensor=torch.empty(64, dtype=torch.uint8)
                )
                buffer = UploadBuffer(device, torch.empty_like(source), source)
                try:
                    with (
                        patch.object(buffer, "_submit") as submit,
                        patch(
                            "oh_my_vllm.operators.runtime.uploads.np.copyto",
                            wraps=np.copyto,
                        ) as copyto,
                    ):
                        buffer.copy(source, 7)
                        torch.testing.assert_close(buffer.tensor, source)
                        self.assertEqual(
                            copyto.call_count,
                            int(any(source is item for item in sources)),
                        )
                        submit.assert_called_once()
                        if source.requires_grad:
                            self.assertTrue(buffer.tensor.requires_grad)
                            buffer.copy(source.detach(), 7)
                            self.assertEqual(copyto.call_count, 0)
                        self._check_upload_rejections(buffer, source)
                finally:
                    buffer.close()

    def test_upload_keeps_torch_staging_owner_until_event_completion(self):
        import gc

        from oh_my_vllm.operators.runtime.uploads import UploadPool, Uploads

        allocations, pending, waits, lifecycle, device = self._upload_lifecycle_device(
            UploadPool
        )
        source = torch.tensor([1, 2], dtype=torch.int32)
        destination = torch.zeros(2, dtype=torch.int32)
        from oh_my_vllm.operators.runtime import torch_adapter

        submit = Mock(
            side_effect=lambda destination, source: pending.append(
                (torch.from_dlpack(destination), torch.from_dlpack(source))
            )
        )
        with (
            patch.object(
                torch_adapter,
                "_device_module",
                return_value=SimpleNamespace(upload_bytes=submit),
            ),
            patch.object(
                torch_adapter, "native_tensor", wraps=torch_adapter.native_tensor
            ) as bind,
            patch.object(
                torch_adapter, "use_stream", wraps=torch_adapter.use_stream
            ) as scope,
            patch.object(torch_adapter, "current_stream", return_value=7) as current,
            patch(
                "oh_my_vllm.operators.runtime.uploads.np.copyto", wraps=np.copyto
            ) as copyto,
        ):
            array, buffer, snapshot, uploads = self._check_upload_binding_refresh(
                Uploads,
                device,
                source,
                destination,
                current,
                bind,
                copyto,
                scope,
                waits,
                submit,
                torch_adapter,
            )

            self._check_upload_launch_failures(
                pending, submit, uploads, snapshot, source, buffer, device
            )
            del source, array
            gc.collect()
            self._check_upload_close_failures(device, uploads, buffer, lifecycle)

        torch.testing.assert_close(destination, torch.tensor([9, 9], dtype=torch.int32))
        self.assertTrue(waits)
        device.upload_pool.close()
        self.assertFalse(allocations)

    def _check_upload_rejections(self, buffer, source):
        before = buffer.tensor.detach().clone()
        for invalid in (
            np.zeros(tuple(source.shape)),
            torch.empty(source.shape, device="meta", dtype=source.dtype),
            torch.empty((3,), dtype=source.dtype),
            torch.empty(
                source.shape,
                dtype=torch.float32 if source.dtype != torch.float32 else torch.int32,
            ),
            torch.empty((4,), dtype=source.dtype)[::2],
        ):
            with self.assertRaisesRegex(ValueError, "shape or dtype changed"):
                buffer.copy(invalid, 7)
        torch.testing.assert_close(buffer.tensor, before)

    def _check_upload_close_failures(self, device, uploads, buffer, lifecycle):
        with (
            patch.object(
                device.runtime,
                "cudaEventSynchronize",
                side_effect=RuntimeError("wait failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "wait failed"),
        ):
            uploads.close()
        self.assertIsNotNone(buffer._native)
        self.assertIsNotNone(buffer.host_numpy)
        self.assertIsNotNone(buffer.storage)
        release_storage = device.upload_pool.release

        def release_checked(storage):
            self.assertIsNone(buffer._native)
            self.assertIsNone(buffer.host_numpy)
            self.assertIsNone(buffer.tensor)
            lifecycle.append("release")
            release_storage(storage)

        with patch.object(device.upload_pool, "release", side_effect=release_checked):
            uploads.close()
        self.assertEqual(lifecycle[-3:], ["wait", "release", "destroy"])
        self.assertFalse(buffer.event)

    def _check_upload_launch_failures(
        self, pending, submit, uploads, snapshot, source, buffer, device
    ):
        def fail_after_launch(destination, source):
            pending.append((torch.from_dlpack(destination), torch.from_dlpack(source)))
            raise RuntimeError("upload failed")

        submit.side_effect = fail_after_launch
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        self.assertFalse(buffer.pending)
        self.assertFalse(buffer.untracked)
        self.assertFalse(pending)
        submit.side_effect = lambda destination, source: pending.append(
            (torch.from_dlpack(destination), torch.from_dlpack(source))
        )
        with (
            patch.object(
                device.runtime,
                "cudaEventRecord",
                side_effect=RuntimeError("record failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "record failed"),
        ):
            uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        self.assertFalse(buffer.pending)
        self.assertFalse(pending)
        uploads.update(np.copyto, snapshot.numpy(), source.numpy())

    def _check_upload_binding_refresh(
        self,
        Uploads,
        device,
        source,
        destination,
        current,
        bind,
        copyto,
        scope,
        waits,
        submit,
        torch_adapter,
    ):
        uploads = Uploads(device)
        snapshot = uploads.prepare(destination, source)
        uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        buffer = next(iter(uploads.buffers.values()))
        before = snapshot.clone()
        writer = Mock()
        with (
            patch(
                "oh_my_vllm.operators.runtime.uploads._eager_stream",
                side_effect=RuntimeError("Graph recording"),
            ),
            self.assertRaisesRegex(RuntimeError, "Graph recording"),
        ):
            uploads.update(writer)
        writer.assert_not_called()
        torch.testing.assert_close(snapshot, before)
        self.assertIsInstance(buffer.tensor, torch.Tensor)
        self.assertEqual(buffer.tensor.device.type, "cpu")
        array = buffer.host_numpy
        first_binding = buffer._native
        self.assertTrue(np.shares_memory(array, buffer.tensor.numpy()))
        source.fill_(9)
        current.return_value = 11
        uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        torch.testing.assert_close(destination, torch.tensor([1, 2], dtype=torch.int32))
        self.assertIs(buffer.host_numpy, array)
        self.assertIs(buffer._native, first_binding)
        self.assertEqual(bind.call_count, 2)
        self.assertEqual(copyto.call_count, 2)
        self.assertEqual([call.args[1] for call in scope.call_args_list], [7, 11])
        self.assertEqual(waits, [True])
        rebound = torch.zeros_like(destination)
        destination.set_(rebound)
        before_submit = submit.call_count
        with (
            patch.object(
                torch_adapter,
                "native_tensor",
                side_effect=RuntimeError("bind failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "bind failed"),
        ):
            uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        self.assertEqual(submit.call_count, before_submit)
        self.assertFalse(buffer.pending)
        uploads.update(np.copyto, snapshot.numpy(), source.numpy())
        self.assertEqual(bind.call_count, 4)
        self.assertIsNot(buffer._native, first_binding)
        first_binding = None
        return array, buffer, snapshot, uploads

    def _upload_lifecycle_device(self, UploadPool):
        allocations, pending, waits, lifecycle = {}, [], [], []
        idle = [False]

        def wait_safe():
            idle[0] = True

        def allocate(pointer, size, flags):
            storage = ctypes.create_string_buffer(size)
            address = ctypes.addressof(storage)
            allocations[address] = storage
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
            return 0

        def create_event(pointer, flags):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 19
            return 0

        def destroy_event(event):
            lifecycle.append("destroy")
            return 0

        def wait(event):
            lifecycle.append("wait")
            waits.append(True)
            for destination, source in pending:
                destination.copy_(source)
            pending.clear()
            return 0

        def release(pointer):
            del allocations[pointer.value]
            return 0

        def capture(stream, status):
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

        device = SimpleNamespace(
            runtime=SimpleNamespace(
                cudaHostAlloc=allocate,
                cudaFreeHost=release,
                cudaEventCreateWithFlags=create_event,
                cudaEventDestroy=destroy_event,
                cudaEventRecord=lambda *_: 0,
                cudaEventSynchronize=wait,
                cudaStreamIsCapturing=capture,
            ),
            checked=lambda status: self.assertEqual(status, 0),
            synchronize=lambda: wait(None),
            pinned_quiescent=lambda: idle[0],
            wait_pinned_safe=wait_safe,
            view=lambda value, shape, offset: value,
        )
        device.upload_pool = UploadPool(device)
        return allocations, pending, waits, lifecycle, device
