import ctypes
import unittest
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import tvm_ffi
from oh_my_vllm.kernels.mega import TileExecutionError, close, execute, prepare
from oh_my_vllm.kernels.tile_plan import Capacity, StepMetadata, Task
from oh_my_vllm.worker.device import Device
from oh_my_vllm.worker.zmq_bridge import error_kind


@contextmanager
def submission_stream(device):
    stream = ctypes.c_void_p()
    device.checked(device.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
    try:
        yield stream.value
    finally:
        device.checked(device.runtime.cudaStreamDestroy(stream))


class PersistentExecution(unittest.TestCase):
    def test_fan_in_reuse_capture_and_failure(self):
        device = Device()
        capacity = Capacity(tasks=128, edges=128, queue=64, rows=4096, workers=8)
        data = np.arange(4096, dtype=np.float32)
        buffers = (
            device.upload(data),
            *(device.upload(np.full_like(data, -1)) for _ in range(3)),
        )
        tasks = []
        for start in range(0, 4096, 128):
            first = len(tasks)
            tasks.extend(
                (
                    Task("scale", (0,), 1, start, 128, factor=2),
                    Task("scale", (0,), 2, start, 128, factor=3),
                    Task("add", (1, 2), 3, start, 128, depends=(first, first + 1)),
                    Task("copy", (3,), 1, start, 128, depends=(first + 2,)),
                )
            )
        plan = prepare(
            "elementwise",
            capacity,
            dict(device=device, buffers=buffers, tasks=tuple(tasks), outputs=(1, 3)),
        )
        try:
            for epoch, rows in (
                (1, 4096),
                (2, 783),
                (3, 784),
                (4, 785),
                (5, 0),
                (6, 4096),
            ):
                before = device.download(buffers[3], np.float32)
                current = data + epoch
                with submission_stream(device) as stream:
                    with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream):
                        device.module.copy_bytes(
                            buffers[0], tvm_ffi.from_dlpack(current)
                        )
                        completion = execute(plan, StepMetadata(epoch, (rows,), rows))
                    with self.assertRaises(RuntimeError):
                        execute(plan, StepMetadata(epoch + 1, (rows,), rows))
                    self.assertEqual(completion.wait().epoch, epoch)
                self.assertEqual(completion.completed_tasks, 128)
                actual = device.download(buffers[3], np.float32)
                np.testing.assert_array_equal(actual[:rows], current[:rows] * 5)
                np.testing.assert_array_equal(actual[rows:], before[rows:])
                np.testing.assert_array_equal(
                    device.download(buffers[1], np.float32)[:rows], current[:rows] * 5
                )
            with self.assertRaises(ValueError):
                execute(plan, StepMetadata(6, (4096,), 4096))
            # Capture only launch; submit metadata before replay.
            with (
                device.graph(plan.launch) as replay,
                submission_stream(device) as stream,
            ):
                current = data + 7
                with self.assertRaises(RuntimeError):
                    close(plan)
                with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream):
                    device.module.copy_bytes(buffers[0], tvm_ffi.from_dlpack(current))
                    completion = execute(
                        plan, StepMetadata(7, (4096,), 4096), replay=replay
                    )
                self.assertEqual(completion.wait().completed_tasks, 128)
            np.testing.assert_array_equal(
                device.download(buffers[3], np.float32), current * 5
            )
        finally:
            close(plan)
        with self.assertRaises(RuntimeError):
            execute(plan, StepMetadata(8, (4096,), 4096))
        with self.assertRaisesRegex(RuntimeError, "operator tensors overlap"):
            prepare(
                "elementwise",
                capacity,
                dict(
                    device=device,
                    buffers=(buffers[0], buffers[0]),
                    tasks=(Task("copy", (0,), 1, 0, 128),),
                    outputs=(1,),
                ),
            )
        # A single resident CTA makes overflow at this fan-out deterministic.
        tasks = (
            Task("copy", (0,), 1, 0, 128),
            Task("copy", (1,), 2, 0, 128, depends=(0,)),
            Task("copy", (1,), 3, 0, 128, depends=(0,)),
        )
        plan = prepare(
            "elementwise",
            replace(capacity, queue=1, workers=1),
            dict(device=device, buffers=buffers, tasks=tasks, outputs=(2, 3)),
        )
        try:
            completion = execute(plan, StepMetadata(1, (128,), 128))
            with self.assertRaises(TileExecutionError) as caught:
                completion.wait()
            self.assertEqual(
                error_kind({"type": "execute"}, caught.exception), "device"
            )
            with self.assertRaises(RuntimeError):
                execute(plan, StepMetadata(2, (128,), 128))
        finally:
            close(plan)
        for invalid_rows, expected_error in ((4097, 4), (128, 2)):
            plan = prepare(
                "elementwise",
                capacity,
                dict(
                    device=device,
                    buffers=buffers,
                    tasks=(Task("copy", (0,), 1, 0, 128),),
                    outputs=(1,),
                ),
            )
            try:
                with device.graph(plan.launch) as replay:
                    execute(plan, StepMetadata(1, (128,), 128), replay=replay).wait()
                    invalid = np.array((1, invalid_rows), dtype=np.int64)
                    if invalid_rows > capacity.rows:
                        invalid[0] = 2
                    device.module.copy_bytes(
                        plan.metadata, tvm_ffi.from_dlpack(invalid)
                    )
                    replay()
                    # A later valid epoch must not clear the device's previous failure.
                    completion = execute(
                        plan, StepMetadata(3, (128,), 128), replay=replay
                    )
                    with self.assertRaises(TileExecutionError):
                        completion.wait()
                    self.assertEqual(
                        int(device.download(plan.control, np.int64)[2]), expected_error
                    )
            finally:
                close(plan)
        rows = 2**31 - 1
        large = device.empty((rows,), "float32")
        tail = device.view(large, (1,), rows - 1)
        value = np.array((7,), dtype=np.float32)
        device.module.copy_bytes(tail, tvm_ffi.from_dlpack(value))
        plan = prepare(
            "elementwise",
            Capacity(tasks=1, edges=0, queue=1, rows=rows),
            dict(
                device=device,
                buffers=(large,),
                tasks=(Task("scale", (0,), 0, rows - 1, 1, factor=2),),
                outputs=(0,),
            ),
        )
        try:
            execute(plan, StepMetadata(1, (rows,), rows)).wait()
            np.testing.assert_array_equal(device.download(tail, np.float32), value * 2)
        finally:
            close(plan)


if __name__ == "__main__":
    unittest.main()
