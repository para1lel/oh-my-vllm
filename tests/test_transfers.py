import ctypes
import unittest
from types import SimpleNamespace


class PinnedTransfers(unittest.TestCase):
    def test_incomplete_cuda_event_is_not_a_transfer_failure(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        def checked(status):
            if status:
                raise RuntimeError(f"CUDA failure {status}")

        engine = TransferEngine.__new__(TransferEngine)
        engine.device = SimpleNamespace(checked=checked)
        self.assertFalse(engine.ready(600))
        self.assertTrue(engine.ready(0))
        with self.assertRaisesRegex(RuntimeError, "700"):
            engine.ready(700)

    def test_pool_defers_growth_and_preserves_live_slices(self):
        from oh_my_vllm.operators.runtime.transfers import PinnedPool

        allocations = {}
        idle = [False]

        def allocate(pointer, size, flags):
            self.assertTrue(idle[0], "pinning overlapped active CUDA work")
            memory = ctypes.create_string_buffer(size)
            address = ctypes.addressof(memory)
            allocations[address] = memory
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
            return 0

        def release(pointer):
            self.assertTrue(idle[0])
            del allocations[pointer.value]
            return 0

        device = SimpleNamespace(
            runtime=SimpleNamespace(cudaHostAlloc=allocate, cudaFreeHost=release),
            checked=lambda status: self.assertEqual(status, 0),
        )
        pool = PinnedPool(device, 1024, lambda: idle[0])
        self.assertIsNone(pool.acquire(256))
        self.assertFalse(allocations)
        idle[0] = True
        first, second = pool.acquire(256), pool.acquire(512)
        first.array[:] = 17
        second.array[:] = 29
        self.assertEqual(pool.allocated_bytes, 1024)
        self.assertEqual(pool.used_bytes, 768)
        idle[0] = False
        pool.release(first)
        self.assertIsNone(first.pool)
        self.assertIsNone(first.slab)
        self.assertIsNone(first.tensor)
        self.assertIsNone(first.array)
        self.assertEqual(pool.used_bytes, 512)
        with self.assertRaises(ValueError):
            pool.release(first)
        first.close()
        self.assertEqual(pool.used_bytes, 512)
        third = pool.acquire(256)
        fourth = pool.acquire(256)
        self.assertIsNotNone(third)
        self.assertIsNotNone(fourth)
        self.assertNotEqual(third.array.ctypes.data, fourth.array.ctypes.data)
        third.array[:] = 42
        fourth.array[:] = 53
        self.assertTrue((third.array == 42).all())
        self.assertTrue((second.array == 29).all())
        self.assertEqual(pool.used_bytes, 1024)
        self.assertIsNone(pool.acquire(512))
        second.close()
        third.close()
        third.close()
        with self.assertRaises(ValueError):
            pool.release(third)
        fourth.close()
        self.assertEqual(pool.used_bytes, 0)
        self.assertEqual(len(allocations), 1)
        idle[0] = True
        pool.close()
        self.assertFalse(allocations)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.acquire(1)

    def test_pool_rejects_invalid_sizes_and_live_close(self):
        from oh_my_vllm.operators.runtime.transfers import PinnedPool

        device = SimpleNamespace()
        pool = PinnedPool(device, 0, lambda: True)
        self.assertIsNone(pool.acquire(1))
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                pool.acquire(size)
        pool.close()


if __name__ == "__main__":
    unittest.main()
