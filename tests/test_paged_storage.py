import gc
import importlib.util
import unittest
import weakref

import torch
import tvm_ffi


class PagedStorageTests(unittest.TestCase):
    def test_paged_runtime_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("oh_my_vllm.operators.runtime.paged"),
            "the bounded CUDA VMM storage runtime is missing",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaPagedStorageTests(unittest.TestCase):
    def setUp(self):
        from oh_my_vllm.operators.runtime import paged

        self.paged = paged
        self.module = paged._module()
        gc.collect()
        self.reclaimer = self.allocate((1, 1, 1))
        self.baseline = paged.mapped_bytes()
        self.addCleanup(self.check_cleanup)

    def check_cleanup(self):
        gc.collect()
        self.reclaim()
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def reclaim(self):
        self.reclaimer._paged_storage.reserve(0, 0)

    def allocate(self, shape=(4, 1 << 20, 32), dtype=torch.float32):
        return self.paged.allocate_paged(shape, dtype=dtype, device="cuda:0")

    def test_huge_logical_storage_has_no_physical_pages(self):
        value = self.allocate((64, 1 << 26, 32))
        self.assertEqual(tuple(value.shape), (64, 1 << 26, 32))
        self.assertTrue(value.is_contiguous())
        self.assertEqual(value.stride(), (1 << 31, 32, 1))
        self.assertGreater(value.numel() * value.element_size(), 1 << 38)
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)
        self.assertNotEqual(value.data_ptr(), 0)

    def test_small_slots_have_independent_padded_strides(self):
        value = self.allocate((4, 3, 7))
        self.assertEqual(tuple(value.shape), (4, 3, 7))
        self.assertGreaterEqual(value.stride(0), 21)
        self.assertEqual(value.stride()[1:], (7, 1))
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)
        value._paged_storage.reserve(1, 3)
        first_bytes = self.paged.mapped_bytes() - self.baseline
        value[1].fill_(13)
        value._paged_storage.reserve(2, 2)
        self.assertEqual(self.paged.mapped_bytes() - self.baseline, 2 * first_bytes)
        torch.testing.assert_close(value[1], torch.full_like(value[1], 13))
        torch.testing.assert_close(value[2, :2], torch.zeros_like(value[2, :2]))
        value._paged_storage.release(1)
        self.assertEqual(self.paged.mapped_bytes() - self.baseline, first_bytes)
        value._paged_storage.reserve(1, 3)
        torch.testing.assert_close(value[1], torch.zeros_like(value[1]))

    def test_byte_aligned_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.int64, torch.uint8):
            with self.subTest(dtype=dtype):
                value = self.allocate((2, 3, 7), dtype=dtype)
                self.assertEqual(value.dtype, dtype)
                value._paged_storage.reserve(0, 3)
                torch.testing.assert_close(value[0], torch.zeros_like(value[0]))
                value[0].fill_(9)
                value._paged_storage.reserve(1, 3)
                torch.testing.assert_close(value[1], torch.zeros_like(value[1]))
                torch.testing.assert_close(value[0], torch.full_like(value[0], 9))
                del value
                self.reclaim()
                self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_selected_slots_grow_without_replacing_addresses(self):
        value = self.allocate()
        owner = value._paged_storage
        pointer = value.data_ptr()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            owner.reserve(2, 1)
            first_bytes = self.paged.mapped_bytes() - self.baseline
            self.assertGreater(first_bytes, 0)
            torch.testing.assert_close(value[2, :1], torch.zeros_like(value[2, :1]))
            value[2, :1].fill_(7)
            owner.reserve(2, 1)
            self.assertEqual(self.paged.mapped_bytes() - self.baseline, first_bytes)
            stop = first_bytes // (32 * value.element_size()) + 1
            owner.reserve(2, stop)
            self.assertEqual(self.paged.mapped_bytes() - self.baseline, 2 * first_bytes)
            torch.testing.assert_close(value[2, :1], torch.full_like(value[2, :1], 7))
            torch.testing.assert_close(
                value[2, 1:stop], torch.zeros_like(value[2, 1:stop])
            )
            owner.reserve(0, 1)
            self.assertEqual(self.paged.mapped_bytes() - self.baseline, 3 * first_bytes)
            torch.testing.assert_close(value[0, :1], torch.zeros_like(value[0, :1]))
        stream.synchronize()
        self.assertEqual(value.data_ptr(), pointer)

    def test_release_waits_for_other_streams_and_reuse_is_zero(self):
        value = self.allocate()
        owner = value._paged_storage
        pointer = value.data_ptr()
        owner.reserve(1, 1)
        one_slot = self.paged.mapped_bytes() - self.baseline
        owner.reserve(3, 1)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            value[1, :1].fill_(11)
        owner.release(1)
        self.assertTrue(stream.query())
        self.assertEqual(self.paged.mapped_bytes() - self.baseline, one_slot)
        owner.reserve(1, 1)
        self.assertEqual(value.data_ptr(), pointer)
        torch.testing.assert_close(value[1, :1], torch.zeros_like(value[1, :1]))
        owner.release(1)
        owner.release(1)
        owner.release(3)
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_views_keep_native_allocation_alive_without_python_cycle(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 4)
        value[0, :4].fill_(3)
        allocated = self.paged.mapped_bytes()
        view = value[0, :4].detach()
        reference = weakref.ref(value)
        del value
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(self.paged.mapped_bytes(), allocated)
        torch.testing.assert_close(view, torch.full_like(view, 3))
        del view
        gc.collect()
        self.assertEqual(self.paged.mapped_bytes(), allocated)
        self.reclaim()
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_next_eager_reclamation_waits_for_device_work(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        output = torch.empty_like(value[0, :4])
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            value[0, :4].fill_(5)
            output.copy_(value[0, :4])
        del value
        gc.collect()
        self.reclaim()
        self.assertTrue(stream.query())
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)
        torch.testing.assert_close(output, torch.full_like(output, 5))

    def test_cleanup_deferred_during_capture_is_retried(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 4)
        scratch = torch.zeros(1, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=stream),
            tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.cuda_stream),
        ):
            scratch.add_(1)
            del value
        gc.collect()
        self.reclaim()
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_last_owner_and_getter_do_not_touch_cuda(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 4)
        allocated = self.paged.mapped_bytes()
        del value
        gc.collect()
        self.assertEqual(self.paged.mapped_bytes(), allocated)
        self.assertEqual(self.module.mapped_bytes(), allocated)
        self.reclaim()
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_plain_capture_defers_last_dlpack_view_and_getter_is_pure(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 4)
        allocated = self.paged.mapped_bytes()
        view = value[0, :4].detach()
        del value
        scratch = torch.zeros(1, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scratch.add_(1)
            del view
            self.assertEqual(self.paged.mapped_bytes(), allocated)
            self.assertEqual(self.module.mapped_bytes(), allocated)
            self.assertTrue(torch.cuda.is_current_stream_capturing())
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(scratch.item(), 1)
        self.assertEqual(self.paged.mapped_bytes(), allocated)
        self.reclaim()
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_eager_allocate_and_release_collect_pending(self):
        for operation in ("allocate", "release"):
            with self.subTest(operation=operation):
                value = self.allocate()
                value._paged_storage.reserve(0, 4)
                allocated = self.paged.mapped_bytes()
                del value
                self.assertEqual(self.paged.mapped_bytes(), allocated)
                if operation == "allocate":
                    replacement = self.allocate()
                    del replacement
                else:
                    self.reclaimer._paged_storage.release(0)
                self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_cross_stream_reserve_orders_new_page_zeroing(self):
        value = self.allocate()
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(first):
            torch.cuda._sleep(10_000_000)
            value._paged_storage.reserve(0, 1)
        with torch.cuda.stream(second):
            value._paged_storage.reserve(0, 1)
            torch.testing.assert_close(value[0, :1], torch.zeros_like(value[0, :1]))
            value[0, :1].fill_(19)
        second.synchronize()
        mapped = self.paged.mapped_bytes() - self.baseline
        stop = mapped // (32 * value.element_size()) + 1
        with torch.cuda.stream(first):
            torch.cuda._sleep(10_000_000)
            value._paged_storage.reserve(0, stop)
        with torch.cuda.stream(second):
            value._paged_storage.reserve(0, stop)
            torch.testing.assert_close(value[0, :1], torch.full_like(value[0, :1], 19))
            torch.testing.assert_close(
                value[0, 1:stop], torch.zeros_like(value[0, 1:stop])
            )
        second.synchronize()

    def test_invalid_inputs_do_not_map_pages(self):
        value = self.allocate()
        native = value._paged_storage._tensor
        for slot, stop in ((-1, 1), (4, 1), (0, -1), (0, (1 << 20) + 1)):
            with self.subTest(slot=slot, stop=stop), self.assertRaises(RuntimeError):
                self.module.commit_rows(native, slot, stop)
        for slot in (-1, 4):
            with self.subTest(slot=slot), self.assertRaises(RuntimeError):
                self.module.release_slot(native, slot)
        foreign = torch.empty((4, 32, 32), device="cuda")
        for tensor in (foreign, value[0], value.transpose(1, 2)):
            with self.subTest(shape=tensor.shape), self.assertRaises(RuntimeError):
                self.module.commit_rows(tensor, 0, 1)
        for shape, dtype in (
            ((4,), "float32"),
            ((0, 1 << 20, 32), "float32"),
            ((4, -1, 32), "float32"),
            ((4, 1 << 62, 32), "float32"),
            ((4, 1 << 20, 32), "bool"),
            ((4, 1 << 20, 32), "float32x2"),
        ):
            with (
                self.subTest(shape=shape, dtype=dtype),
                self.assertRaises(RuntimeError),
            ):
                self.module.allocate(tvm_ffi.Shape(shape), dtype, 0)
        value._paged_storage.reserve(0, 0)
        self.assertEqual(self.paged.mapped_bytes(), self.baseline)

    def test_mapping_and_release_reject_capture(self):
        value = self.allocate()
        value._paged_storage.reserve(0, 1)
        before = self.paged.mapped_bytes()
        for operation in (
            lambda: self.allocate(),
            lambda: value._paged_storage.reserve(1, 1),
            lambda: value._paged_storage.release(0),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                value[0, :1].add_(0)
                with self.assertRaisesRegex(
                    RuntimeError, "do not support CUDA capture"
                ):
                    operation()
            self.assertEqual(self.paged.mapped_bytes(), before)


if __name__ == "__main__":
    unittest.main()
