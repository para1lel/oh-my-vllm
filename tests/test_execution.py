import ctypes
import unittest
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from oh_my_vllm.models.qwen38.inference import Worker
from oh_my_vllm.operators.runtime.device import Device


class GraphOwner:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail

    @contextmanager
    def graph_lease(self):
        self.events.append(("lease",))
        try:
            yield
        finally:
            self.events.append(("release",))

    def run(self):
        self.events.append(("record",))
        if self.fail:
            raise ValueError("recording failed")


def graph_device(events, failure=None):
    device = Device.__new__(Device)
    handles = {
        "cudaStreamCreateWithFlags": (0, 11),
        "cudaStreamEndCapture": (1, 12),
        "cudaGraphInstantiateWithFlags": (0, 13),
    }

    def call(name, *args):
        events.append((name, *args))
        if name == failure:
            return 1
        if name in handles:
            argument, value = handles[name]
            ctypes.cast(args[argument], ctypes.POINTER(ctypes.c_void_p))[0] = value
        return 0

    names = (
        *handles,
        "cudaStreamBeginCapture",
        "cudaStreamDestroy",
        "cudaGraphDestroy",
        "cudaGraphExecDestroy",
        "cudaGraphLaunch",
    )
    device.runtime = SimpleNamespace(
        **{name: lambda *args, name=name: call(name, *args) for name in names},
        cudaGetErrorString=lambda _: b"injected CUDA failure",
    )
    device.synchronize = lambda: events.append(("synchronize",))
    return device


class Execution(unittest.TestCase):
    def test_cache_capacity_is_fixed_and_never_silently_reduced(self):
        from oh_my_vllm.models.qwen38.cache_budget import cache_budget

        config = dict(
            num_gpu_blocks=None,
            mamba_blocks=128,
            max_num_seqs=4,
            max_num_batched_tokens=32768,
            num_speculative_tokens=0,
        )
        gib = 1 << 30
        for speculative, expected in ((0, 1824), (4, 1608)):
            config["num_speculative_tokens"] = speculative
            for total, free in ((192 * gib, 162 * gib), (200 * gib, 160 * gib)):
                result = cache_budget(
                    config,
                    total_bytes=total,
                    free_bytes=free,
                    overhead_bytes=16 * gib,
                )
                self.assertEqual(result["num_gpu_blocks"], expected)
                self.assertEqual(result["policy"], "static_blocks")
            with self.assertRaisesRegex(ValueError, "cache does not fit"):
                cache_budget(
                    config,
                    total_bytes=180 * gib,
                    free_bytes=100 * gib,
                    overhead_bytes=16 * gib,
                )
            config["num_gpu_blocks"] = 512
            result = cache_budget(
                config,
                total_bytes=180 * gib,
                free_bytes=100 * gib,
                overhead_bytes=16 * gib,
            )
            self.assertEqual(result["num_gpu_blocks"], 512)
            for invalid in (0, 1, -1, True, 2.5, 1 << 32):
                config["num_gpu_blocks"] = invalid
                with self.assertRaises(ValueError):
                    cache_budget(
                        config,
                        total_bytes=180 * gib,
                        free_bytes=150 * gib,
                        overhead_bytes=16 * gib,
                    )
            config["num_gpu_blocks"] = None

    def test_upload_snapshots_source_and_waits_before_reuse_or_release(self):
        from oh_my_vllm.operators.runtime.uploads import UploadPool, Uploads

        events, allocations, pending = [], {}, []
        idle, safe_waits = [False], []

        def wait_safe():
            safe_waits.append(True)
            idle[0] = True

        def allocate(pointer, size, flags):
            self.assertTrue(idle[0], "pinning requires an idle window")
            storage = ctypes.create_string_buffer(size)
            address = ctypes.addressof(storage)
            allocations[address] = storage
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
            return 0

        def create(pointer, flags):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 1
            return 0

        def wait(event):
            events.append("wait")
            for destination, source in pending:
                np.copyto(destination, source)
            pending.clear()
            return 0

        def release(pointer):
            self.assertFalse(pending)
            self.assertTrue(idle[0], "unpinning requires an idle window")
            events.append("free")
            del allocations[pointer.value]
            return 0

        def capture(stream, status):
            ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

        device = SimpleNamespace(
            runtime=SimpleNamespace(
                cudaHostAlloc=allocate,
                cudaFreeHost=release,
                cudaEventCreateWithFlags=create,
                cudaEventDestroy=lambda _: 0,
                cudaEventRecord=lambda *_: 0,
                cudaEventSynchronize=wait,
                cudaStreamIsCapturing=capture,
            ),
            module=SimpleNamespace(
                upload_bytes=lambda dst, src: pending.append((dst, src))
            ),
            checked=lambda status: self.assertEqual(status, 0),
            synchronize=lambda: wait(None),
            pinned_quiescent=lambda: idle[0],
            wait_pinned_safe=wait_safe,
        )
        source, destination = np.array([1, 2], np.int32), np.zeros(2, np.int32)
        device.upload_pool = UploadPool(device)
        with (
            patch("tvm_ffi.from_dlpack", side_effect=lambda array: array),
            patch("tvm_ffi.get_raw_stream", return_value=7),
        ):
            uploads = Uploads(device)
            uploads.copy(destination, source)
            self.assertEqual(len(safe_waits), 1)
            source[:] = 3
            self.assertEqual(destination.tolist(), [0, 0])
            uploads.copy(destination, source)
            self.assertEqual(destination.tolist(), [1, 2])
            source[:] = 4
            uploads.close()
            self.assertEqual(destination.tolist(), [3, 3])
            self.assertEqual(events, ["wait", "wait"])
            self.assertEqual(device.upload_pool.cached_bytes, source.nbytes)
            uploads = Uploads(device)
            uploads.copy(destination, source)
            self.assertEqual(device.upload_pool.cached_bytes, 0)
            self.assertEqual(len(allocations), 1)
            uploads.close()
            self.assertEqual(destination.tolist(), [4, 4])
            self.assertEqual(events, ["wait", "wait", "wait"])
            idle[0] = False
            smaller = device.upload_pool.acquire(4)
            self.assertEqual(smaller.size, 8)
            self.assertEqual(len(allocations), 1)
            self.assertEqual(len(safe_waits), 1)
            device.upload_pool.release(smaller)
            device.upload_pool.close()
            self.assertEqual(events, ["wait", "wait", "wait", "free"])
            self.assertFalse(allocations)
            uploads.close()

            device.upload_pool = UploadPool(device, limit=16)
            large = device.upload_pool.acquire(16)
            small = [device.upload_pool.acquire(8) for _ in range(3)]
            idle[0] = False
            waits_before = len(safe_waits)
            frees_before = events.count("free")
            device.upload_pool.release(large)
            device.upload_pool.release(small[0])
            device.upload_pool.release(small[1])
            self.assertEqual(device.upload_pool.cached_bytes, 16)
            self.assertEqual(device.upload_pool.retired_bytes, 16)
            reused = device.upload_pool.acquire(4)
            self.assertEqual(reused.size, 8)
            self.assertEqual(device.upload_pool.retired_bytes, 8)
            device.upload_pool.release(reused)
            self.assertEqual(len(safe_waits), waits_before)
            self.assertEqual(events.count("free"), frees_before)
            device.upload_pool.release(small[2])
            self.assertEqual(len(safe_waits), waits_before + 1)
            self.assertEqual(device.upload_pool.retired_bytes, 0)
            self.assertEqual(len(allocations), 1)
            oversized = device.upload_pool.acquire(64)
            idle[0] = False
            waits_before = len(safe_waits)
            device.upload_pool.release(oversized)
            self.assertEqual(len(safe_waits), waits_before + 1)
            self.assertEqual(device.upload_pool.retired_bytes, 0)
            self.assertEqual(len(allocations), 1)
            device.upload_pool.close()
            self.assertFalse(allocations)

            device.upload_pool = UploadPool(device, limit=8)
            cached = device.upload_pool.acquire(8)
            retired = device.upload_pool.acquire(8)
            idle[0] = False
            device.upload_pool.release(cached)
            device.upload_pool.release(retired)
            failed_address = retired.pointer.value
            device.runtime.cudaFreeHost = lambda pointer: (
                1 if pointer.value == failed_address else release(pointer)
            )
            with self.assertRaises(AssertionError):
                device.upload_pool.close()
            self.assertEqual(device.upload_pool.retired_bytes, 8)
            self.assertEqual(list(allocations), [failed_address])
            device.runtime.cudaFreeHost = release
            device.upload_pool.close()
            self.assertEqual(device.upload_pool.retired_bytes, 0)
            self.assertFalse(allocations)

            device.upload_pool = UploadPool(device, limit=0)
            device.runtime.cudaStreamIsCapturing = capture
            device.runtime.cudaEventRecord = lambda *_: 0
            uploads = Uploads(device)
            uploads.copy(destination, source)
            uploads.copy(destination.copy(), source)
            failed_address = next(iter(allocations))
            device.runtime.cudaFreeHost = lambda pointer: (
                1 if pointer.value == failed_address else release(pointer)
            )
            with self.assertRaises(AssertionError):
                uploads.close()
            self.assertEqual(list(allocations), [failed_address])
            device.runtime.cudaFreeHost = release
            uploads.close()
            self.assertFalse(allocations)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                uploads.copy(destination, source)
            uploads = Uploads(device)
            device.runtime.cudaEventRecord = lambda *_: 1
            with self.assertRaises(AssertionError):
                uploads.copy(destination, source)
            self.assertEqual(destination.tolist(), [4, 4])
            uploads.close()
            self.assertFalse(allocations)

            def captured(stream, status):
                ctypes.cast(status, ctypes.POINTER(ctypes.c_int))[0] = 1
                return 0

            device.runtime.cudaStreamIsCapturing = captured
            uploads = Uploads(device)
            with self.assertRaisesRegex(RuntimeError, "before CUDA Graph"):
                uploads.copy(destination, source)
            self.assertFalse(allocations)
            uploads.close()

    def test_partial_prefix_copies_both_planes_and_restores_exact_state(self):
        from oh_my_vllm.models.qwen38.cache import Cache

        copies = []
        cache = Cache.__new__(Cache)
        cache.blocks = cache.mamba_blocks = 8
        cache.private_start, cache.scratch_start, cache.slots = 8, 20, 24
        cache.indices, cache.free_indices = {}, deque(range(2))
        cache.fa = tuple((layer,) for layer in range(17))
        cache.device = SimpleNamespace(
            view=lambda tensor, shape, offset: (tensor, shape, offset),
            module=SimpleNamespace(
                copy_bytes=lambda dst, src: copies.append((dst, src))
            ),
        )
        cache.copy_prefix([(2, 3, 316)])
        self.assertEqual(len(copies), 34)
        for layer in range(17):
            for plane in range(2):
                destination, source = copies[layer * 2 + plane]
                self.assertEqual(
                    source, (layer, (316, 4, 256), (4 + plane) * 784 * 1024)
                )
                self.assertEqual(
                    destination, (layer, (316, 4, 256), (6 + plane) * 784 * 1024)
                )
        request = SimpleNamespace(
            state_position=0, state_slot=None, position=lambda token: (token,) * 3
        )
        row = dict(
            request_id=1,
            num_computed_tokens=316,
            token_ids=[7],
            prefix_state_slot=5,
            fa_block_table=[3],
            mamba_block_table=[6],
            cache_checkpoint=(317, 6),
        )
        batch = cache.batch([row], [request], False)
        self.assertEqual(batch.schedule.read_slots, (5,))
        self.assertEqual(batch.writes, (6,))
        self.assertEqual(batch.metadata[1].tolist(), [3 * 784 + 316])
        cache.commit(request, batch, 0, None)
        self.assertEqual((request.state_position, request.state_slot), (317, 6))
        row.update(num_computed_tokens=317, cache_checkpoint=None)
        continued = cache.batch([row], [request], False)
        self.assertEqual(continued.schedule.read_slots, (6,))
        self.assertNotIn(continued.writes[0], (5, 6))
        row["cache_checkpoint"] = (318, 6)
        with self.assertRaisesRegex(ValueError, "distinct private state"):
            cache.batch([row], [request], False)

    def test_prefill_controls_the_entire_execute(self):
        worker = Worker.__new__(Worker)
        worker.speculation = None
        worker.requests = {
            "decode": SimpleNamespace(prompt_len=16, history=list(range(17))),
            "prefill": SimpleNamespace(prompt_len=40000, history=list(range(40000))),
        }
        decode = dict(request_id="decode", num_computed_tokens=16, token_ids=[13])
        prefill = dict(
            request_id="prefill", num_computed_tokens=0, token_ids=[1] * 40000
        )
        for rows, expected in (
            ([decode], [(True, True)]),
            ([prefill], [(False, False), (False, False)]),
            ([decode, prefill], [(True, False), (False, False), (False, False)]),
        ):
            calls = []

            def batch(items, verify, graph, calls=calls):
                calls.append((verify, graph))
                return [dict(request_id=item["request_id"]) for item in items]

            worker.execute_batch = batch
            result = worker.execute(
                dict(
                    rpc_id="execution",
                    scheduled=rows,
                    finished_request_ids=[],
                    preempted_request_ids=[],
                )
            )
            self.assertEqual(calls, expected)
            self.assertEqual(result["rpc_id"], "execution")
            self.assertEqual(
                {row["request_id"] for row in result["outputs"]},
                {row["request_id"] for row in rows},
            )

    def test_stop_truncation_controls_accepted_slot(self):
        from oh_my_vllm.models.qwen38.cache import Cache

        for emitted, expected_slot, expected_position in ((1, 10, 101), (0, 99, 100)):
            worker = Worker.__new__(Worker)
            request = SimpleNamespace(
                prompt_len=16,
                state_slot=99,
                state_position=100,
                finish="stop",
                thinking_count=0,
                drafts=[],
                append=lambda tokens, n=emitted: (tokens[:n], ""),
            )
            rows = [
                dict(
                    request_id="stop",
                    num_computed_tokens=100,
                    token_ids=[1, 2, 3, 4, 5],
                )
            ]
            batch = SimpleNamespace(
                rows=5,
                scheduled=rows,
                writes=[10, 11, 12, 13, 14],
                schedule=SimpleNamespace(sequence_rows=(0, 5)),
            )
            worker.requests = {"stop": request}
            cache = Cache.__new__(Cache)
            cache.batch = lambda *_, batch=batch: batch
            worker.cache = cache
            worker.model = SimpleNamespace(forward=lambda *_: ((None, None), None))
            worker.sampler = lambda *_: None
            worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
            worker.speculation = SimpleNamespace(
                accept=lambda *_: [[2, 3, 4, 5]],
                context=lambda *_: {},
                propose=lambda *_: None,
            )
            result = worker.execute_batch(rows, True, True)
            self.assertEqual(result[0]["token_ids"], [2][:emitted])
            self.assertEqual(request.state_slot, expected_slot)
            self.assertEqual(request.state_position, expected_position)

    def test_graph_owns_capture_replay_and_cleanup(self):
        with (
            patch("tvm_ffi.use_raw_stream") as use_stream,
            patch("tvm_ffi.get_raw_stream", return_value=77),
        ):
            for failure in (
                None,
                "cudaStreamBeginCapture",
                "cudaStreamEndCapture",
                "cudaGraphInstantiateWithFlags",
                "record",
            ):
                events = []
                device = graph_device(events, failure)
                owner = GraphOwner(events, failure == "record")
                if failure:
                    with (
                        self.assertRaises(
                            ValueError if failure == "record" else RuntimeError
                        ),
                        device.graph(owner.run),
                    ):
                        self.fail("failed recording must not produce a replay")
                    self.assertEqual(events[-1][0], "release")
                    self.assertIn("cudaStreamDestroy", [event[0] for event in events])
                    continue
                with device.graph(owner.run) as replay:
                    self.assertNotIn("cudaGraphLaunch", [event[0] for event in events])
                    replay()
                    self.assertEqual(events[-1][-1], 77)
                names = [event[0] for event in events]
                self.assertLess(
                    names.index("synchronize"), names.index("cudaGraphExecDestroy")
                )
                self.assertLess(
                    names.index("cudaGraphExecDestroy"), names.index("release")
                )
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    replay()
            events = []
            device = graph_device(events)
            with (
                self.assertRaises(KeyboardInterrupt),
                device.graph(GraphOwner(events).run) as replay,
                patch.object(
                    device.runtime, "cudaGraphLaunch", side_effect=KeyboardInterrupt
                ),
            ):
                replay()
            self.assertEqual(events[-1][0], "release")
            self.assertEqual(sum(event[0] == "synchronize" for event in events), 2)
            self.assertEqual(use_stream.call_count, 6)


if __name__ == "__main__":
    unittest.main()
