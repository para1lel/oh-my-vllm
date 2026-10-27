import ctypes
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        "cudaEventCreateWithFlags": (0, 21),
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
        "cudaEventRecord",
        "cudaEventDestroy",
        "cudaStreamIsCapturing",
        "cudaStreamWaitEvent",
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
    device.transfer_engines = set()
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

            class Tensor:
                def __init__(self, array):
                    self.array = array
                    self.shape, self.dtype = array.shape, array.dtype
                    self.device = tvm_ffi.device("cuda:0")

                def is_contiguous(self):
                    return self.array.flags.c_contiguous

                def numel(self):
                    return self.array.size

                def data_ptr(self):
                    return self.array.ctypes.data

            def view(tensor, shape, offset=0):
                return Tensor(
                    tensor.array.reshape(-1)[
                        offset : offset + int(np.prod(shape))
                    ].reshape(shape)
                )

            import tvm_ffi

            device.view = view
            device.module.upload_bytes = lambda dst, src: pending.append(
                (dst.array if isinstance(dst, Tensor) else dst, src)
            )
            device.runtime.cudaStreamIsCapturing = capture
            device.runtime.cudaEventRecord = lambda *_: 0
            device.upload_pool = UploadPool(device)
            larger = device.upload_pool.acquire(1024)
            device.upload_pool.release(larger)
            uploads = Uploads(device)
            target = Tensor(np.full((3, 4), -9, np.int32))
            for capacity in (0, -1, True, 2.5):
                with self.assertRaises(ValueError):
                    uploads.prepare_range(target, capacity_bytes=capacity)
            self.assertEqual(len(allocations), 1)
            span = uploads.prepare_range(target, capacity_bytes=12)
            self.assertEqual(span.staging_bytes, 16)
            self.assertEqual(len(allocations), 2)
            self.assertIs(uploads.prepare_range(target, capacity_bytes=12), span)
            with self.assertRaises(ValueError):
                uploads.prepare_range(target, capacity_bytes=16)
            values = np.array([1, 99, 2, 99, 3], np.int32)[::2]
            span.copy(values, offset=1)
            values[:] = 77
            self.assertEqual(target.array.reshape(-1).tolist(), [-9] * 12)
            span.copy(np.array([[7], [8]], np.int32), offset=8)
            self.assertEqual(
                target.array.reshape(-1).tolist(),
                [-9, 1, 2, 3, -9, -9, -9, -9, -9, -9, -9, -9],
            )
            before = len(events), len(pending), len(allocations)
            span.copy(np.empty((0, 2), np.int32), offset=12)
            for value, offset in (
                ([1], 0),
                (np.ones(1, np.float32), 0),
                (np.ones(1, dtype=">i4"), 0),
                (np.ones(1, np.int32), -1),
                (np.ones(1, np.int32), True),
                (np.ones(1, np.int32), 12),
                (np.ones(4, np.int32), 0),
            ):
                with self.assertRaises(ValueError):
                    span.copy(value, offset=offset)
            self.assertEqual(before, (len(events), len(pending), len(allocations)))
            device.runtime.cudaStreamIsCapturing = captured
            with self.assertRaises(RuntimeError):
                uploads.prepare_range(target, capacity_bytes=12)
            with self.assertRaises(RuntimeError):
                span.copy(np.empty(0, np.int32), offset=12)
            self.assertEqual(before, (len(events), len(pending), len(allocations)))
            device.runtime.cudaStreamIsCapturing = capture
            span.copy(np.array([5], np.int32))
            for _ in range(20):
                span.copy(np.array([5], np.int32))
            self.assertEqual(len(allocations), 2)
            uploads.close()
            self.assertEqual(
                target.array.reshape(-1).tolist(),
                [5, 1, 2, 3, -9, -9, -9, -9, 7, 8, -9, -9],
            )
            with self.assertRaises(RuntimeError):
                span.copy(np.empty(0, np.int32))
            device.upload_pool.close()
            self.assertFalse(allocations)

            device.upload_pool = UploadPool(device, limit=0)
            uploads = Uploads(device)
            span = uploads.prepare_range(target, capacity_bytes=4)
            device.runtime.cudaEventRecord = lambda *_: 1
            with self.assertRaises(AssertionError):
                span.copy(np.array([13], np.int32))
            self.assertEqual(target.array[0, 0], 13)
            self.assertFalse(pending)
            device.runtime.cudaEventRecord = lambda *_: 0
            device.runtime.cudaFreeHost = lambda _: 1
            with self.assertRaises(AssertionError):
                uploads.close()
            self.assertEqual(len(allocations), 1)
            device.runtime.cudaFreeHost = release
            uploads.close()
            self.assertFalse(allocations)

    def test_execute_delegates_prefill_and_mixed_graph_policy(self):
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
            ([prefill], [(False, None), (False, None)]),
            ([decode, prefill], [(True, None), (False, None), (False, None)]),
        ):
            calls = []

            def batch(items, verify, graph, calls=calls):
                calls.append((verify, graph))
                if not verify:
                    for item in items:
                        end = item["num_computed_tokens"] + len(item["token_ids"])
                        if end < 40000:
                            self.assertEqual(end % 784, 0)
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

    def test_execute_rejects_a_poisoned_device_before_model_work(self):
        worker = Worker.__new__(Worker)
        calls = []

        def reject():
            calls.append("health")
            raise RuntimeError("CUDA Graph device is unusable")

        worker.device = SimpleNamespace(ensure_healthy=reject)
        worker.offload = SimpleNamespace(
            check_execute=lambda *_: calls.append("offload")
        )
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            worker.execute(
                dict(
                    scheduled=[],
                    finished_request_ids=[],
                    preempted_request_ids=[],
                )
            )
        self.assertEqual(calls, ["health"])

    def test_adaptive_prefill_reuses_only_successful_compatible_workspaces(self):
        from oh_my_vllm.models.qwen38 import model

        events, fail = [], [False]

        def compute(*_):
            if fail[0]:
                raise ValueError("injected forward failure")
            events.append("compute")

        @contextmanager
        def graph(operation):
            events.append("capture")
            operation()
            try:
                yield lambda: events.append("replay")
            finally:
                events.append("graph_closed")

        def prepare(workspace, device, weights, cache, batch, plans):
            workspace.device, workspace.resources = device, ExitStack()
            workspace.replay, workspace.runs = None, 0
            workspace.media = workspace.media_rows = None
            workspace.first_arguments = (
                None,
                model.gdn.Inputs(object(), object()),
                None,
                None,
            )
            workspace.calls, workspace.warmups = (
                [compute],
                [lambda: events.append("warmup")],
            )
            workspace.output, workspace.hidden = object(), object()
            workspace.metadata_inputs = SimpleNamespace(close=lambda: None)
            workspace.fa = workspace.ffn = workspace.hidden_operation = None
            workspace.gdn = SimpleNamespace(
                run=compute,
                warmup=lambda *_: events.append("warmup"),
                graph_lease=nullcontext,
                close=lambda: None,
            )

        device = SimpleNamespace(graph=graph, synchronize_compute=lambda: None)
        binding = object(), object()
        media = SimpleNamespace(
            rows=lambda entries: binding if entries else (None, None)
        )
        owner = model.Model(device, None, None, False, media)

        def call(rows=64, *, graph=None, verify=False, media=False):
            batch = SimpleNamespace(
                rows=rows,
                schedule=SimpleNamespace(verify=verify, sequence_rows=(0, rows)),
                scheduled=[dict(num_computed_tokens=0, token_ids=[1])],
                requests=[SimpleNamespace(media=media)],
            )
            result = owner.forward(batch, graph)
            self.assertEqual(len(result), 3, "forward must report actual Graph use")
            workspace = next(iter(owner.decode.values())) if verify else owner.prefill
            self.assertIs(result[0], workspace.output)
            self.assertIs(result[1], workspace.hidden)
            return result[2]

        with (
            patch.object(model.Workspace, "__init__", prepare),
            patch.object(model.Workspace, "update", lambda *_: None),
            patch.object(
                model,
                "make_plans",
                side_effect=lambda batch, _: (SimpleNamespace(reuse_key=batch.rows),),
            ),
        ):
            try:
                self.assertEqual([call() for _ in range(6)], [False] * 4 + [True, True])
                self.assertEqual(events.count("capture"), 1)
                self.assertNotIn("warmup", events)
                retained = owner.prefill
                self.assertEqual([call(1025) for _ in range(5)], [False] * 5)
                self.assertIsNot(owner.prefill, retained)
                self.assertIn("graph_closed", events)
                self.assertEqual([call(1024) for _ in range(5)], [False] * 4 + [True])
                self.assertFalse(call(1024, media=True, graph=True))
                self.assertEqual([call(1024) for _ in range(5)], [False] * 4 + [True])
                self.assertFalse(call(1024, graph=False))
                self.assertTrue(call(1025, graph=True))
                self.assertTrue(call(1, verify=True))
                decode = next(iter(owner.decode.values()))
                call(63)
                self.assertIs(next(iter(owner.decode.values())), decode)
                fail[0] = True
                with self.assertRaisesRegex(ValueError, "injected forward failure"):
                    call(63)
                fail[0] = False
                self.assertEqual([call(63) for _ in range(4)], [False] * 3 + [True])
            finally:
                owner.close()

        from oh_my_vllm.models.qwen38 import speculation

        request = SimpleNamespace(
            state_position=2,
            mtp_position=0,
            history=[0, 1],
            position=lambda p: (p,) * 3,
        )
        batch = SimpleNamespace(
            requests=[request],
            scheduled=[
                dict(request_id="prefill", num_computed_tokens=0, fa_block_table=[0])
            ],
            schedule=SimpleNamespace(
                verify=False, sequence_rows=(0, 2), read_slots=(0,)
            ),
            metadata=(None, None, np.zeros((1, 1), np.int32)),
        )
        cache = SimpleNamespace()
        owner = speculation.Speculation(
            SimpleNamespace(view=lambda *_: self.fail("prefill must retain hidden")),
            None,
            cache,
            SimpleNamespace(rows=lambda _: (None, None)),
            Mock(),
        )
        hidden = object()
        with patch.object(
            speculation,
            "Context",
            side_effect=lambda *args: Mock(hidden=args[4], endpoint_rows=()),
        ) as context:
            try:
                for value, count in ((hidden, 1), (hidden, 1), (object(), 2)):
                    request.mtp_position = 0
                    self.assertEqual(owner.context(batch, value, True, {}), {})
                    self.assertEqual(context.call_count, count)
                    self.assertIs(context.call_args.args[-1], False)
                    self.assertIs(owner.context_prefill.hidden, value)
                    owner.context_prefill.copy.assert_not_called()
                    self.assertFalse(owner.context_decode)
            finally:
                owner.close()

    def test_prefill_fragments_preserve_blocks_and_token_intervals(self):
        worker = Worker.__new__(Worker)
        for limit in (32764, 32768, 783):
            rows = [
                dict(
                    request_id=index,
                    num_computed_tokens=start,
                    token_ids=list(range(count)),
                )
                for index, (start, count) in enumerate(
                    (
                        (0, 40000),
                        (32144, 623),
                        (317, 467),
                        (0, 1568),
                        (32767, 1),
                        (317, 40000),
                        *((0, 784),) * 4,
                    )
                )
            ]
            with self.subTest(limit=limit):
                fragments = list(worker.fragments(rows, limit, 4, block_size=784))
                restored = {row["request_id"]: [] for row in rows}
                positions = {
                    row["request_id"]: row["num_computed_tokens"] for row in rows
                }
                for batch in fragments:
                    self.assertLessEqual(
                        sum(len(row["token_ids"]) for row in batch), limit
                    )
                    self.assertLessEqual(len(batch), 4)
                    for row in batch:
                        request = row["request_id"]
                        self.assertEqual(row["num_computed_tokens"], positions[request])
                        restored[request].extend(row["token_ids"])
                        positions[request] += len(row["token_ids"])
                for row in rows:
                    self.assertEqual(restored[row["request_id"]], row["token_ids"])
                if limit >= 784:
                    self.assertEqual(len(fragments[0][0]["token_ids"]), 32144)
        tails = [
            dict(request_id=i, num_computed_tokens=784, token_ids=[i] * count)
            for i, count in enumerate((317, 450))
        ]
        self.assertEqual(list(worker.fragments(tails, 784, 4, block_size=784)), [tails])

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
            worker.model = SimpleNamespace(
                forward=lambda *_: ((None, None), None, True)
            )
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
        from oh_my_vllm.models.qwen38.speculation import GraphOperation

        for eager_first, expected in (
            (False, ["run", "run", "replay"]),
            (True, ["run", "run", "replay"]),
        ):
            events = []

            @contextmanager
            def graph(operation, events=events):
                operation()
                yield lambda: events.append("replay")

            operation = GraphOperation(
                SimpleNamespace(
                    uploads=lambda: SimpleNamespace(close=lambda: None),
                    synchronize_compute=lambda: None,
                    graph=graph,
                )
            )
            operation.run = lambda events=events: events.append("run")
            try:
                if eager_first:
                    operation.execute(False)
                operation.execute(True)
                self.assertEqual(events, expected)
            finally:
                operation.close()

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
