import ctypes
import gc
import unittest
import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.inference import Worker
from oh_my_vllm.operators.runtime.device import Device


class Execution(unittest.TestCase):
    def test_transfer_keeps_torch_owners_until_completion(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        source = torch.arange(4, dtype=torch.float32)
        destination = torch.zeros(4, dtype=torch.float32)
        calls = []
        device = SimpleNamespace(
            runtime=SimpleNamespace(
                cudaEventCreateWithFlags=lambda *_: 0,
                cudaEventRecord=lambda *_: 0,
                cudaStreamWaitEvent=lambda *_: 0,
                cudaEventQuery=lambda *_: 0,
                cudaStreamQuery=lambda *_: 0,
                cudaStreamSynchronize=lambda *_: 0,
            ),
            checked=lambda status: self.assertEqual(status, 0),
            transfer_engines=set(),
            destroy_handles=lambda handles: None,
        )
        engine = TransferEngine.__new__(TransferEngine)
        engine.device = device
        engine.streams = {"h2d": ctypes.c_void_p(11)}
        engine.pending, engine.events, engine.closed = {}, [], False
        with (
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch_adapter.current_stream",
                return_value=7,
            ),
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch.ops.oh_my_vllm.native_copy_batch",
                side_effect=lambda inputs, outputs, stream: calls.append(
                    (inputs, outputs, stream)
                ),
            ),
        ):
            engine.submit(1, (destination,), (source,), "h2d")
            source_ref, destination_ref = weakref.ref(source), weakref.ref(destination)
            del source, destination
            gc.collect()
            self.assertIsNotNone(source_ref())
            self.assertIsNotNone(destination_ref())
            self.assertEqual(calls[0][2], 11)
            engine.close()
            calls.clear()
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(destination_ref())

    def test_metadata_uses_torch_host_arenas_and_upload_sources(self):
        from oh_my_vllm.worker.metadata import Metadata

        copied = []

        class Uploads:
            def copy(self, destination, source):
                copied.append((destination, source))
                destination.copy_(source)

            def close(self):
                pass

        device = SimpleNamespace(
            uploads=lambda: Uploads(),
            scratch=lambda shape, dtype: torch.zeros(
                shape, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
            ),
            view=lambda value, shape, offset: value.reshape(-1)[
                offset : offset + int(torch.tensor(shape).prod())
            ].reshape(shape),
        )
        values = (
            np.array([1, 2], dtype=np.int32),
            torch.tensor([3.0]),
            torch.tensor([4], dtype=torch.int64),
            torch.empty((0, 3), dtype=torch.int32),
            torch.tensor(5, dtype=torch.uint8),
        )
        for single_upload, uploads in ((False, 4), (True, 1)):
            with self.subTest(single_upload=single_upload):
                copied.clear()
                metadata = Metadata(device, values, single_upload=single_upload)
                self.assertIs(metadata.single_upload, single_upload)
                self.assertEqual(len(copied), uploads)
                self.assertTrue(
                    all(
                        isinstance(value, torch.Tensor)
                        for value in metadata.host_arenas.values()
                    )
                )
                for actual, expected in zip(metadata.arrays, values, strict=True):
                    torch.testing.assert_close(actual, torch.as_tensor(expected))
                    self.assertEqual(
                        actual.storage_offset() * actual.element_size() % 16, 0
                    )
                if single_upload:
                    self.assertEqual(set(metadata.device_arenas), {torch.uint8})
                    self.assertEqual(metadata.device_arenas[torch.uint8].numel(), 64)
                updated = tuple(torch.as_tensor(value) + 7 for value in values)
                metadata.update(updated)
                for actual, expected in zip(metadata.arrays, updated, strict=True):
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(len(copied), 2 * uploads)
                with self.assertRaisesRegex(ValueError, "shape or dtype changed"):
                    metadata.update((torch.zeros(3, dtype=torch.int32), *values[1:]))
                self.assertEqual(len(copied), 2 * uploads)
                metadata.close()
        self.assertTrue(all(isinstance(source, torch.Tensor) for _, source in copied))

    def test_qwen_metadata_upload_selection_at_all_three_call_sites(self):
        from oh_my_vllm.models.qwen38 import model, speculation

        def scratch(shape, dtype):
            return torch.zeros(
                shape, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
            )

        device = SimpleNamespace(
            uploads=lambda: SimpleNamespace(copy=lambda *_: None, close=lambda: None),
            scratch=scratch,
            empty=scratch,
            zeros=scratch,
            zero=lambda value: value.zero_(),
            upload=lambda value: value.clone(),
            view=lambda value, shape, offset=0: value.reshape(-1)[
                offset : offset + int(np.prod(shape))
            ].reshape(shape),
        )
        weights = SimpleNamespace(
            layers=(),
            mtp_input=None,
            mtp_layer=SimpleNamespace(attention=None, feed_forward=None),
            mtp_sample=None,
            attention_variant=lambda *_: None,
        )
        for count in (1, 2, 4):
            for verify, graph in ((False, True), (True, True), (True, False)):
                with self.subTest(batch=count, verify=verify, graph=graph):
                    batch = SimpleNamespace(
                        rows=count,
                        requests=[
                            SimpleNamespace(
                                state_position=2,
                                mtp_position=0,
                                history=[0, 1],
                                position=lambda value: (value,) * 3,
                            )
                            for _ in range(count)
                        ],
                        scheduled=[
                            dict(
                                request_id=index,
                                num_computed_tokens=0,
                                fa_block_table=[0],
                            )
                            for index in range(count)
                        ],
                        schedule=SimpleNamespace(
                            verify=verify,
                            sequence_rows=tuple(range(count + 1)),
                            read_slots=(0,) * count,
                        ),
                        metadata=(
                            np.zeros((3, count), np.int64),
                            np.zeros(count, np.int32),
                            np.zeros((count, 1), np.int32),
                            np.arange(count + 1, dtype=np.int32),
                            np.ones(count, np.int32),
                        ),
                        tokens=np.ones(count, np.int64),
                        writes=list(range(count)),
                    )
                    cache = SimpleNamespace(parallel=count, hidden=None, fa=(None,))
                    operation = Mock(outputs=(torch.zeros(count, 5120),) * 2)
                    with (
                        patch.object(model, "prepare_fa", return_value=operation),
                        patch.object(model, "prepare_gdn", return_value=operation),
                        patch.object(model, "prepare_ffn", return_value=operation),
                        patch.object(
                            speculation, "prepare_context", return_value=operation
                        ),
                        patch.object(
                            speculation, "prepare_draft", return_value=operation
                        ),
                        patch.object(speculation, "Sampling"),
                        patch.object(speculation.Context, "execute"),
                    ):
                        workspace = model.Workspace(
                            device, weights, cache, batch, (None,) * 4
                        )
                        owner = speculation.Speculation(
                            device,
                            weights,
                            cache,
                            SimpleNamespace(rows=lambda _: (None, None)),
                            Mock(),
                        )
                        draft = speculation.Draft(
                            device, weights, count, True, table_width=1
                        )
                        try:
                            owner.context(batch, torch.zeros(count, 5120), graph, {})
                            context = (
                                next(iter(owner.context_decode.values()))
                                if graph and verify
                                else owner.context_prefill
                            )
                            for metadata, enabled in (
                                (workspace.metadata_inputs, verify and count >= 2),
                                (
                                    context.metadata_inputs,
                                    graph and verify and count >= 2,
                                ),
                                (draft.metadata_inputs, count >= 2),
                            ):
                                self.assertIs(metadata.single_upload, enabled)
                                self.assertEqual(
                                    set(metadata.device_arenas),
                                    {torch.uint8}
                                    if enabled
                                    else {torch.int32, torch.int64},
                                )
                        finally:
                            workspace.close()
                            owner.close()
                            draft.close()

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

    def test_upload_keeps_torch_staging_owner_until_event_completion(self):
        import gc

        from oh_my_vllm.operators.runtime.uploads import UploadPool, Uploads

        allocations, pending, waits = {}, [], []
        idle = [False]

        def wait_safe():
            idle[0] = True

        def allocate(pointer, size, flags):
            storage = ctypes.create_string_buffer(size)
            address = ctypes.addressof(storage)
            allocations[address] = storage
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = address
            return 0

        def wait(event):
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
                cudaEventCreateWithFlags=lambda pointer, flags: 0,
                cudaEventDestroy=lambda _: 0,
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
        source = torch.tensor([1, 2], dtype=torch.int32)
        destination = torch.zeros(2, dtype=torch.int32)
        with (
            patch(
                "oh_my_vllm.operators.runtime.uploads.torch.ops.oh_my_vllm.native_upload",
                side_effect=lambda source, destination: pending.append(
                    (destination, source)
                ),
            ),
            patch(
                "oh_my_vllm.operators.runtime.uploads.torch_adapter.current_stream",
                return_value=7,
            ),
        ):
            uploads = Uploads(device)
            uploads.copy(destination, source)
            buffer = next(iter(uploads.buffers.values()))
            self.assertIsInstance(buffer.tensor, torch.Tensor)
            self.assertEqual(buffer.tensor.device.type, "cpu")
            del source
            gc.collect()
            uploads.close()

        torch.testing.assert_close(destination, torch.tensor([1, 2], dtype=torch.int32))
        self.assertTrue(waits)
        device.upload_pool.close()
        self.assertFalse(allocations)

    def test_range_upload_bounds_use_tensor_numel(self):
        from oh_my_vllm.operators.runtime.uploads import RangeUpload, UploadPool

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
            view=lambda value, shape, offset: value,
        )
        device.upload_pool = UploadPool(device)
        upload = RangeUpload(
            device, torch.empty(8, dtype=torch.int32, device="meta"), 32
        )
        try:
            with (
                patch(
                    "oh_my_vllm.operators.runtime.uploads.torch.ops.oh_my_vllm.native_upload",
                    side_effect=lambda *_: None,
                ),
                patch(
                    "oh_my_vllm.operators.runtime.uploads.torch_adapter.current_stream",
                    return_value=7,
                ),
            ):
                upload.copy(torch.arange(6, dtype=torch.int32).reshape(2, 3))
        finally:
            upload.close()
            device.upload_pool.close()

    def test_uploads_reject_numpy_runtime_buffers(self):
        from oh_my_vllm.operators.runtime.uploads import Uploads

        device = SimpleNamespace()
        uploads = Uploads(device)
        with self.assertRaisesRegex(ValueError, "CPU Torch"):
            uploads.copy(torch.empty(1), np.zeros(1, np.int32))

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
            side_effect=lambda *args, single_upload: Mock(
                hidden=args[4], endpoint_rows=()
            ),
        ) as context:
            try:
                for value, count in ((hidden, 1), (hidden, 1), (object(), 2)):
                    request.mtp_position = 0
                    self.assertEqual(owner.context(batch, value, True, {}), {})
                    self.assertEqual(context.call_count, count)
                    self.assertIs(context.call_args.args[-1], False)
                    self.assertIs(context.call_args.kwargs["single_upload"], False)
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

        events = []
        device = Device.__new__(Device)
        device.graph_failed = False
        device.transfer_engines = set()
        device.wait_pinned_safe = lambda: events.append("handoff")

        class Graph:
            def replay(self):
                events.append("replay")

            def reset(self):
                events.append("reset")

        @contextmanager
        def capture(*_args, **_kwargs):
            yield

        with (
            patch("torch.cuda.CUDAGraph", return_value=Graph()),
            patch("torch.cuda.graph_pool_handle", return_value="pool"),
            patch("torch.cuda.Stream", return_value=SimpleNamespace()),
            patch("torch.cuda.current_stream", return_value=SimpleNamespace()),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.graph", side_effect=capture),
            patch("oh_my_vllm.operators.runtime.device.torch_adapter.use_stream"),
            patch("torch.cuda.synchronize", side_effect=lambda: events.append("sync")),
        ):
            with device.graph(lambda: events.append("capture")) as replay:
                replay()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                replay()
        self.assertEqual(events, ["handoff", "capture", "replay", "sync", "reset"])

    def test_graph_operation_converts_host_copies_to_torch(self):
        from oh_my_vllm.models.qwen38.speculation import GraphOperation

        copied = []

        class Uploads:
            def copy(self, destination, source):
                copied.append((destination, source))

            def close(self):
                pass

        operation = GraphOperation(SimpleNamespace(uploads=lambda: Uploads()))
        source = torch.arange(4, dtype=torch.int32)
        try:
            operation.copy(object(), np.arange(4, dtype=np.int32))
            operation.copy(object(), source)
        finally:
            operation.close()
        self.assertIsInstance(copied[0][1], torch.Tensor)
        self.assertEqual(copied[0][1].device.type, "cpu")
        self.assertTrue(copied[0][1].is_contiguous())
        self.assertIs(copied[1][1], source)


if __name__ == "__main__":
    unittest.main()
