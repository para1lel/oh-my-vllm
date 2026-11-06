import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.inference import Worker
from oh_my_vllm.operators.runtime.device import Device


class ExecutionPolicy(unittest.TestCase):
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

        events, fail, prepare, owner, call = self._adaptive_prefill_fixture(model)

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

        self._check_prefill_context_reuse()

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
        from oh_my_vllm.models.qwen38.speculation_ops import GraphOperation

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
        device.capture_stream = None
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
        from oh_my_vllm.models.qwen38.speculation_ops import GraphOperation

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

    def _check_prefill_context_reuse(self):
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
                    self.assertIs(context.call_args.kwargs["single_upload"], True)
                    self.assertIs(owner.context_prefill.hidden, value)
                    owner.context_prefill.copy.assert_not_called()
                    self.assertFalse(owner.context_decode)
            finally:
                owner.close()

    def _adaptive_prefill_fixture(self, model):
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

        return events, fail, prepare, owner, call
