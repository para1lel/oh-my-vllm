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
        from tests.qwen_radix_physical_fixtures import (
            execute_message,
            request,
            row,
            worker,
        )

        owner = worker(request())
        decode = row(
            16,
            21,
            sampling="verify",
            deepest=16,
            contiguous=16,
            resume={"end": 16, "slot": 1, "version": 1},
        )
        decode["sampling"]["drafts"] = 4
        pages = [
            {
                "page_index": index,
                "slot": index + 1,
                "version": 1,
                "valid_tokens": 0,
                "access": "write",
            }
            for index in range(157)
        ]
        first = row(end=20000, pages=pages)
        second = dict(first, request_id=19)
        first["request_id"] = 18
        owner.requests.update({18: request(), 19: request()})
        for step, (rows, expected) in enumerate(
            (
                ([decode], [(True, True)]),
                ([first, second], [(False, None), (False, None)]),
                ([decode, first, second], [(True, None), (False, None), (False, None)]),
            ),
            1,
        ):
            calls = []

            def batch(items, verify, graph, calls=calls):
                calls.append((verify, graph))
                return [dict(request_id=item["request_id"]) for item in items]

            owner.execute_batch = batch
            message = execute_message(rows[0])
            message.update(scheduled=rows, step_id=step)
            result = owner.execute(message)
            self.assertEqual(calls, expected)
            self.assertEqual(result["rpc_id"], message["rpc_id"])
            self.assertEqual(
                {item["request_id"] for item in result["outputs"]},
                {item["request_id"] for item in rows},
            )

    def test_execute_rejects_a_poisoned_device_before_model_work(self):
        worker = Worker.__new__(Worker)
        calls = []

        def reject():
            calls.append("health")
            raise RuntimeError("CUDA Graph device is unusable")

        worker.device = SimpleNamespace(ensure_healthy=reject)
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

    def test_packing_preserves_scheduler_token_ranges_and_exports(self):
        from tests.qwen_radix_physical_fixtures import config

        worker = Worker.__new__(Worker)
        for limit, mtp in ((32764, True), (32768, False), (783, False)):
            worker.config = config(
                max_num_batched_tokens=limit, num_speculative_tokens=4 if mtp else 0
            )
            worker.speculation = object() if mtp else None
            rows = [
                dict(
                    request_id=index,
                    forward_start=317,
                    forward_end=317 + count,
                    token_ids=list(range(count)),
                    model_state=object(),
                )
                for index, count in enumerate((limit // 2, limit // 3, limit // 2, 1))
            ]
            groups = list(worker.batches(rows))
            restored = [item for group in groups for item in group]
            self.assertEqual(restored, rows)
            for actual, original in zip(restored, rows, strict=True):
                self.assertIs(actual, original)
            self.assertTrue(
                all(
                    sum(len(item["token_ids"]) for item in group) <= limit
                    for group in groups
                )
            )
            with self.assertRaises(ValueError):
                list(worker.batches([dict(token_ids=[0] * (limit + 1))]))

    def test_stop_truncation_controls_accepted_slot(self):
        from tests.qwen_radix_physical_fixtures import (
            execute_message,
            reference,
            request,
            row,
            worker,
        )

        for emitted in (1, 2, 0):
            state = request(history=list(range(101)), finish=None)
            owner = worker(state)
            owner.execute(execute_message(row(end=100)))

            def append(tokens, count=emitted, state=state):
                kept = tokens[:count]
                state.history.extend(kept)
                state.finish = "stop"
                return kept, ""

            state.append = append
            item = row(
                100,
                105,
                sampling="verify",
                deepest=100,
                contiguous=100,
                resume=reference(100, 5),
                exports=[
                    {**reference(end, slot), "role": "private_endpoint"}
                    for end, slot in zip(range(101, 106), range(6, 11), strict=True)
                ],
            )
            item["sampling"]["drafts"] = 4
            owner.sampler = lambda *_: None
            owner.mtp_endpoint = lambda *_: None
            owner.speculation = SimpleNamespace(
                accept=lambda *_: [[2, 3, 4, 5]],
                context=lambda *_: {},
                propose=lambda *_: None,
            )
            if not emitted:
                with self.assertRaisesRegex(ValueError, "no accepted output"):
                    owner.execute_batch([item], True, True)
                self.assertEqual(state.state_position, 100)
            else:
                result = owner.execute_batch([item], True, True)
                self.assertEqual(result[0]["token_ids"], [2, 3][:emitted])
                self.assertEqual(state.state_slot, 5 + emitted)
                self.assertEqual(state.state_position, 100 + emitted)

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
            scheduled=[dict(request_id="prefill", forward_start=0)],
            schedule=SimpleNamespace(
                verify=False, sequence_rows=(0, 2), read_slots=(0,)
            ),
            metadata=(None, None, np.ones((1, 1), np.int32)),
            plans=(
                SimpleNamespace(
                    end=2,
                    write_mask=(True, True),
                    pages=({"slot": 1, "access": "write"},),
                ),
            ),
        )
        cache = SimpleNamespace(
            layout=SimpleNamespace(max_context=128), validate_context=Mock()
        )
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
                scheduled=[dict(forward_start=0, token_ids=[1])],
                requests=[SimpleNamespace(media=media)],
            )
            result = owner.forward(batch, graph)
            self.assertEqual(len(result), 3, "forward must report actual Graph use")
            workspace = next(iter(owner.decode.values())) if verify else owner.prefill
            self.assertIs(result[0], workspace.output)
            self.assertIs(result[1], workspace.hidden)
            return result[2]

        return events, fail, prepare, owner, call
