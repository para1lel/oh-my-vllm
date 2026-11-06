import gc
import sys
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.cache import Cache
from oh_my_vllm.operators.runtime.device import Device

from tests.qwen_radix_physical_fixtures import Device as CacheDevice
from tests.qwen_radix_physical_fixtures import config, reference, row


class GraphOwner:
    def __init__(self, events):
        self.events = events

    def run(self):
        self.events.append("operation")

    @contextmanager
    def graph_lease(self):
        self.events.append("lease")
        try:
            yield
        finally:
            self.events.append("release")


class GraphCapture(unittest.TestCase):
    def device(self, events):
        device = Device.__new__(Device)
        device.graph_failed = False
        device.capture_stream = None
        device.transfer_engines = set()
        device._failed_graph_resources = []
        device.wait_pinned_safe = lambda: events.append("handoff")
        return device

    @contextmanager
    def fake_capture(self, *_args, **_kwargs):
        yield

    def patches(self, events, graph=None, *, capturing=False, synchronize=None):
        graph = graph or SimpleNamespace(
            replay=lambda: events.append("replay"),
            reset=lambda: events.append("reset"),
        )
        stack = ExitStack()
        for context in (
            patch("torch.cuda.CUDAGraph", return_value=graph),
            patch("torch.cuda.graph_pool_handle", return_value="pool"),
            patch(
                "torch.cuda.Stream",
                return_value=SimpleNamespace(cuda_stream=77),
            ),
            patch(
                "torch.cuda.current_stream",
                return_value=SimpleNamespace(cuda_stream=77),
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
            patch("torch.cuda.graph", side_effect=self.fake_capture),
            patch("oh_my_vllm.operators.runtime.device.torch_adapter.use_stream"),
            patch("torch.cuda.synchronize", side_effect=synchronize),
        ):
            stack.enter_context(context)
        return stack

    def test_graph_uses_torch_capture_and_replay(self):
        events = []
        device = self.device(events)
        graph = SimpleNamespace(
            replay=lambda: events.append("replay"),
            reset=lambda: events.append("reset"),
        )
        with (
            self.patches(
                events, graph=graph, synchronize=lambda: events.append("sync")
            ),
            device.graph(lambda: events.append("capture")) as replay,
        ):
            replay()
        self.assertEqual(events, ["handoff", "capture", "replay", "sync", "reset"])

    def test_graph_capture_uses_one_non_default_stream_and_restores_it(self):
        events = []
        device = self.device(events)
        capture_stream = SimpleNamespace(cuda_stream=88)
        graph_streams = []

        @contextmanager
        def capture(*_, stream=None, **__):
            graph_streams.append(stream)
            events.append("capture-enter")
            try:
                yield
            finally:
                events.append("capture-exit")

        @contextmanager
        def use_stream(*_, **__):
            events.append("stream-enter")
            try:
                yield
            finally:
                events.append("stream-exit")

        with (
            self.patches(events),
            patch("torch.cuda.Stream", return_value=capture_stream) as make_stream,
            patch("torch.cuda.graph", side_effect=capture),
            patch(
                "oh_my_vllm.operators.runtime.device.torch_adapter.use_stream",
                side_effect=use_stream,
            ),
        ):
            device.capture(lambda: events.append("capture"))
            device.capture(lambda: events.append("capture"))
        self.assertEqual(make_stream.call_count, 1)
        self.assertEqual(graph_streams, [capture_stream, capture_stream])
        self.assertEqual(events.count("stream-enter"), 2)
        self.assertEqual(events.count("stream-exit"), 2)
        self.assertEqual(events.count("capture-exit"), 2)

    def test_graph_capture_failure_restores_the_caller_stream(self):
        events = []
        device = self.device(events)
        capture_stream = SimpleNamespace(cuda_stream=88)

        @contextmanager
        def capture(*_, stream=None, **__):
            events.append("capture-enter")
            try:
                yield
            except BaseException:
                events.append("capture-error")
                raise
            finally:
                events.append("capture-exit")

        @contextmanager
        def use_stream(*_, **__):
            events.append("stream-enter")
            try:
                yield
            finally:
                events.append("stream-exit")

        with (
            self.patches(events),
            patch("torch.cuda.Stream", return_value=capture_stream),
            patch("torch.cuda.graph", side_effect=capture),
            patch(
                "oh_my_vllm.operators.runtime.device.torch_adapter.use_stream",
                side_effect=use_stream,
            ),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            device.capture(
                lambda: (_ for _ in ()).throw(RuntimeError("capture failed"))
            )
        self.assertIn("capture-error", events)
        self.assertIn("stream-exit", events)
        self.assertIn("capture-exit", events)
        self.assertIs(device.capture_stream, capture_stream)
        self.assertFalse(device.graph_failed)

    def test_graph_rejects_an_active_torch_capture(self):
        events = []
        device = self.device(events)
        with (
            self.patches(events, capturing=True),
            self.assertRaisesRegex(RuntimeError, "already capturing"),
        ):
            device.capture(lambda: events.append("capture"))
        self.assertEqual(events, [])

    def test_graph_releases_owner_after_torch_graph_cleanup(self):
        events = []
        device = self.device(events)
        owner = GraphOwner(events)
        with self.patches(events, synchronize=lambda: None), device.graph(owner.run):
            self.assertEqual(events, ["lease", "handoff", "operation"])
        self.assertEqual(events, ["lease", "handoff", "operation", "reset", "release"])

    def test_graph_keeps_callable_and_input_owner_without_an_owning_lease(self):
        class Owner:
            def __init__(self):
                self.input = torch.ones(1)

        class Operation:
            def __init__(self, owner, events):
                self.owner = weakref.ref(owner)
                self.input = torch.zeros(1)
                self.events = events

            @property
            def __self__(self):
                return self.owner()

            def __call__(self):
                self.events.append("operation")

        for forwarded in (False, True):
            for failure in (None, "sync", "reset"):
                with self.subTest(forwarded=forwarded, failure=failure):
                    events = []
                    device = self.device(events)
                    owner = Owner()
                    if forwarded:
                        owner.graph_lease = GraphOwner(events).graph_lease
                    operation = Operation(owner, events)
                    references = tuple(
                        weakref.ref(value)
                        for value in (owner, owner.input, operation, operation.input)
                    )

                    def synchronize(failure=failure):
                        if failure == "sync":
                            raise RuntimeError("sync failed")

                    def reset(failure=failure, references=references):
                        self.assertTrue(
                            all(reference() is not None for reference in references)
                        )
                        if failure == "reset":
                            raise RuntimeError("reset failed")

                    graph = SimpleNamespace(replay=lambda: None, reset=reset)
                    with self.patches(events, graph=graph, synchronize=synchronize):
                        context = device.graph(operation)
                        context.__enter__()
                        del owner, operation
                        gc.collect()
                        self.assertTrue(
                            all(reference() is not None for reference in references)
                        )
                        if failure is None:
                            context.__exit__(None, None, None)
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, f"{failure} failed"
                            ):
                                context.__exit__(None, None, None)
                        del context
                        gc.collect()
                        if failure is None:
                            self.assertTrue(
                                all(reference() is None for reference in references)
                            )
                            self.assertEqual(device._failed_graph_resources, [])
                            if forwarded:
                                self.assertIn("release", events)
                        else:
                            self.assertTrue(
                                all(reference() is not None for reference in references)
                            )
                            self.assertEqual(len(device._failed_graph_resources), 1)
                            self.assertTrue(device.graph_failed)
                            self.assertNotIn("release", events)

    def test_graph_cleanup_failure_poison_the_device_and_retains_owners(self):
        events = []
        device = self.device(events)
        owner = GraphOwner(events)
        graph = SimpleNamespace(replay=lambda: events.append("replay"))
        with (
            self.patches(
                events,
                graph=graph,
                synchronize=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
            ),
            self.assertRaisesRegex(RuntimeError, "sync failed"),
            device.graph(owner.run),
        ):
            pass
        self.assertTrue(device.graph_failed)
        self.assertIs(device._failed_graph_resources[0][0], graph)
        self.assertIsNotNone(device._failed_graph_resources[0][1])
        self.assertNotIn("release", events)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            device.capture(lambda: None)

    def test_graph_reset_failure_poisons_the_device_and_retains_owners(self):
        events = []
        device = self.device(events)
        owner = GraphOwner(events)
        graph = SimpleNamespace(
            replay=lambda: events.append("replay"),
            reset=lambda: (_ for _ in ()).throw(RuntimeError("reset failed")),
        )
        with (
            self.patches(
                events,
                graph=graph,
                synchronize=lambda: events.append("sync"),
            ),
            self.assertRaisesRegex(RuntimeError, "reset failed"),
            device.graph(owner.run),
        ):
            pass
        self.assertTrue(device.graph_failed)
        self.assertIs(device._failed_graph_resources[0][0], graph)
        self.assertIsNotNone(device._failed_graph_resources[0][1])
        self.assertNotIn("release", events)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            device.capture(lambda: None)

    def test_decode_workspace_cache_closes_oldest_graph_owner(self):
        from oh_my_vllm.models.qwen38 import model

        closed = []
        live = []
        peak = [0]

        def initialize(workspace, _, __, ___, batch, ____):
            workspace.output = object()
            workspace.hidden = None
            workspace.runs = 0
            workspace.key = batch.rows
            live.append(workspace.key)
            peak[0] = max(peak[0], len(live))

        def close(workspace):
            closed.append(workspace)
            live.remove(workspace.key)

        def batch(rows):
            return SimpleNamespace(
                rows=rows,
                schedule=SimpleNamespace(verify=True),
                scheduled=[dict(request_id=str(rows), token_ids=[1])],
                requests=[SimpleNamespace(media=None)],
            )

        owner = model.Model(
            SimpleNamespace(),
            None,
            None,
            False,
            SimpleNamespace(rows=lambda _: (None, None)),
        )
        with (
            patch.object(model, "MAX_GRAPH_WORKSPACES", 2),
            patch.object(
                model,
                "make_plans",
                side_effect=lambda batch, _: (SimpleNamespace(reuse_key=batch.rows),),
            ),
            patch.object(model.Workspace, "__init__", initialize),
            patch.object(model.Workspace, "close", close),
            patch.object(model.Workspace, "update", lambda *_: None),
            patch.object(model.Workspace, "bind_media", lambda *_: None),
            patch.object(
                model.Workspace,
                "run",
                lambda workspace, graph: workspace.output,
            ),
        ):
            for rows in (0, 1, 0, 2):
                owner.forward(batch(rows), False)
                self.assertLessEqual(len(live), 2)
            self.assertEqual(len(owner.decode), 2)
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0].key, 1)
            self.assertEqual(peak[0], 2)
            owner.close()
            self.assertEqual(sorted(workspace.key for workspace in closed), [0, 1, 2])

    def test_speculation_context_cache_closes_oldest_graph_owner(self):
        fake_guidance = SimpleNamespace(
            LLExecutor=lambda *_: None,
            LLMatcher=SimpleNamespace(grammar_from_json_schema=lambda *_: None),
        )
        with patch.dict(sys.modules, {"llguidance": fake_guidance}):
            from oh_my_vllm.models.qwen38 import speculation

            request = SimpleNamespace(
                state_position=2,
                mtp_position=0,
                history=[0, 1],
                finish=False,
                generated=0,
                limit=10,
                position=lambda value: (value,) * 3,
            )
            cache = Cache(
                CacheDevice(),
                config(mamba_blocks=4, num_gpu_blocks=256, num_speculative_tokens=4),
            )
            item = row(
                end=2,
                sampling="verify",
                exports=[
                    {**reference(end, slot), "role": "private_endpoint"}
                    for end, slot in ((1, 5), (2, 6))
                ],
            )
            item.update(request_id="request", token_ids=request.history.copy())
            item["sampling"]["drafts"] = 1
            batch = cache.batch([item], [request], True)
            cache.complete_exports(batch)
            cache.commit(request, batch, 0, None)
            closed = []
            live = []
            peak = [0]
            plan_keys = iter((0, 1, 0, 2))

            def make_context(*args, single_upload):
                key = args[3].reuse_key
                live.append(key)
                peak[0] = max(peak[0], len(live))
                return SimpleNamespace(
                    copy=lambda *_: None,
                    hidden=args[4],
                    endpoint_rows=(object(),),
                    execute=lambda *_: None,
                    metadata_inputs=SimpleNamespace(update=lambda *_: None),
                    close=lambda: (closed.append(key), live.remove(key)),
                )

            owner = speculation.Speculation(
                SimpleNamespace(view=lambda *_: object()),
                None,
                cache,
                SimpleNamespace(rows=lambda _: (None, None)),
                SimpleNamespace(record=lambda *_, **__: None),
            )
            with (
                patch.object(speculation, "MAX_GRAPH_WORKSPACES", 2),
                patch.object(
                    speculation,
                    "context_plan",
                    side_effect=lambda **_: SimpleNamespace(
                        reuse_key=next(plan_keys), batch=1
                    ),
                ),
                patch.object(speculation, "Context", side_effect=make_context),
            ):
                for _ in range(4):
                    request.mtp_position = 0
                    owner.context(batch, np.zeros((1, 1), np.float32), True, set())
                    self.assertEqual(request.mtp_position, request.state_position - 1)
                    self.assertLessEqual(len(live), 2)
                self.assertEqual(len(owner.context_decode), 2)
                self.assertEqual(closed, [1])
                self.assertEqual(peak[0], 2)
                owner.close()
                self.assertEqual(sorted(closed), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
