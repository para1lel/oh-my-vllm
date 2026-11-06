import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import model_context
from oh_my_vllm.models.deepseek_v41.trace import (
    RoutingTrace,
)


class DeepSeekTraceRouting(unittest.TestCase):
    def test_moe_paths_record_routes_once_with_nested_dispatch(self):
        from oh_my_vllm.models.deepseek_v41 import model
        from oh_my_vllm.models.deepseek_v41.dispatch import local_expert_sum

        for strategy in ("TP4", "TEP4", "DEP4"):
            for shape in ((1, 1, 4), (2, 2, 4)):
                for path in ("ordinary", "rows_hook", "local_hook", "bound_hook"):
                    with (
                        self.subTest(strategy=strategy, shape=shape, path=path),
                        patch.object(model_context, "world_size", 1),
                        patch.object(model_context, "expert_world_size", 1),
                        patch.object(model_context, "default_dtype", torch.bfloat16),
                    ):
                        moe, rows, count, ids, trace = self._moe_trace_fixture(
                            model, strategy, shape, local_expert_sum
                        )
                        options = {"logical_rows": 1} if shape[0] > 1 else {}
                        if path != "ordinary":

                            def hook(
                                *, layer, rows, routing, valid_rows, plan, path=path
                            ):
                                if path == "local_hook":
                                    return local_expert_sum(
                                        layer, rows, routing[1], routing[0]
                                    ) + layer.shared_experts(rows)
                                return layer._forward_rows(rows, routing=routing)

                            if path == "bound_hook":
                                moe.dispatch_combine = hook
                            else:
                                options["dispatch_combine"] = hook
                        with patch.object(
                            trace, "record", wraps=trace.record
                        ) as record:
                            output = moe(rows, **options)
                            record.assert_called_once()
                            self.assertEqual(record.call_args.args[0], 0)
                            self.assertTrue(torch.equal(record.call_args.args[1], ids))
                        self.assertIs(moe._routing_trace, trace)
                        self.assertTrue(torch.equal(output, torch.zeros_like(rows)))
                        mask = torch.arange(count).remainder(2) == 0
                        credited = int(mask.sum())
                        counters = trace.counters(
                            mask, expert_union=lambda _presence: None
                        )
                        self.assertEqual(
                            counters,
                            {
                                "expert_assignments": [
                                    [credited * 2, credited * 2, credited, credited]
                                ],
                                "dispatch_rows": [[credited] * 4]
                                if strategy == "DEP4"
                                else [[0] * 4],
                                "touched_experts": [6 if strategy == "TP4" else 2],
                            },
                        )

    def test_hook_failure_restores_the_routing_trace_for_the_next_call(self):
        from oh_my_vllm.models.deepseek_v41.model import MoE

        moe = MoE.__new__(MoE)
        torch.nn.Module.__init__(moe)
        moe.dim, moe.layer_id, moe.data_parallel = 4, 0, False
        routing = (torch.ones(1, 6), torch.arange(6).view(1, 6))
        moe.gate = Mock(return_value=routing)
        trace = SimpleNamespace(record=Mock())
        moe._routing_trace = trace

        def fail(**_options):
            self.assertIsNone(moe._routing_trace)
            raise RuntimeError("dispatch failed")

        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            moe(torch.ones(1, 1, 4), dispatch_combine=fail)
        self.assertIs(moe._routing_trace, trace)
        trace.record.reset_mock()
        moe(torch.ones(1, 1, 4), dispatch_combine=lambda **options: options["rows"])
        trace.record.assert_called_once()

    def test_trace_rejects_invalid_shapes_before_writing(self):
        trace = RoutingTrace(2, 3, rank=0, strategy="tp4", device="cpu")
        for layer, ids, first in (
            (2, torch.zeros(1, 6), 0),
            (0, torch.zeros(1, 5), 0),
            (0, torch.zeros(2, 6), 2),
        ):
            with self.subTest(layer=layer, first=first), self.assertRaises(ValueError):
                trace.record(layer, ids, first)
        with self.assertRaises(ValueError):
            trace.counters(torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            trace.counters(torch.ones(3, dtype=torch.int32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_graph_replay_writes_new_routes_into_fixed_storage(self):
        trace = RoutingTrace(1, 1, rank=1, strategy="tep4", device="cuda")
        ids = torch.tensor([[0, 1, 96, 97, 192, 288]], dtype=torch.int32, device="cuda")
        mask = torch.ones(1, dtype=torch.bool, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            trace.record(0, ids)
        ids.copy_(
            torch.tensor([[96, 97, 98, 99, 100, 101]], dtype=torch.int32, device="cuda")
        )
        graph.replay()
        self.assertEqual(
            trace.counters(mask),
            {
                "expert_assignments": [[0, 6, 0, 0]],
                "dispatch_rows": [[0, 0, 0, 0]],
                "touched_experts": [6],
            },
        )

    def _moe_trace_fixture(self, model, strategy, shape, local_expert_sum):
        moe = model.MoE(
            0,
            model.ModelArgs(
                dim=4,
                moe_inter_dim=4,
                n_routed_experts=384,
                n_activated_experts=6,
                expert_dtype=None,
                parallel_strategy=strategy,
            ),
        )
        for parameter in moe.parameters():
            parameter.data.zero_()
        rows = torch.ones(shape, dtype=torch.bfloat16)
        count = rows.numel() // 4
        ids = (
            torch.tensor([0, 1, 96, 97, 192, 288], dtype=torch.int32)
            .expand(count, -1)
            .clone()
        )
        ids[1::2] = torch.arange(378, 384, dtype=torch.int32)
        weights = torch.ones_like(ids, dtype=torch.float32)
        moe.gate.forward = Mock(return_value=(weights, ids))
        moe.dep4_exchange = lambda layer, rows, routing: local_expert_sum(
            layer, rows, routing[1], routing[0]
        )
        trace = RoutingTrace(1, count, rank=0, strategy=strategy.lower(), device="cpu")
        moe._routing_trace = trace
        return moe, rows, count, ids, trace
