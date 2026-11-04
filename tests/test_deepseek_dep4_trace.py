import datetime
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
from oh_my_vllm.models.deepseek_v41.dep4_trace import Dep4Trace
from oh_my_vllm.models.deepseek_v41.trace import RoutingTrace, target_call_id

from tests.test_deepseek_dep4 import _Expert, _Model, _MoE


class _TraceMoE(_MoE):
    def __init__(self, rank):
        super().__init__(rank)
        with torch.inference_mode():
            self.register_buffer("scratch", torch.zeros(1))

    def _grouped_routed(self, *args):
        self.scratch.zero_()
        return None


class _TraceModel(_Model):
    def __init__(self, rank):
        super().__init__(rank)
        self.rank = rank
        for layer in self.layers:
            moe = _TraceMoE(rank)
            moe.n_routed_experts = 384
            moe.experts_start_idx = rank * 96
            moe.experts_end_idx = (rank + 1) * 96
            moe.experts = [_Expert(index) for index in range(384)]
            layer.ffn = moe

    def forward(self, input_ids, start_pos=0, *, routing_trace=None, **kwargs):
        rows = input_ids.flatten()[:, None].expand(-1, 4).to(torch.bfloat16)
        ids = torch.tensor([[0, 1, 96, 97, 192, 288]], dtype=torch.int32)
        ids = ids.expand(rows.size(0), -1).clone()
        if self.rank == 1 and start_pos == 2:
            ids[1:] = torch.arange(378, 384, dtype=torch.int32)
        weights = torch.full(ids.shape, 0.125)
        for index, layer in enumerate(self.layers):
            routing_trace.record(index, ids)
            rows = layer.ffn.dep4_exchange(layer.ffn, rows, (weights, ids)).to(
                torch.bfloat16
            )
        return rows


def _trace_peer(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/store",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = _TraceModel(rank)
        executor = Dep4Executor(model)
        request = SimpleNamespace(cache_slot=rank, prompt_len=2, state_position=0)
        output = Mock()
        worker = SimpleNamespace(
            model=model,
            rank=rank,
            requests={rank + 1: request} if rank < 2 else {},
            dep4_executor=executor,
            target_call_id_provider=lambda: executor.call,
        )
        trace = Dep4Trace(worker, output)
        executor.accounting = trace

        def legacy(start, length):
            model(
                torch.ones(1, length, dtype=torch.long), start, cache_slots_cpu=(rank,)
            )
            request.state_position = start + length
            trace.record(
                "target",
                requests=[dict(id=rank + 1, start=start, count=length, prompt=2)],
                accepted=0,
                proposed=0,
            )

        @torch.inference_mode()
        def run():
            if rank >= 2:
                return
            legacy(0, 2)
            routes = RoutingTrace(2, 6, rank=rank, strategy="dep4", device="cpu")
            model(
                torch.ones(1, 6, dtype=torch.long),
                2,
                cache_slots_cpu=(rank,),
                routing_trace=routes,
            )
            call = target_call_id(worker, -1)
            count = 6 if rank == 0 else 1
            mask = torch.arange(6) < count
            counters = routes.counters(mask)
            mask.fill_(False)
            routes.indices.fill_(383)
            if rank == 0:
                model.forward_spec(torch.ones(1, 4, dtype=torch.bfloat16), start_pos=1)
                assert executor.call == 3 and call == 2
            trace.record(
                "accounting_call",
                rank=rank,
                call=call,
                phase="decode",
                request_rows=[
                    dict(
                        id=rank + 1,
                        start=2,
                        count=count,
                        accepted=count - 1,
                        proposed=5,
                        logit_rows=count,
                    )
                ],
                **counters,
            )
            if rank == 1:
                legacy(3, 1)
            else:
                routes = RoutingTrace(2, 3, rank=rank, strategy="dep4", device="cpu")
                model(
                    torch.ones(1, 3, dtype=torch.long),
                    8,
                    cache_slots_cpu=(rank,),
                    routing_trace=routes,
                )
                call = target_call_id(worker, -1)
                counters = routes.counters(torch.tensor([True, True, False]))
                model.forward_spec(torch.ones(1, 4, dtype=torch.bfloat16), start_pos=9)
                assert executor.call == call == 4
                trace.record(
                    "accounting_call",
                    rank=rank,
                    call=call,
                    phase="decode",
                    request_rows=[
                        dict(
                            id=rank + 1,
                            start=8,
                            count=2,
                            accepted=1,
                            proposed=2,
                            logit_rows=2,
                        )
                    ],
                    **counters,
                )

        with (
            patch("torch.distributed.all_reduce", wraps=dist.all_reduce) as reduce,
            patch.object(trace, "readback", wraps=trace.readback) as readback,
        ):
            executor.execute(run)
            assert reduce.call_count == readback.call_count == 1
            assert reduce.call_args.kwargs["op"] == dist.ReduceOp.MAX
            assert reduce.call_args.args[0].shape == (4, 2, 384)
            executor.execute(lambda: None)
            assert reduce.call_count == readback.call_count == 1
        records = [
            dict(call.kwargs)
            for call in output.record.call_args_list
            if call.args == ("accounting_call",)
        ]
        Path(directory, f"rank{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


class DeepSeekDep4Trace(unittest.TestCase):
    def test_dep4_moe_records_routes_once_per_layer_and_call(self):
        from oh_my_vllm.models.deepseek_v41.model import ModelArgs, MoE

        with (
            patch("oh_my_vllm.models.deepseek_v41.model.world_size", 1),
            patch("oh_my_vllm.models.deepseek_v41.model.expert_world_size", 4),
            patch("oh_my_vllm.models.deepseek_v41.model.default_dtype", torch.bfloat16),
        ):
            moe = MoE(
                0,
                ModelArgs(
                    dim=4, moe_inter_dim=4, expert_dtype=None, parallel_strategy="DEP4"
                ),
            )
        moe.shared_experts = torch.nn.Identity()
        ids = torch.arange(6, dtype=torch.int32).expand(2, -1)
        weights = torch.ones(2, 6)
        moe._routing_trace = SimpleNamespace(record=Mock())
        moe.dep4_exchange = Mock(return_value=torch.zeros(2, 4))
        with patch.object(moe.gate, "forward", return_value=(weights, ids)):
            for _call in range(2):
                moe._routing_trace.record.reset_mock()
                moe(torch.ones(2, 1, 4))
                moe._routing_trace.record.assert_called_once_with(0, ids)

    def test_empty_peer_keeps_inference_context_until_counter_flush(self):
        with torch.inference_mode():
            scratch = torch.zeros(1)
        executor = Dep4Executor(_Model(0))
        executor.exchange = Mock(side_effect=lambda *_args: scratch.zero_())
        executor.accounting = SimpleNamespace(
            begin=Mock(), flush=lambda: scratch.zero_()
        )
        with patch.object(
            executor, "_pending", side_effect=[(1, 1, 1, False), (0, 4, 0, False)]
        ):
            executor.execute(torch.inference_mode()(lambda: None))

    def test_interleaved_proposal_and_next_target_flush_once_with_empty_peers(self):
        """Check CPU call alignment and masks, without checkpoint or Graph execution."""
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_trace_peer, args=(directory,), nprocs=4, join=True)
            ranks = [
                json.loads(Path(directory, f"rank{rank}.json").read_text())
                for rank in range(4)
            ]
        for rank, records in enumerate(ranks):
            self.assertEqual([row["call"] for row in records], [1, 2, 3, 4])
            self.assertEqual(
                [row["phase"] for row in records],
                ["prefill", "decode", "decode", "decode"],
            )
            for record in records:
                self.assertEqual(
                    record["touched_experts"], [2, 2] if rank < 2 else [1, 1]
                )
                if (
                    rank >= 2
                    or (rank == 0 and record["call"] == 3)
                    or (rank == 1 and record["call"] == 4)
                ):
                    self.assertEqual(record["request_rows"], [])
                    self.assertEqual(record["expert_assignments"], [[0] * 4] * 2)
                    self.assertEqual(record["dispatch_rows"], [[0] * 4] * 2)
        self.assertEqual(ranks[0][1]["expert_assignments"], [[12, 12, 6, 6]] * 2)
        self.assertEqual(ranks[1][1]["expert_assignments"], [[2, 2, 1, 1]] * 2)
        self.assertEqual(ranks[0][1]["dispatch_rows"], [[6] * 4] * 2)
        self.assertEqual(ranks[1][1]["dispatch_rows"], [[1] * 4] * 2)
        self.assertEqual(ranks[0][3]["expert_assignments"], [[4, 4, 2, 2]] * 2)
        self.assertEqual(ranks[0][3]["dispatch_rows"], [[2] * 4] * 2)
        self.assertEqual(
            ranks[0][3]["request_rows"],
            [dict(id=1, start=8, count=2, accepted=1, proposed=2, logit_rows=2)],
        )

    def test_legacy_chunks_rejection_and_restart_do_not_credit_recomputed_rows(self):
        request = SimpleNamespace(cache_slot=0, prompt_len=4, state_position=4)
        worker = SimpleNamespace(
            model=SimpleNamespace(
                layers=[None, None], parameters=lambda: iter([torch.zeros(1)])
            ),
            requests={1: request},
            dep4_executor=SimpleNamespace(group=None),
            rank=0,
        )
        output = Mock()
        trace = Dep4Trace(worker, output)

        def capture(call, start, length):
            phase, _options, payload = trace.prepare(
                (torch.ones(1, length, dtype=torch.long), start),
                dict(cache_slots_cpu=(0,)),
            )
            trace.begin(call, phase)
            ids = (
                torch.tensor([[0, 1, 96, 97, 192, 288]], dtype=torch.int32)
                .expand(length, -1)
                .clone()
            )
            ids[1:] = torch.arange(378, 384, dtype=torch.int32)
            for layer in range(2):
                payload[1].record(layer, ids)
            trace.capture(call, payload)

        def finish(start, length, proposed=0):
            trace.record(
                "target",
                requests=[dict(id=1, start=start, count=length, prompt=4)],
                proposed=proposed,
            )
            with patch("torch.distributed.all_reduce"):
                trace.flush()

        capture(1, 0, 2)
        capture(2, 2, 2)
        finish(0, 4)
        records = [
            call.kwargs
            for call in output.record.call_args_list
            if call.args == ("accounting_call",)
        ]
        self.assertEqual(
            [record["request_rows"][0]["logit_rows"] for record in records], [0, 1]
        )
        request.state_position = 5
        capture(3, 4, 6)
        finish(4, 6, proposed=5)
        rejected = output.record.call_args.kwargs
        self.assertEqual(rejected["expert_assignments"], [[2, 2, 1, 1]] * 2)
        self.assertEqual(rejected["request_rows"][0]["count"], 1)
        capture(4, 0, 4)
        finish(0, 4)
        self.assertEqual(output.record.call_args.kwargs["request_rows"], [])
        self.assertEqual(
            output.record.call_args.kwargs["expert_assignments"], [[0] * 4] * 2
        )
        self.assertEqual(trace.frontiers[1], 5)
        request.state_position = 6
        capture(5, 0, 6)
        finish(0, 6)
        resumed = output.record.call_args.kwargs
        self.assertEqual(resumed["phase"], "decode")
        self.assertEqual(
            resumed["request_rows"],
            [dict(id=1, start=5, count=1, accepted=0, proposed=0, logit_rows=1)],
        )
        self.assertEqual(resumed["expert_assignments"], [[0, 0, 0, 6]] * 2)
        self.assertEqual(trace.frontiers[1], 6)
        worker.requests[2] = SimpleNamespace(cache_slot=1, prompt_len=8)
        phase, _options, _payload = trace.prepare(
            (torch.ones(2, 4, dtype=torch.long), 0),
            dict(cache_slots_cpu=(0, 1)),
        )
        self.assertEqual(phase, "prefill")
        phase, _options, _payload = trace.prepare(
            (torch.ones(1, 4, dtype=torch.long), 0),
            dict(cache_slots_cpu=(0,)),
        )
        self.assertEqual(phase, "prefill")

    def test_flush_rejects_wrong_accepted_mask_and_missing_layers(self):
        for wrong_mask in (False, True):
            with self.subTest(wrong_mask=wrong_mask):
                request = SimpleNamespace(cache_slot=0, prompt_len=0)
                worker = SimpleNamespace(
                    model=SimpleNamespace(
                        layers=[None, None], parameters=lambda: iter([torch.zeros(1)])
                    ),
                    requests={1: request},
                    dep4_executor=SimpleNamespace(group=None),
                    rank=0,
                )
                trace = Dep4Trace(worker, Mock())
                routes = RoutingTrace(2, 2, rank=0, strategy="dep4", device="cpu")
                _phase, _options, payload = trace.prepare(
                    (torch.ones(1, 2, dtype=torch.long),),
                    dict(cache_slots_cpu=(0,), routing_trace=routes),
                )
                trace.begin(1, "decode")
                ids = torch.tensor(
                    [[0, 1, 96, 97, 192, 288]], dtype=torch.int32
                ).expand(2, -1)
                routes.record(0, ids)
                if wrong_mask:
                    routes.record(1, ids)
                trace.capture(1, payload)
                values = dict(
                    rank=0,
                    call=1,
                    phase="decode",
                    request_rows=[
                        dict(
                            id=1, start=0, count=1, accepted=0, proposed=1, logit_rows=1
                        )
                    ],
                    **routes.counters(torch.tensor([not wrong_mask, wrong_mask])),
                )
                trace.record("accounting_call", **values)
                with self.assertRaisesRegex(ValueError, "missing or duplicated"):
                    trace.record("accounting_call", **values)
                with (
                    patch("torch.distributed.all_reduce"),
                    self.assertRaisesRegex(
                        ValueError, "incomplete routing layers or accepted rows"
                    ),
                ):
                    trace.flush()

    def test_unregistered_and_mixed_phase_calls_are_rejected(self):
        worker = SimpleNamespace(
            requests={
                1: SimpleNamespace(cache_slot=0, prompt_len=4),
                2: SimpleNamespace(cache_slot=1, prompt_len=2),
            },
            model=SimpleNamespace(layers=[None]),
            rank=0,
        )
        trace = Dep4Trace(worker, Mock())
        for slots, positions, length in (
            ((2,), (0,), 1),
            ((0, 1), (0, 2), 1),
            ((0,), (3,), 2),
        ):
            with (
                self.subTest(slots=slots, positions=positions),
                self.assertRaises(ValueError),
            ):
                trace.prepare(
                    (torch.ones(len(slots), length, dtype=torch.long),),
                    dict(cache_slots_cpu=slots, cache_positions_cpu=positions),
                )
