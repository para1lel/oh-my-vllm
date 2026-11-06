import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.trace import (
    NATIVE_MODULES,
    RoutingTrace,
    TargetTrace,
    rank_trace_path,
    record_accounting_call,
    runtime_identity,
    target_call_id,
)

from tests.deepseek_speculation_fixtures import _wire_message
from tests.deepseek_trace_routing import DeepSeekTraceRouting as DeepSeekTraceRouting
from tests.test_deepseek_speculation import _chain_row, _chain_worker, _worker


class DeepSeekTrace(unittest.TestCase):
    def test_continuous_verification_credits_old_drafts_and_advances_one_target_call(
        self,
    ):
        worker = _chain_worker(acceptance=2, limit=10)
        worker.rank, worker.strategy = 0, "tp4"
        worker.routing_trace_enabled = True
        request = worker.requests[1]
        output = worker._execute_row(_chain_row(worker, request.history))
        proposed = [len(output["new_draft_token_ids"])]
        while not request.finish:
            output = worker._execute_row(
                _chain_row(
                    worker, [request.history[-1], *output["new_draft_token_ids"]]
                )
            )
            proposed.append(len(output["new_draft_token_ids"]))
        records = [
            call.kwargs
            for call in worker.trace.record.call_args_list
            if call.args == ("accounting_call",)
        ]
        self.assertEqual(proposed, [5, 5, 2, 0])
        self.assertEqual([record["call"] for record in records], [1, 2, 3, 4])
        self.assertEqual(
            [record["phase"] for record in records],
            ["prefill", "decode", "decode", "decode"],
        )
        self.assertEqual(
            [record["request_rows"] for record in records[1:]],
            [
                [dict(id=1, start=3, count=3, accepted=2, proposed=5, logit_rows=3)],
                [dict(id=1, start=6, count=3, accepted=2, proposed=5, logit_rows=3)],
                [dict(id=1, start=9, count=3, accepted=2, proposed=2, logit_rows=3)],
            ],
        )
        self.assertEqual(
            [record["expert_assignments"] for record in records], [[[18, 0, 0, 0]]] * 4
        )
        self.assertEqual(worker.target_trace_frontiers, {1: 12})
        self.assertEqual(len(worker.model.targets), 4)
        self.assertEqual(len(worker.model.proposals), 3)
        self.assertFalse(
            any(
                call.args == ("accounting_error",)
                for call in worker.trace.record.call_args_list
            )
        )

    def test_rank_paths_keep_rank_zero_and_separate_other_writers(self):
        self.assertEqual(rank_trace_path("/tmp/trace.jsonl", 0), "/tmp/trace.jsonl")
        self.assertEqual(
            rank_trace_path("/tmp/trace.jsonl", 2), "/tmp/trace-rank2.jsonl"
        )
        self.assertEqual(
            rank_trace_path("/tmp/rank{rank}.jsonl", 3), "/tmp/rank3.jsonl"
        )

    def test_runtime_identity_rejects_a_changed_native_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "module.so"
            binary.write_bytes(b"native")
            for name in NATIVE_MODULES:
                (root / f"{name}.json").write_text(
                    json.dumps(
                        {
                            "library": str(binary),
                            "binary_id": hashlib.sha256(b"native").hexdigest(),
                            "source_fingerprint": "source",
                            "source_id": name,
                        }
                    )
                )
            with (
                patch(
                    "oh_my_vllm.models.deepseek_v41.trace.cache_directory",
                    return_value=root,
                ),
                patch(
                    "oh_my_vllm.models.deepseek_v41.trace.source_fingerprint",
                    return_value="source",
                ),
                patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(uuid="GPU-selected"),
                ),
                patch(
                    "oh_my_vllm.models.deepseek_v41.trace.source_identity",
                    return_value="worktree-content",
                ),
                patch.dict("os.environ", {"OH_MY_VLLM_SOURCE_COMMIT": "commit"}),
            ):
                for gpu_uuid in ("selected", "GPU-selected"):
                    with (
                        self.subTest(gpu_uuid=gpu_uuid),
                        patch(
                            "torch.cuda.get_device_properties",
                            return_value=SimpleNamespace(uuid=gpu_uuid),
                        ),
                    ):
                        identity = runtime_identity(2)
                        self.assertEqual(identity["gpu_uuid"], "GPU-selected")
                self.assertEqual(identity["rank"], 2)
                self.assertEqual(identity["source_commit"], "commit")
                self.assertEqual(identity["source_digest"], "worktree-content")
                self.assertEqual(set(identity["native_modules"]), set(NATIVE_MODULES))
                binary.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "native identity differs"):
                    runtime_identity(2)

    def test_routes_exclude_rejected_rows_and_deduplicate_destinations(self):
        ids = torch.tensor(
            [
                [0, 1, 96, 97, 192, 288],
                [378, 379, 380, 381, 382, 383],
                [0, 1, 2, 3, 4, 5],
            ],
            dtype=torch.int32,
        )
        mask = torch.tensor([True, False, True])
        for strategy, touched, dispatch in (
            ("tp4", 10, [[0, 0, 0, 0]]),
            ("tep4", 6, [[0, 0, 0, 0]]),
            ("dep4", 6, [[2, 1, 1, 1]]),
        ):
            with self.subTest(strategy=strategy):
                trace = RoutingTrace(2, 3, rank=0, strategy=strategy, device="cpu")
                trace.record(0, ids[:2])
                trace.record(0, ids[2:], first=2)
                trace.record(
                    1,
                    torch.tensor(
                        [[96, 97, 98, 99, 100, 101]], dtype=torch.int32
                    ).expand_as(ids),
                )
                counters = trace.counters(mask, expert_union=lambda _presence: None)
                self.assertEqual(
                    counters["expert_assignments"], [[8, 2, 1, 1], [0, 12, 0, 0]]
                )
                self.assertEqual(
                    counters["touched_experts"], [touched, 6 * int(strategy == "tp4")]
                )
                self.assertEqual(
                    counters["dispatch_rows"],
                    dispatch
                    + ([[0, 2, 0, 0]] if strategy == "dep4" else [[0, 0, 0, 0]]),
                )

    def test_dep4_empty_rank_counts_the_received_expert_union(self):
        trace = RoutingTrace(2, 1, rank=2, strategy="dep4", device="cpu")

        def receive(presence):
            self.assertEqual(presence.dtype, torch.uint8)
            self.assertEqual(presence.shape, (2, 384))
            self.assertEqual(int(presence.sum()), 0)
            presence[0, 192:194] = 1
            presence[1, 200] = 1

        union = Mock(side_effect=receive)
        result = trace.counters(torch.zeros(1, dtype=torch.bool), expert_union=union)
        union.assert_called_once()
        self.assertEqual(result["touched_experts"], [2, 1])
        self.assertEqual(result["expert_assignments"], [[0, 0, 0, 0]] * 2)
        self.assertEqual(result["dispatch_rows"], [[0, 0, 0, 0]] * 2)

    def test_executor_call_id_is_bound_before_later_calls_and_empty_records_are_kept(
        self,
    ):
        executor = SimpleNamespace(call=9)
        worker = SimpleNamespace(
            rank=2,
            trace=SimpleNamespace(record=Mock()),
            target_call_id_provider=lambda: executor.call,
        )
        call = target_call_id(worker, 1)
        executor.call = 10
        record_accounting_call(
            worker,
            call=call,
            phase="decode",
            request_rows=[],
            counters={"touched_experts": [0]},
        )
        self.assertEqual(worker.trace.record.call_args.kwargs["call"], 9)
        self.assertEqual(worker.trace.record.call_args.kwargs["request_rows"], [])
        with self.assertRaises(ValueError):
            target_call_id(SimpleNamespace(target_call_id_provider=lambda: -1), 1)

    def test_accounting_call_flushes_the_file_under_a_trace_proxy(self):
        from oh_my_vllm.worker.trace import Trace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.jsonl"
            with patch.dict("os.environ", {"OH_MY_VLLM_EXECUTION_TRACE": str(path)}):
                writer = Trace({}, {}, execution={})
            try:
                worker = SimpleNamespace(
                    rank=0, trace=SimpleNamespace(output=writer, record=writer.record)
                )
                record_accounting_call(
                    worker,
                    call=1,
                    phase="decode",
                    request_rows=[],
                    counters={"touched_experts": [0]},
                )
                event = json.loads(path.read_text().splitlines()[-1])
                self.assertEqual(event["operation"], "accounting_call")
                self.assertEqual(event["call"], 1)
            finally:
                writer.close()

    def test_prefill_calls_record_each_chunk_and_decode_credits_only_accepted_rows(
        self,
    ):
        worker = SimpleNamespace(
            rank=0,
            strategy="tp4",
            device="cpu",
            routing_trace_enabled=True,
            model=SimpleNamespace(layers=[None]),
            trace=SimpleNamespace(record=Mock()),
        )
        collector = TargetTrace.for_requests(worker, [(1, 3), (2, 3)])
        for start, length, logits in ((0, 2, False), (2, 1, True)):
            routes = collector.routes(length)
            routes.record(0, torch.arange(6, dtype=torch.int32).expand(2 * length, -1))
            collector.append(routes, start, length, logits)
            self.assertEqual(collector.records, [])
        records = [call.kwargs for call in worker.trace.record.call_args_list]
        self.assertEqual([record["call"] for record in records], [1, 2])
        self.assertEqual(
            [record["request_rows"][0]["logit_rows"] for record in records], [0, 1]
        )
        self.assertEqual(
            [record["expert_assignments"] for record in records],
            [[[24, 0, 0, 0]], [[12, 0, 0, 0]]],
        )
        collector = TargetTrace.for_requests(worker, [(1, 3)])
        routes = collector.routes(6)
        routes.record(0, torch.arange(6, dtype=torch.int32).expand(6, -1))
        collector.append(routes, 3, 6, True, verification=True)
        self.assertEqual(worker.trace.record.call_count, 2)
        collector.finish(accepted=2, proposed=5)
        record = worker.trace.record.call_args.kwargs
        self.assertEqual(record["call"], 3)
        self.assertEqual(record["expert_assignments"], [[18, 0, 0, 0]])
        self.assertEqual(
            record["request_rows"],
            [dict(id=1, start=3, count=3, accepted=2, proposed=5, logit_rows=3)],
        )
        self.assertIsNone(TargetTrace.for_requests(SimpleNamespace(), []))
        worker.strategy = "dep4"
        self.assertIsNone(TargetTrace.for_requests(worker, [(1, 3)]))

    def test_verification_is_one_call_with_a_small_prefill_limit(self):
        worker = _worker()
        worker.rank, worker.strategy, worker.device = 0, "tp4", torch.device("cpu")
        worker.prefill_rows = 2
        worker.routing_trace_enabled = True
        worker.trace = SimpleNamespace(record=Mock(), output=None)

        def forward(inputs, *, routing_trace, **_options):
            routing_trace.record(
                0, torch.arange(6, dtype=torch.int32).expand(inputs.numel(), -1)
            )
            return None, torch.zeros(1, inputs.size(1), 2), None

        worker.model = Mock(side_effect=forward)
        worker.model.layers = [None]
        trace = TargetTrace.for_requests(worker, [(1, 3)])
        worker._target([1, 2, 3, 4, 5, 6], 3, 0, full_logits=True, accounting=trace)
        trace.finish(accepted=0, proposed=5)
        self.assertEqual(worker.model.call_count, 1)
        self.assertEqual(worker.trace.record.call_count, 1)
        record = worker.trace.record.call_args.kwargs
        self.assertEqual(
            record["request_rows"],
            [dict(id=1, start=3, count=1, accepted=0, proposed=5, logit_rows=1)],
        )
        self.assertEqual(record["expert_assignments"], [[6, 0, 0, 0]])

    def test_history_replay_has_no_duplicate_credit_or_missing_logits(self):
        worker = SimpleNamespace(
            rank=0,
            strategy="tp4",
            device="cpu",
            routing_trace_enabled=True,
            model=SimpleNamespace(layers=[None]),
            trace=SimpleNamespace(record=Mock()),
        )

        def target(start, length, computes_logits):
            collector = TargetTrace.for_requests(worker, [(1, 3)])
            routes = collector.routes(length)
            routes.record(0, torch.arange(6, dtype=torch.int32).expand(length, -1))
            collector.append(routes, start, length, computes_logits)
            collector.finish()
            return worker.trace.record.call_args.kwargs

        target(0, 3, True)
        target(3, 1, True)
        target(4, 1, True)
        replay = target(0, 6, True)
        self.assertEqual(replay["phase"], "decode")
        self.assertEqual(
            replay["request_rows"],
            [dict(id=1, start=5, count=1, accepted=0, proposed=0, logit_rows=1)],
        )
        self.assertEqual(replay["expert_assignments"], [[6, 0, 0, 0]])
        replay = target(3, 2, False)
        self.assertEqual(replay["request_rows"], [])
        self.assertEqual(replay["expert_assignments"], [[0, 0, 0, 0]])
        replay = target(0, 6, True)
        self.assertEqual(replay["request_rows"], [])
        self.assertEqual(replay["expert_assignments"], [[0, 0, 0, 0]])

    def test_default_aligned_decode_keeps_separate_request_targets(self):
        worker = _chain_worker(prompts=((10, 11, 12), (20, 21, 22)), limit=2)
        worker.rank, worker.strategy = 0, "tp4"
        worker.routing_trace_enabled = True
        worker.execute(
            _wire_message(
                worker,
                dict(
                    rpc_id=1,
                    scheduled=[
                        _chain_row(worker, request.history, key=key)
                        for key, request in worker.requests.items()
                    ],
                    finished_request_ids=[],
                    preempted_request_ids=[],
                ),
            )
        )
        worker.trace.record.reset_mock()
        rows = [
            _chain_row(worker, [request.history[-1]], key=key)
            for key, request in worker.requests.items()
        ]
        with (
            patch.object(
                worker,
                "_execute_prefill_group",
                side_effect=AssertionError("decode cannot use prefill arithmetic"),
            ),
            patch.object(worker, "_execute_row", wraps=worker._execute_row) as execute,
        ):
            output = worker.execute(
                _wire_message(
                    worker,
                    dict(
                        rpc_id=2,
                        scheduled=rows,
                        finished_request_ids=[],
                        preempted_request_ids=[],
                    ),
                )
            )
        self.assertEqual([row["request_id"] for row in output["outputs"]], [1, 2])
        self.assertEqual(
            [
                ({key: call.args[0][key] for key in row},)
                for call, row in zip(execute.call_args_list, rows, strict=True)
            ],
            [(row,) for row in rows],
        )
        records = [
            call.kwargs
            for call in worker.trace.record.call_args_list
            if call.args == ("accounting_call",)
        ]
        self.assertEqual([record["phase"] for record in records], ["decode", "decode"])
        self.assertEqual(
            [record["expert_assignments"] for record in records], [[[6, 0, 0, 0]]] * 2
        )
        self.assertEqual(
            [record["request_rows"] for record in records],
            [
                [dict(id=key, start=3, count=1, accepted=0, proposed=0, logit_rows=1)]
                for key in (1, 2)
            ],
        )

    def test_explicit_dispatch_hook_receives_metadata_before_default_dep4_exchange(
        self,
    ):
        from oh_my_vllm.models.deepseek_v41.model import MoE

        moe = MoE.__new__(MoE)
        torch.nn.Module.__init__(moe)
        moe.dim = 4
        moe.layer_id = 0
        moe.data_parallel = True
        routing = (torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int32))
        moe.gate = Mock(return_value=routing)
        moe._routing_trace = SimpleNamespace(record=Mock())
        moe.dep4_exchange = Mock(
            side_effect=AssertionError("default exchange must not run")
        )
        rows = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        plan, valid = object(), torch.tensor([True, False])
        hook = Mock(return_value=rows.flatten(0, 1))
        value = moe(
            rows, logical_rows=1, plan=plan, valid_rows=valid, dispatch_combine=hook
        )
        torch.testing.assert_close(value, rows, rtol=0, atol=0)
        self.assertEqual(hook.call_count, 1)
        self.assertIs(hook.call_args.kwargs["layer"], moe)
        self.assertIs(hook.call_args.kwargs["routing"], routing)
        self.assertIs(hook.call_args.kwargs["valid_rows"], valid)
        self.assertIs(hook.call_args.kwargs["plan"], plan)
        moe.dep4_exchange.assert_not_called()
        moe._routing_trace.record.assert_called_once_with(0, routing[1])
        moe._routing_trace.record.reset_mock()
        moe.dep4_exchange = Mock(return_value=torch.zeros(2, 4))
        moe.shared_experts = Mock(return_value=torch.zeros(2, 4))
        moe(rows, logical_rows=1)
        moe._routing_trace.record.assert_called_once_with(0, routing[1])
        moe.dep4_exchange.assert_called_once()
