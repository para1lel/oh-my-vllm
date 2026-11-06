import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

from .deepseek_accounting import (
    BindingError,
    TraceError,
    digest,
)
from .deepseek_runtime_trace import assemble_trace, rank_paths, read_rank_events
from .test_deepseek_benchmark import runtime_fixture, write_runtime_events


class DeepSeekRuntimeTrace(unittest.TestCase):
    def test_manifest_uses_response_ids_and_http_usage(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=2, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        responses.reverse()
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(
            [row["id"] for row in trace["manifest"]["requests"]], [102, 101]
        )
        self.assertEqual(trace["binding"]["manifest_sha256"], digest(trace["manifest"]))
        for field, value in (("completion_tokens", 3), ("prompt_tokens", True)):
            broken = deepcopy(responses)
            broken[0]["usage"][field] = value
            with self.subTest(field=field), self.assertRaises(TraceError):
                assemble_trace(events, broken, harness=harness, case=case)
        for response_id in (None, "chatcmpl_test_0", "chatcmpl_test_102x"):
            broken = deepcopy(responses)
            broken[0]["id"] = response_id
            with self.subTest(response_id=response_id), self.assertRaises(TraceError):
                assemble_trace(events, broken, harness=harness, case=case)
        responses[0]["id"] = responses[1]["id"]
        with self.assertRaisesRegex(TraceError, "repeat"):
            assemble_trace(events, responses, harness=harness, case=case)

    def test_runtime_identities_reject_a_different_gpu_uuid_for_each_strategy(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                case = dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=1,
                    input_tokens=2,
                    output_tokens=2,
                )
                harness, events, responses = runtime_fixture(case)
                for rank in range(4):
                    uuid = f"GPU-{UUID(int=rank)}"
                    harness["topology"][rank]["uuid"] = uuid
                    events[rank][0]["execution"]["runtime"]["gpu_uuid"] = uuid
                events[0][0]["execution"]["runtime"]["gpu_uuid"] = f"GPU-{UUID(int=99)}"
                with self.assertRaisesRegex(
                    BindingError, "rank 0 runtime identity differs"
                ):
                    assemble_trace(events, responses, harness=harness, case=case)

    def test_runtime_identities_reject_changed_evidence_and_fallback(self):
        case = dict(
            mode="dspark5", strategy="tep4", batch=1, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        for path in (
            ("execution", "runtime", "source_commit"),
            ("execution", "runtime", "source_digest"),
            ("execution", "runtime", "native_modules", "projection", "binary"),
            ("execution", "runtime", "gpu_uuid"),
            ("checkpoint", "config"),
            ("checkpoint", "index"),
            ("tokenizer", "tokenizer.json"),
            ("cache_memory", "effective_strategy"),
        ):
            broken = deepcopy(events)
            parent = broken[2][0]
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = "changed"
            with self.subTest(path=path), self.assertRaises(BindingError):
                assemble_trace(broken, responses, harness=harness, case=case)
        broken = deepcopy(events)
        broken[1].append(deepcopy(broken[1][0]))
        with self.assertRaisesRegex(BindingError, "one initial identity"):
            assemble_trace(broken, responses, harness=harness, case=case)
        case["strategy"] = "dep4"
        for rows in events:
            rows[0]["cache_memory"]["strategy"] = "dep4"
        with self.assertRaisesRegex(BindingError, "fallback"):
            assemble_trace(events, responses, harness=harness, case=case)

    def test_mixed_target_call_records_an_error_and_rejects_the_complete_trace(self):
        import torch
        from oh_my_vllm.models.deepseek_v41.trace import TargetTrace

        worker = SimpleNamespace(
            rank=0,
            strategy="tp4",
            device="cpu",
            routing_trace_enabled=True,
            model=SimpleNamespace(layers=[None]),
            target_trace_frontiers={101: 3, 102: 3},
            trace=SimpleNamespace(record=Mock()),
        )
        collector = TargetTrace.for_requests(worker, [(101, 3), (102, 4)])
        routes = collector.routes(1)
        routes.record(0, torch.arange(6, dtype=torch.int32).expand(2, -1))
        collector.append(routes, 3, 1, True)
        worker.trace.record.assert_called_once()
        operation = worker.trace.record.call_args.args[0]
        error = worker.trace.record.call_args.kwargs
        self.assertEqual(operation, "accounting_error")
        self.assertEqual(error["call"], 1)
        self.assertIn("mixed", error["reason"])
        self.assertEqual(worker.target_trace_frontiers, {101: 3, 102: 3})
        for start, frontier, verification in ((0, 0, False), (2, 2, True)):
            with self.subTest(start=start, verification=verification):
                worker.trace.record.reset_mock()
                worker.target_trace_frontiers = {101: frontier}
                collector = TargetTrace.for_requests(worker, [(101, 3)])
                routes = collector.routes(6)
                routes.record(0, torch.arange(6, dtype=torch.int32).expand(6, -1))
                collector.append(routes, start, 6, True, verification=verification)
                collector.finish(accepted=2, proposed=5)
                worker.trace.record.assert_called_once()
                self.assertEqual(
                    worker.trace.record.call_args.args[0], "accounting_error"
                )
                self.assertIn(
                    "prompt boundary", worker.trace.record.call_args.kwargs["reason"]
                )
                self.assertEqual(worker.target_trace_frontiers, {101: frontier})

        case = dict(
            mode="ordinary", strategy="tp4", batch=2, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        events[0].append(dict(operation=operation, **error))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_runtime_events(path, events)
            loaded = read_rank_events(path)
            self.assertEqual(loaded[0][-1]["operation"], "accounting_error")
            with self.assertRaisesRegex(TraceError, "accounting_error.*call 1"):
                assemble_trace(loaded, responses, harness=harness, case=case)

    def test_call_selection_keeps_complete_calls_and_rejects_cross_case_work(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=1, input_tokens=2, output_tokens=2
        )
        harness, events, responses = runtime_fixture(case)
        for rows in events:
            other = deepcopy(rows[1])
            other["call"] = 90
            other["request_rows"][0]["id"] = 999
            rows.append(other)
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual({row["call"] for row in trace["calls"]}, {0, 1})
        for rank in (0, 3):
            broken = deepcopy(events)
            foreign = deepcopy(broken[rank][1]["request_rows"][0])
            foreign["id"] = 999
            broken[rank][1]["request_rows"].append(foreign)
            with self.subTest(rank=rank), self.assertRaisesRegex(TraceError, "mixes"):
                assemble_trace(broken, responses, harness=harness, case=case)
        for change in ("missing_rank", "duplicate", "missing_layer", "gap"):
            broken = deepcopy(events)
            if change == "missing_rank":
                broken[3].pop(1)
            elif change == "duplicate":
                broken[1].append(deepcopy(broken[1][1]))
            elif change == "missing_layer":
                broken[0][1]["expert_assignments"].pop()
            else:
                broken[0][2]["request_rows"][0]["start"] += 1
            with self.subTest(change=change), self.assertRaises(TraceError):
                assemble_trace(broken, responses, harness=harness, case=case)

    def test_dep4_keeps_empty_ranks_and_observed_owner(self):
        case = dict(
            mode="dspark5", strategy="dep4", batch=1, input_tokens=2, output_tokens=4
        )
        harness, events, responses = runtime_fixture(case)
        # Move the actual owner away from both ordinal and request-ID modulo four.
        for left, right in zip(events[0][1:], events[2][1:], strict=True):
            left["request_rows"], right["request_rows"] = (
                right["request_rows"],
                left["request_rows"],
            )
            left["expert_assignments"], right["expert_assignments"] = (
                right["expert_assignments"],
                left["expert_assignments"],
            )
            left["dispatch_rows"], right["dispatch_rows"] = (
                right["dispatch_rows"],
                left["dispatch_rows"],
            )
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(trace["manifest"]["requests"][0]["owner_rank"], 2)
        self.assertEqual(len(trace["calls"]), 8)
        self.assertEqual(trace["binding"]["accepted"], 2)
        self.assertTrue(
            all(not row["request_rows"] for row in trace["calls"] if row["rank"] != 2)
        )
        self.assertTrue(
            any(row["touched_experts"][0] for row in trace["calls"] if row["rank"] == 0)
        )
        broken = deepcopy(events)
        broken[1][2]["request_rows"] = broken[2][2]["request_rows"]
        broken[2][2]["request_rows"] = []
        with self.assertRaisesRegex(TraceError, "owner changed"):
            assemble_trace(broken, responses, harness=harness, case=case)
        broken = deepcopy(events)
        broken[0][1]["request_rows"] = deepcopy(broken[2][1]["request_rows"])
        with self.assertRaisesRegex(TraceError, "owner changed"):
            assemble_trace(broken, responses, harness=harness, case=case)

    def test_reader_rejects_missing_rank_and_unflushed_records(self):
        case = dict(
            mode="ordinary", strategy="tp4", batch=1, input_tokens=2, output_tokens=1
        )
        _, events, _ = runtime_fixture(case)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.jsonl"
            write_runtime_events(path, events)
            self.assertEqual(read_rank_events(path), events)
            rank_paths(path)[3].unlink()
            with self.assertRaisesRegex(TraceError, "rank 3"):
                read_rank_events(path)
            rank_paths(path)[3].write_text('{"operation":')
            with self.assertRaisesRegex(TraceError, "rank 3"):
                read_rank_events(path)
