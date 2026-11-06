import errno
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from oh_my_vllm.models.qwen38.trace_identity import loaded_identity
from oh_my_vllm.worker.trace import Trace

from .benchmark_case import archive_case
from .benchmark_trace import join_events, prompt_tokens, wire_calls
from .qwen_fp8_evidence import Blocked, digest
from .test_benchmark_trace import fixture


class TraceObservationTests(unittest.TestCase):
    def test_missing_or_wrong_prompt_rows_are_blocked_at_the_input_boundary(self):
        for rows in (None, [None], [{}]):
            event = dict(operation="worker_scheduled", scheduled=rows)
            with self.subTest(rows=rows), self.assertRaises(Blocked):
                prompt_tokens([event], 7)

    def test_prepared_observation_uses_real_parameters_without_advancing_rng(self):
        import io

        from oh_my_vllm.models.qwen38.request import Request
        from oh_my_vllm.models.qwen38.trace_identity import record_prepared

        request = Request.__new__(Request)
        request.tokenizer = SimpleNamespace(generation=dict(temperature=0.2, top_k=20))
        request.options = dict(sampling=dict(temperature=0.95, repetition_guard=False))
        request.seed, request.minimum, request.limit = 7, 64, 64
        request.random_offset, request.proposal_offset, request.accept_offset = (
            17,
            19,
            23,
        )
        trace = Trace.__new__(Trace)
        trace.output = io.StringIO()
        record_prepared(trace, 1, dict(rpc_id=19, request_id=7), request, [10, 11])
        observed = json.loads(trace.output.getvalue())
        self.assertEqual(observed["sampling_f32_bits"]["temperature"], "3f733333")
        self.assertEqual(observed["sampling_f32_bits"]["top_k"], "41a00000")
        self.assertEqual(observed["seed"], 7)
        self.assertFalse(observed["repetition_guard"])
        self.assertEqual(observed["min_tokens"], 64)
        self.assertEqual(observed["prompt_token_sha256"], digest([10, 11]))
        self.assertEqual(
            (request.random_offset, request.proposal_offset, request.accept_offset),
            (17, 19, 23),
        )

    def test_prepare_requires_matching_rust_rpc_and_prompt(self):
        for mutation in ("missing", "rpc", "prompt", "duplicate"):
            worker, rust, manifest, artifacts, _ = fixture()
            if mutation == "missing":
                worker = [event for event in worker if event["operation"] != "prepared"]
            elif mutation == "rpc":
                rust[1]["rpc_id"] += 1
            elif mutation == "prompt":
                rust[1]["prompt_token_sha256"] = "f" * 64
            else:
                rust.insert(2, rust[1].copy())
            with self.subTest(mutation=mutation), self.assertRaises(Blocked):
                join_events(worker, rust, manifest, artifacts)

    def test_optional_write_and_flush_failures_preserve_the_result(self):
        for stage in ("write", "flush"):
            writer = SimpleNamespace(write=Mock(), flush=Mock())
            getattr(writer, stage).side_effect = OSError(errno.ENOSPC, "CPU fixture")
            trace = Trace.__new__(Trace)
            trace.output = writer
            expected = dict(type="execute_result", outputs=[])
            trace.record("worker_result", **expected)
            self.assertIsNone(trace.output)
            self.assertIn("CPU fixture", trace.failure)
            trace.record("worker_result", **expected)
            self.assertEqual(getattr(writer, stage).call_count, 1)
            self.assertEqual(expected, dict(type="execute_result", outputs=[]))

    def test_optional_open_and_close_failures_do_not_escape(self):
        with (
            patch.dict(
                "os.environ", OH_MY_VLLM_EXECUTION_TRACE="/external/trace.jsonl"
            ),
            patch.object(
                Path, "open", side_effect=OSError(errno.ENOSPC, "CPU fixture")
            ),
        ):
            trace = Trace({}, {})
        self.assertIsNone(trace.output)
        self.assertIn("CPU fixture", trace.failure)
        trace.output = SimpleNamespace(
            close=Mock(side_effect=OSError(errno.EIO, "close"))
        )
        trace.close()
        self.assertIsNone(trace.output)

    def test_missing_forward_dispatch_and_orphan_dispatch_are_blocked(self):
        for orphan in (False, True):
            worker, rust, manifest, artifacts, _ = fixture()
            if orphan:
                worker.append(dict(operation="dispatch", worker_epoch=999, step_id=999))
            else:
                worker = [event for event in worker if event["operation"] != "dispatch"]
            with self.subTest(orphan=orphan), self.assertRaises(Blocked):
                join_events(worker, rust, manifest, artifacts)

    def test_zero_row_head_only_and_export_only_do_not_need_dispatch(self):
        for action, kind in (("head_only", "target"), ("export_only", "none")):
            row = dict(
                request_id=7,
                lease_id=2,
                action=action,
                sampling=dict(kind=kind),
                forward_start=2,
                forward_end=2,
                endpoint=2,
                token_ids=[],
            )
            events = [
                dict(
                    operation="worker_scheduled",
                    worker_epoch=1,
                    step_id=2,
                    rpc_id=2,
                    scheduled=[row],
                ),
                dict(
                    operation="worker_result",
                    worker_epoch=1,
                    step_id=2,
                    rpc_id=2,
                    outputs=[],
                ),
            ]
            scheduled, _, dispatches = wire_calls(events)
            self.assertEqual(scheduled[(1, 2)]["scheduled"], [row])
            self.assertEqual(dispatches, {})
            events.append(dict(operation="dispatch", worker_epoch=1, step_id=2))
            with self.assertRaisesRegex(Blocked, "zero-row"):
                wire_calls(events)

    def test_missing_or_corrupt_optional_run_manifest_returns_identity_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for data in (None, "{incomplete", "null"):
                if data is not None:
                    path.write_text(data)
                with patch.dict("os.environ", OH_MY_VLLM_RUN_MANIFEST=str(path)):
                    observed = loaded_identity({}, {})
                self.assertIn("identity_error", observed)
                self.assertNotIn("artifact_id", observed)

    def test_null_worker_record_preserves_http_response_and_has_no_score(self):
        _, _, manifest, artifacts, responses = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = SimpleNamespace(
                mode="ordinary",
                trace=root / "worker.jsonl",
                rust_trace=root / "rust.jsonl",
                arguments=SimpleNamespace(
                    artifact_id=digest(artifacts),
                    run_identity=artifacts,
                    reference_version=artifacts["accounting"]["version"],
                ),
            )
            service.trace.write_text("null\n")
            service.rust_trace.write_text(
                json.dumps(dict(operation="complete", requests=[dict(id=7)])) + "\n"
            )
            result = archive_case(
                service,
                None,
                1,
                2,
                2,
                [manifest["requests"][0]["body"]],
                responses,
                {},
                100.0,
                102.0,
                b"",
                b"",
            )
        self.assertEqual(result["responses"], responses)
        self.assertEqual(result["lower_bound"]["status"], "blocked")
        self.assertIsNone(result["efficiency"])
        self.assertIsNone(result["lower_bound"]["seconds"])
