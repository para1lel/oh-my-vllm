import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.operators.runtime import native
from oh_my_vllm.worker.device_identity import device_identity

from .benchmark_case import archive_case
from .benchmark_trace import join_events, wire_digest
from .qwen_fp8_evidence import Blocked, digest
from .qwen_fp8_roofline import lower_bound
from .test_qwen_fp8_formal import archived_fixture


def dispatch_fixture(rows, at):
    counts = [len(row["token_ids"]) for row in rows]
    return dict(
        operation="dispatch",
        worker_epoch=at["worker_epoch"],
        step_id=at["step_id"],
        rows=sum(counts),
        max_queries=max(counts),
        verify=False,
        gdn_ba_precision="fp32",
        requests=[
            dict(request_id=row["request_id"], lease_id=row["lease_id"]) for row in rows
        ],
    )


def prepared_fixture():
    import io

    from oh_my_vllm.models.qwen38.request import Request
    from oh_my_vllm.models.qwen38.trace_identity import record_prepared
    from oh_my_vllm.worker.trace import Trace

    request = Request.__new__(Request)
    request.tokenizer = SimpleNamespace(generation={})
    request.options = dict(sampling=dict(seed=0, repetition_guard=True))
    request.seed, request.minimum, request.limit = 0, 2, 2
    trace = Trace.__new__(Trace)
    trace.output = io.StringIO()
    record_prepared(trace, 1, dict(rpc_id=19, request_id=7), request, [10, 11])
    worker = json.loads(trace.output.getvalue())
    rust = {
        name: worker[name]
        for name in (
            "worker_epoch",
            "rpc_id",
            "request_id",
            "prompt_token_sha256",
            "seed",
            "repetition_guard",
            "min_tokens",
            "max_tokens",
        )
    }
    rust.update(operation="prepare_validated", normalized_sampling_f32_bits={})
    return worker, rust


def fixture():
    events, manifest, artifacts, responses = archived_fixture()
    artifacts["rust_sources_sha256"] = "b" * 64
    manifest["artifact_id"] = digest(artifacts)
    loaded = copy.deepcopy(events[0])
    loaded.update(
        artifact_id=digest(artifacts),
        run_manifest=artifacts,
        cache_layout=loaded["cache_memory"],
    )
    worker = [loaded]
    rust = [
        dict(
            operation="rust_identity",
            worker_epoch=1,
            binary=artifacts["binary"],
            artifact_id=digest(artifacts),
            rust_sources_sha256=artifacts["rust_sources_sha256"],
            cache_layout=loaded["cache_layout"],
        )
    ]
    worker_prepared, rust_prepared = prepared_fixture()
    worker.insert(0, worker_prepared)
    rust.append(rust_prepared)
    for event in events[1:]:
        if event["operation"] != "accounting_call":
            rust.append(event)
            continue
        at = dict(
            worker_epoch=event["worker_epoch"],
            step_id=event["step_id"],
            rpc_id=event["step_id"],
        )
        worker.extend(
            [
                dict(operation="worker_scheduled", **at, scheduled=event["scheduled"]),
                dict(operation="worker_result", **at, outputs=event["outputs"]),
                dispatch_fixture(event["scheduled"], at),
            ]
        )
        proof = {key: at[key] for key in ("worker_epoch", "step_id")}
        proof.update(
            scheduled_sha256=wire_digest(event["scheduled"]),
            outputs_sha256=wire_digest(event["outputs"]),
        )
        keys = [
            dict(
                worker_epoch=at["worker_epoch"],
                step_id=at["step_id"],
                request_id=row["request_id"],
                lease_id=row["lease_id"],
            )
            for row in event["scheduled"]
        ]
        rust.append(dict(operation="validated", executions=keys, wire=proof))
    worker = (
        [event for event in worker if event["operation"] == "identity"]
        + [
            event
            for event in worker
            if event["operation"] not in ("identity", "prepared")
        ]
        + [worker_prepared]
    )
    rust = [rust[0]] + [
        event for event in rust[1:] if event["operation"] != "prepare_validated"
    ]
    rust.insert(1, rust_prepared)
    return worker, rust, manifest, artifacts, responses


class BenchmarkTraceTests(unittest.TestCase):
    def test_joins_execution_keys_instead_of_worker_log_order(self):
        worker, rust, manifest, artifacts, _ = fixture()
        worker[1:] = reversed(worker[1:])
        joined = join_events(worker, rust, manifest, artifacts)
        self.assertEqual(joined[1]["operation"], "prepared")
        self.assertEqual(joined[1]["request_id"], 7)
        self.assertEqual(joined[1]["sampling_f32_bits"]["temperature"], "3f800000")
        self.assertEqual(
            [row["step_id"] for row in joined if row["operation"] == "accounting_call"],
            [1, 2, 3],
        )
        self.assertEqual(joined[-1]["requests"][0]["accepted_end"], 4)

    def test_worker_self_report_does_not_establish_validation(self):
        worker, rust, manifest, artifacts, _ = fixture()
        worker.append(dict(operation="accounting_call", validated=True))
        rust = [row for row in rust if row["operation"] != "validated"]
        with self.assertRaisesRegex(Blocked, "missing Rust validation"):
            join_events(worker, rust, manifest, artifacts)

    def test_missing_or_duplicate_execution_is_blocked(self):
        for mutation in ("missing", "duplicate", "lease", "epoch"):
            worker, rust, manifest, artifacts, _ = fixture()
            if mutation == "missing":
                del worker[2]
            elif mutation == "duplicate":
                rust.insert(3, copy.deepcopy(rust[2]))
            else:
                field = "lease_id" if mutation == "lease" else "worker_epoch"
                rust[2]["executions"][0][field] += 1
            with self.subTest(mutation=mutation), self.assertRaises(Blocked):
                join_events(worker, rust, manifest, artifacts)

    def test_same_key_cannot_replace_validated_wire_contents(self):
        for target, field in ((1, "scheduled"), (2, "outputs")):
            worker, rust, manifest, artifacts, _ = fixture()
            worker[target][field][0]["token_ids"][0] += 1
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(Blocked, "validated bytes"),
            ):
                join_events(worker, rust, manifest, artifacts)

    def test_real_endpoint_and_dispatch_facts_are_preserved(self):
        worker, rust, manifest, artifacts, _ = fixture()
        row = worker[1]["scheduled"][0]
        row["sampling"] = dict(kind="none")
        row["model_state"]["endpoint_source"] = dict(kind="private", slot=8)
        row["model_state"]["private_checkpoints"].append(dict(slot=8, version=2))
        rust[2]["wire"]["scheduled_sha256"] = wire_digest(worker[1]["scheduled"])
        dispatch = dict(
            operation="dispatch",
            worker_epoch=1,
            step_id=1,
            rows=2,
            max_queries=2,
            verify=False,
            gdn_ba_precision="fp32",
        )
        worker = [
            event
            for event in worker
            if not (event["operation"] == "dispatch" and event["step_id"] == 1)
        ]
        worker.append(dispatch)
        joined = join_events(worker, rust, manifest, artifacts)
        call = joined[2]
        self.assertEqual(call["scheduled"][0], row)
        self.assertEqual(call["dispatches"], [dispatch])

    def test_loaded_binary_layout_and_run_manifest_must_match(self):
        for field in ("binary", "artifact_id", "rust_sources_sha256", "cache_layout"):
            worker, rust, manifest, artifacts, _ = fixture()
            rust[0][field] = "changed"
            with self.subTest(field=field), self.assertRaises(Blocked):
                join_events(worker, rust, manifest, artifacts)

    def test_missing_complete_has_no_score(self):
        worker, rust, manifest, artifacts, responses = fixture()
        rust = [row for row in rust if row["operation"] != "complete"]
        joined = join_events(worker, rust, manifest, artifacts)
        value = lower_bound(joined, manifest, artifacts, responses)
        self.assertEqual(value["status"], "blocked")
        self.assertIsNone(value["efficiency"])
        self.assertIsNone(value["seconds"])

    def test_cpu_case_archives_raw_evidence_and_keeps_http_denominator(self):
        worker, rust, manifest, artifacts, responses = fixture()
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

            def encoded(rows):
                return "".join(json.dumps(row) + "\n" for row in rows).encode()

            service.trace.write_bytes(encoded(worker))
            service.rust_trace.write_bytes(encoded(rust))
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
                encoded(worker[:1]),
                encoded(rust[:1]),
            )
        self.assertEqual(result["seconds"], 2.0)
        self.assertEqual(
            result["manifest"]["requests"][0]["prompt_token_ids"], [10, 11]
        )
        self.assertEqual(
            result["manifest"]["requests"][0]["body"], manifest["requests"][0]["body"]
        )
        self.assertEqual(result["lower_bound"]["status"], "blocked")
        self.assertIsNone(result["efficiency"])

    def test_native_loaded_identity_is_a_snapshot_of_a_successful_load(self):
        original = copy.deepcopy(native._loaded_identities)
        self.addCleanup(setattr, native, "_loaded_identities", original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "probe.so"
            library.write_bytes(b"CPU loader fixture")
            metadata = dict(
                source_fingerprint="loaded source",
                functions=["probe"],
                ffi_version="fixture",
                source_id="loaded source ID",
                library=str(library),
                binary_id=native.file_sha256(library),
            )
            manifest = root / "trace_probe.json"
            manifest.write_text(json.dumps(metadata))
            module = object()
            with (
                patch.object(native, "cache_directory", return_value=root),
                patch.object(
                    native, "source_fingerprint", return_value="loaded source"
                ),
                patch.object(
                    native, "source_identity", return_value="loaded source ID"
                ),
                patch.object(native, "require_current_environment"),
                patch.object(
                    native.importlib.metadata, "version", return_value="fixture"
                ),
                patch("tvm_ffi.load_module", return_value=module),
            ):
                actual, _, _ = native.load.__wrapped__("trace_probe", ("probe",))
            manifest.write_text('{"binary_id":"expected bytes"}')
            observed = native.loaded_identities()
            observed["trace_probe"]["binary_id"] = "expected bytes"
            self.assertIs(actual, module)
            self.assertEqual(native.loaded_identities()["trace_probe"], metadata)

    def test_device_metadata_must_come_from_the_cuda_uuid(self):
        with patch(
            "oh_my_vllm.worker.device_identity.cuda_device_identity",
            return_value=dict(
                uuid="GPU-actual",
                multiprocessor_count=148,
                l2_cache_bytes=126 * 1024**2,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "selected GPU"):
                device_identity("GPU-expected")
            with (
                patch(
                    "oh_my_vllm.worker.device_identity.subprocess.check_output",
                    return_value="GPU-other, B200, driver, memory, power",
                ),
                self.assertRaisesRegex(ValueError, "different UUID"),
            ):
                device_identity()

    def test_missing_or_malformed_worker_evidence_is_blocked(self):
        _, rust, manifest, artifacts, responses = fixture()
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
            service.rust_trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rust)
            )
            for raw in (None, b"{incomplete"):
                if raw is not None:
                    service.trace.write_bytes(raw)
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
                self.assertEqual(result["lower_bound"]["status"], "blocked")
                self.assertIsNone(result["efficiency"])
                self.assertEqual(result["responses"], responses)

    def test_unicode_wire_digest_is_stable(self):
        self.assertEqual(
            wire_digest({"b": "实际输出", "a": 7}),
            wire_digest({"a": 7, "b": "实际输出"}),
        )
