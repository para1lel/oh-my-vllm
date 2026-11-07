import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from oh_my_vllm.models.qwen38.cache_budget import cache_layout
from oh_my_vllm.models.qwen38.tokenizer import Tokenization, tokenizer_identity
from tokenizers import Tokenizer, decoders, models

from . import qwen_fp8_acceptance as acceptance
from . import qwen_fp8_identity as binding
from .accounting import RATES
from .benchmark_trace import join_events, wire_digest
from .qwen_fp8_evidence import VERSION, digest
from .qwen_fp8_roofline import lower_bound
from .test_qwen_fp8_contract import (
    GENERATION,
    fixture_dispatch,
    fixture_prepared,
    wire_result,
    wire_row,
)
from .test_qwen_fp8_formal import archived_fixture, fixture_artifacts


def encoded(rows):
    return b"".join((json.dumps(row) + "\n").encode() for row in rows)


def save(path, value):
    path.write_text(json.dumps(value))


def assets(root):
    root.mkdir()
    vocab = {str(index): index for index in range(13)}
    for token, index in (("[UNK]", "0"), ("</think>", "1"), ("x", "12")):
        vocab[token] = vocab.pop(index)
    tokenizer = Tokenizer(models.BPE(vocab, [], unk_token="[UNK]"))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[UNK]", "</think>"])
    tokenizer.save(str(root / "tokenizer.json"))
    save(root / "generation_config.json", dict(GENERATION, eos_token_id=0))
    (root / "chat_template.jinja").write_text("{{ messages[0].content }}")
    return root


def run_identity(tokenizer_dir):
    result = fixture_artifacts()
    result.update(
        tokenizer=tokenizer_identity(tokenizer_dir),
        rust_sources_sha256="b" * 64,
        config=dict(model="recorded-checkpoint", mamba_blocks=1, blocks=None),
        accounting=dict(
            version=VERSION, rates=RATES, calculator=binding.calculator_identity()
        ),
    )
    return result


def loaded_identity(artifacts, mode):
    config = dict(
        model_path=artifacts["config"]["model"],
        mamba_blocks=1,
        num_gpu_blocks=64,
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_num_batched_tokens=32768,
        max_model_len=262144,
        block_size=128,
        worker_epoch=1 if mode == "ordinary" else 2,
        num_speculative_tokens=0 if mode == "ordinary" else 4,
    )
    return dict(
        artifacts,
        operation="identity",
        artifact_id=digest(artifacts),
        run_manifest=artifacts,
        config=config,
        cache_layout=cache_layout(config).wire(),
    )


def result_record(key, manifest, responses, reference, ranges):
    return dict(
        **asdict(key),
        manifest=manifest,
        responses=responses,
        lower_bound=reference,
        manifest_id=reference["manifest_id"],
        trace_id=reference["trace_id"],
        seconds=reference["http_seconds"],
        efficiency=reference["efficiency"],
        trace_ranges=ranges,
    )


def matrix_archive(root, tokenizer_dir, change_loaded=None):
    artifacts = run_identity(tokenizer_dir)
    save(root / "identity.json", artifacts)
    for index, shape in enumerate(acceptance.CONFIGURATIONS):
        key = acceptance.CaseKey(*shape)
        loaded = loaded_identity(artifacts, key.mode)
        if change_loaded:
            change_loaded(key.mode, loaded)
        loaded["cache_memory"] = loaded["cache_layout"]
        value = dict(
            **asdict(key),
            fixture_loaded=loaded,
            manifest=dict(
                mode=key.mode, requests=[dict(asdict(key), body={})] * key.batch
            ),
            responses=[{}] * key.batch,
            lower_bound=dict(status="calculated", seconds=4.0),
            seconds=10.0,
            efficiency=0.40,
            trace_ranges={
                name: dict(start=index, end=index + 1) for name in acceptance.STREAMS
            },
        )
        save(root / key.filename, value)
    save(root / "summary.json", dict(configurations=24))


def stub_case(path, key, value, artifacts, tokenizer, streams):
    manifest, responses = acceptance._shape(key, value)
    loaded = value["fixture_loaded"]
    acceptance._loaded_configuration(loaded, artifacts, key.mode)
    return acceptance.LoadedCase(
        key,
        path,
        digest(artifacts),
        manifest,
        responses,
        [loaded],
        value,
        value["lower_bound"],
        value["seconds"],
        value["efficiency"],
    )


@contextmanager
def aggregate_replay():
    with (
        patch.object(acceptance, "_raw_streams", return_value={}),
        patch.object(acceptance, "_load_case", side_effect=stub_case),
    ):
        yield


def raw_call(history, step):
    prompt, final = 32768, step == 65
    start, end = (0, prompt) if step == 1 else (prompt + step - 2, prompt + step - 1)
    row = wire_row(
        start,
        end,
        history[start:end],
        "finalize" if final else "advance",
        "none" if final else "target",
    )
    row["prefill_token_ids"] = history[:prompt] if step == 1 else None
    for page in row["model_state"]["page_table"]:
        page["valid_tokens"] = max(0, min(128, start - page["page_index"] * 128))
    result = wire_result(row, [] if final else [12])
    result["committed_page_valid_ends"] = [
        dict(
            page_index=p["page_index"],
            slot=p["slot"],
            version=p["version"],
            valid_end=min(end, (p["page_index"] + 1) * 128),
        )
        for p in row["model_state"]["page_table"]
    ]
    at = dict(worker_epoch=1, step_id=step)
    worker = [
        dict(operation="worker_scheduled", rpc_id=step, scheduled=[row], **at),
        dict(operation="worker_result", rpc_id=step, outputs=[result], **at),
        *fixture_dispatch(row, step),
    ]
    rust = dict(
        operation="validated",
        executions=[dict(request_id=7, lease_id=1, **at)],
        wire=dict(
            scheduled_sha256=wire_digest([row]),
            outputs_sha256=wire_digest([result]),
            **at,
        ),
    )
    return worker, rust


def real_archive(root, tokenizer_dir):
    artifacts = run_identity(tokenizer_dir)
    loaded = loaded_identity(artifacts, "ordinary")
    initial, manifest, _, responses = archived_fixture()
    key = acceptance.CaseKey("ordinary", 1, 32768, 64)
    history = [10] * key.input_tokens + [12] * key.output_tokens
    request = manifest["requests"][0]
    request.update(
        input_tokens=32768, output_tokens=64, prompt_token_ids=history[:32768]
    )
    request["body"].update(min_tokens=64, max_tokens=64)
    request["body_sha256"] = digest(request["body"])
    manifest["artifact_id"] = digest(artifacts)
    responses[0]["usage"].update(
        prompt_tokens=32768, completion_tokens=64, total_tokens=32832
    )
    responses[0]["choices"][0]["message"] = dict(role="assistant", content="x" * 64)
    prepared = fixture_prepared(request)
    intent = prepared.pop("rust_intent")
    worker = [loaded, prepared]
    observed = {
        name: loaded[name] for name in ("binary", "artifact_id", "cache_layout")
    }
    observed.update(
        operation="rust_identity", worker_epoch=1, rust_sources_sha256="b" * 64
    )
    rust = [
        observed,
        dict(
            operation="prepare_validated",
            worker_epoch=1,
            rpc_id=1,
            request_id=7,
            prompt_token_sha256=prepared["prompt_token_sha256"],
            **intent,
        ),
    ]
    for step in range(1, 66):
        call, proof = raw_call(history, step)
        worker.extend(call)
        if step == 65:
            rust.append(
                dict(operation="accepted", requests=[dict(id=7, accepted_end=32832)])
            )
        rust.append(proof)
    complete = initial[-1]
    complete["requests"][0].update(accepted_end=32832, token_sha256=digest(history))
    complete["requests"][0]["execution"]["step_id"] = 65
    complete["requests"][0]["response"] = responses[0]
    rust.append(complete)
    events = join_events(worker, rust, manifest, artifacts)
    reference = lower_bound(
        events, manifest, artifacts, responses, tokenizer=Tokenization(tokenizer_dir)
    )
    assert reference["status"] == "calculated", reference
    ranges = {
        name: dict(
            file=f"ordinary-{acceptance.STREAMS[name]}",
            start=len(encoded(rows[:1])),
            end=len(encoded(rows)),
        )
        for name, rows in (("worker", worker), ("rust", rust))
    }
    save(root / "identity.json", artifacts)
    save(
        root / key.filename, result_record(key, manifest, responses, reference, ranges)
    )
    for name, rows in (("worker", worker), ("rust", rust)):
        (root / f"ordinary-{acceptance.STREAMS[name]}").write_bytes(encoded(rows))
    return key


class QwenFP8AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-acceptance-cpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.tokenizer = assets(Path(cls.temporary.name) / "tokenizer")
        cls.real = Path(cls.temporary.name) / "real"
        cls.real.mkdir()
        cls.real_key = real_archive(cls.real, cls.tokenizer)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-acceptance-case-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def evaluate(self, root=None):
        return acceptance.evaluate_archive(
            root or self.root, tokenizer_dir=self.tokenizer
        )

    def test_all_24_keys_and_12_inclusive_gates_with_legal_mode_differences(self):
        matrix_archive(self.root, self.tokenizer)
        with aggregate_replay():
            result = self.evaluate()
        self.assertEqual(
            (result.status, result.observed_cases, result.observed_long_cases),
            ("passed", 24, 12),
        )
        ordinary, mtp = result.rows[0].events[0], result.rows[-1].events[0]
        self.assertNotEqual(
            ordinary["config"]["worker_epoch"], mtp["config"]["worker_epoch"]
        )
        self.assertNotEqual(
            ordinary["cache_memory"]["pool_bytes"], mtp["cache_memory"]["pool_bytes"]
        )
        self.assertEqual(ordinary["run_manifest"], mtp["run_manifest"])

    def test_matrix_inventory_and_cross_row_consistency(self):
        for mutation in (
            "short",
            "long",
            "duplicate",
            "extra",
            "overlap",
            "loaded",
            "stream",
        ):
            root = self.root / mutation
            root.mkdir()
            matrix_archive(root, self.tokenizer)
            path = root / "ordinary-1-32768-64.json"
            if mutation in ("short", "long"):
                (
                    path if mutation == "short" else root / "ordinary-1-32768-8192.json"
                ).unlink()
                save(root / "summary.json", dict(configurations=23))
            elif mutation in ("duplicate", "extra"):
                value = json.loads(path.read_bytes())
                value["input_tokens"] = 65536 if mutation == "extra" else 32768
                target = (
                    "ordinary-1-65536-64.json"
                    if mutation == "extra"
                    else "renamed.json"
                )
                save(root / target, value)
                save(root / "summary.json", dict(configurations=25))
            elif mutation in ("overlap", "loaded", "stream"):
                value = json.loads(path.read_bytes())
                if mutation == "overlap":
                    value["trace_ranges"]["worker"]["end"] = 100
                elif mutation == "stream":
                    value["manifest"]["requests"][0]["body"]["stream"] = True
                else:
                    value["fixture_loaded"]["config"]["worker_epoch"] = 9
                save(path, value)
            with self.subTest(mutation=mutation), aggregate_replay():
                result = self.evaluate(root)
                self.assertEqual(result.status, "invalid")
                if mutation == "duplicate":
                    self.assertEqual(len(result.duplicates), 1)
                if mutation == "extra":
                    self.assertEqual(len(result.extra), 1)

    def test_one_long_row_below_forty_percent_fails_without_averaging(self):
        matrix_archive(self.root, self.tokenizer)
        path = self.root / "ordinary-1-32768-8192.json"
        value = json.loads(path.read_bytes())
        value["lower_bound"]["seconds"] = 3.99999
        value["efficiency"] = 0.399999
        save(path, value)
        with aggregate_replay():
            result = self.evaluate()
        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.failed_threshold_cases,
            (acceptance.CaseKey("ordinary", 1, 32768, 8192),),
        )

    def test_calculator_binding_is_mandatory(self):
        for mutation in ("missing", "changed"):
            root = self.root / mutation
            root.mkdir()
            matrix_archive(root, self.tokenizer)
            path = root / "identity.json"
            identity = json.loads(path.read_bytes())
            if mutation == "missing":
                del identity["accounting"]["calculator"]
            else:
                identity["accounting"]["calculator"]["sha256"] = "0" * 64
            save(path, identity)
            with self.subTest(mutation=mutation), aggregate_replay():
                result = self.evaluate(root)
                self.assertEqual(result.status, "invalid")
                self.assertTrue(any("calculator" in error for error in result.errors))
        bad = assets(self.root / "broken-tokenizer")
        (bad / "tokenizer.json").write_text("{}")
        result = acceptance.evaluate_archive(self.real, tokenizer_dir=bad)
        self.assertEqual(result.status, "invalid")

    def test_requested_blocks_mode_and_pool_configuration_are_checked(self):
        for mutation in ("mamba", "blocks", "speculation", "pool"):
            root = self.root / mutation
            root.mkdir()

            def change(mode, loaded, mutation=mutation):
                if mode != "ordinary":
                    return
                if mutation == "pool":
                    loaded["cache_layout"]["pool_bytes"]["kv_codes"] += 1
                else:
                    field, value = {
                        "mamba": ("mamba_blocks", 2),
                        "blocks": ("num_gpu_blocks", 128),
                        "speculation": ("num_speculative_tokens", 4),
                    }[mutation]
                    loaded["config"][field] = value

            matrix_archive(root, self.tokenizer, change)
            with self.subTest(mutation=mutation), aggregate_replay():
                self.assertEqual(self.evaluate(root).status, "invalid")

    def test_real_raw_join_calculator_and_saved_json_round_trip(self):
        normalize = acceptance._canonical
        self.assertEqual(normalize({9: 64, 10: 64}), normalize({"9": 64, "10": 64}))
        self.assertNotEqual(normalize(True), normalize(1))
        shutil.copytree(self.real, self.root, dirs_exist_ok=True)
        result = self.evaluate()
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(
            result.errors, ("the 24-case HTTP matrix is missing, duplicated or extra",)
        )

    def test_raw_bound_usage_identity_and_blocked_mutations_are_invalid(self):
        mutations = (
            "wire",
            "complete",
            "bound",
            "trace_id",
            "usage",
            "identity",
            "blocked",
            "timer",
            "content",
            "message",
            "error",
        )
        for mutation in mutations:
            root = self.root / mutation
            shutil.copytree(self.real, root)
            path = root / self.real_key.filename
            value = json.loads(path.read_bytes())
            if mutation in ("wire", "complete", "identity"):
                name = "worker" if mutation == "wire" else "rust"
                stream = root / f"ordinary-{acceptance.STREAMS[name]}"
                events = [json.loads(line) for line in stream.read_bytes().splitlines()]
                if mutation == "wire":
                    events[2]["scheduled"][0]["token_ids"][0] = 11
                elif mutation == "complete":
                    events[-1]["requests"][0].pop("response")
                else:
                    events[0]["artifact_id"] = "c" * 64
                raw = encoded(events)
                stream.write_bytes(raw)
                value["trace_ranges"][name]["end"] = len(raw)
            elif mutation == "bound":
                value["lower_bound"]["acceptance"]["7"]["output"] = 65
            elif mutation == "trace_id":
                value[mutation] = "0" * 64
            elif mutation == "usage":
                value["responses"][0]["usage"]["completion_tokens"] = 63
            elif mutation == "content":
                value["responses"][0]["choices"][0]["message"]["content"] = "wrong text"
            elif mutation == "message":
                value["responses"][0]["choices"][0]["message"] = {}
            elif mutation == "error":
                value["responses"][0]["error"] = {"message": "failed"}
            elif mutation == "blocked":
                value["lower_bound"].update(
                    status="blocked", seconds=None, efficiency=None
                )
            else:
                value["manifest"]["timer"]["finished"] += 1
            save(path, value)
            with self.subTest(mutation=mutation):
                result = self.evaluate(root)
                self.assertEqual((result.status, len(result.rows)), ("invalid", 0))
                if mutation == "complete":
                    self.assertTrue(any("response" in error for error in result.errors))

    def test_raw_range_boundaries_are_strict(self):
        changes = dict(
            boolean=dict(start=True),
            overrun=dict(end=10**9),
            half_line=dict(start=1),
            mode=dict(file="mtp4-execution.jsonl"),
            truncated={},
        )
        for mutation, change in changes.items():
            root = self.root / mutation
            shutil.copytree(self.real, root)
            path = root / self.real_key.filename
            value = json.loads(path.read_bytes())
            interval = value["trace_ranges"]["worker"]
            interval.update(change)
            if mutation == "truncated":
                stream = root / interval["file"]
                stream.write_bytes(stream.read_bytes() + b"{")
            save(path, value)
            with self.subTest(mutation=mutation):
                result = self.evaluate(root)
                self.assertEqual((result.status, len(result.rows)), ("invalid", 0))
