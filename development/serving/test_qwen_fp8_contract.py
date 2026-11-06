import hashlib
import json
import struct
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .qwen_fp8_math import ba_precision
from .qwen_fp8_roofline import lower_bound
from .qwen_fp8_trace import LAYOUT, digest, validate, validate_execution

GENERATION = dict(
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    min_p=0,
    frequency_penalty=0,
    presence_penalty=0.5,
    repetition_penalty=1,
    repetition_guard=True,
)
GENERATION_BYTES = json.dumps(GENERATION).encode()
GENERATION_SHA256 = hashlib.sha256(GENERATION_BYTES).hexdigest()


@contextmanager
def fixture_tokenizer(artifacts):
    with tempfile.TemporaryDirectory(prefix="qwen-accounting-cpu-") as directory:
        path = Path(directory)
        (path / "generation_config.json").write_bytes(GENERATION_BYTES)
        yield SimpleNamespace(
            path=path,
            generation=dict(GENERATION),
            identity=deepcopy(artifacts["tokenizer"]),
        )


def validate_fixture(*args):
    with fixture_tokenizer(args[2]) as tokenizer:
        return validate_execution(*args, tokenizer=tokenizer)


def fixture_dispatch(row, step):
    if row["action"] != "forward":
        return []
    count = row["forward_end"] - row["forward_start"]
    verify = row["sampling"]["kind"] == "verify"
    return [
        dict(
            operation="dispatch",
            worker_epoch=1,
            step_id=step,
            execution="eager",
            rows=count,
            max_queries=count,
            verify=verify,
            gdn_ba_precision=ba_precision(count, count, verify),
            requests=[dict(request_id=row["request_id"], lease_id=row["lease_id"])],
        )
    ]


def wire_row(start, end, inputs, purpose="advance", kind="target", action="forward"):
    pages = [
        dict(
            page_index=index,
            slot=index + 1,
            version=1,
            valid_tokens=min(128, start),
            access="write",
        )
        for index in range((end + 127) // 128)
    ]
    state = dict(
        kind="qwen_radix_v1",
        layout_fingerprint=LAYOUT,
        checkpoint_hit_end=0,
        kv_contiguous_end=0,
        resume=dict(end=start, slot=0 if start == 0 else 2, version=1),
        page_table=pages,
        kv_spans=[dict(start=start, end=end, mode="write")] if end > start else [],
        endpoint_source=dict(kind="forward"),
        page_copies=[],
        private_checkpoints=[dict(slot=i, version=1) for i in range(2, 8)],
        exports=[dict(end=end, slot=2, version=1, role="private_endpoint")],
    )
    return dict(
        request_id=7,
        lease_id=1,
        forward_start=start,
        forward_end=end,
        prefill_token_ids=[10, 11] if start == 0 else None,
        endpoint=end,
        accepted_end=end,
        token_ids=inputs,
        purpose=purpose,
        action=action,
        sampling=dict(kind=kind),
        model_state=state,
    )


def wire_result(row, emitted):
    end = row["forward_end"]
    return dict(
        request_id=7,
        lease_id=1,
        purpose=row["purpose"],
        action=row["action"],
        error=None,
        token_ids=emitted,
        num_accepted_draft_tokens=0,
        new_draft_token_ids=[],
        computed_end=end,
        accepted_end=end + len(emitted),
        validated_exports=[dict(end=end, slot=2, version=1)],
        committed_page_valid_ends=[
            dict(page_index=0, slot=1, version=1, valid_end=end)
        ],
        mtp_accepted_endpoint=None,
    )


def fixture_prepared(request):
    return dict(
        operation="prepared",
        worker_epoch=1,
        rpc_id=1,
        request_id=request["id"],
        prompt_token_sha256=digest(request["prompt_token_ids"]),
        sampling_f32_bits={
            name: struct.pack(">f", value).hex()
            for name, value in GENERATION.items()
            if name != "repetition_guard"
        },
        seed=request["seed"],
        repetition_guard=True,
        min_tokens=request["output_tokens"],
        max_tokens=request["output_tokens"],
        rust_intent=dict(
            normalized_sampling_f32_bits={"presence_penalty": "3f000000"},
            seed=request["seed"],
            repetition_guard=True,
            min_tokens=request["output_tokens"],
            max_tokens=request["output_tokens"],
        ),
    )


class FormalInputContract(unittest.TestCase):
    def fixture(self):
        from .test_qwen_fp8_formal import archived_fixture

        args = archived_fixture()
        args[0].insert(1, fixture_prepared(args[1]["requests"][0]))
        with fixture_tokenizer(args[2]) as tokenizer:
            self.assertTrue(validate(*args, tokenizer=tokenizer)[7].complete)
        return args

    def assert_blocked(self, args, reason):
        with fixture_tokenizer(args[2]) as tokenizer:
            result = lower_bound(*args, tokenizer=tokenizer)
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["seconds"])
        self.assertIsNone(result["efficiency"])
        self.assertIn(reason, " ".join(result["reasons"]))

    def test_sampling_is_bound_to_loaded_generation_and_frozen_request(self):
        for field in GENERATION:
            args = self.fixture()
            args[1]["sampling"].pop(field)
            with self.subTest(field=field):
                self.assert_blocked(args, "sampling")
        args = self.fixture()
        args[1].pop("sampling")
        self.assert_blocked(args, "sampling")
        args = self.fixture()
        args[1]["sampling"]["temperature"] = 0.5
        self.assert_blocked(args, "sampling")
        args = self.fixture()
        request = args[1]["requests"][0]
        request["body"]["temperature"] = 1.0
        request["body_sha256"] = digest(request["body"])
        self.assert_blocked(args, "request sampling")
        result = lower_bound(*self.fixture())
        self.assertIn("loaded tokenizer sampling", result["reasons"][0])
        self.assertIsNone(result["seconds"])

    def test_sampling_generation_bytes_and_loaded_values_must_match(self):
        args = self.fixture()
        with fixture_tokenizer(args[2]) as tokenizer:
            tokenizer.generation["temperature"] = 0.5
            result = lower_bound(*args, tokenizer=tokenizer)
            self.assertIn("loaded tokenizer", result["reasons"][0])
            self.assertIsNone(result["efficiency"])
            tokenizer.generation = dict(GENERATION)
            (tokenizer.path / "generation_config.json").write_text("{}")
            result = lower_bound(*args, tokenizer=tokenizer)
            self.assertIn("generation bytes", result["reasons"][0])
            self.assertIsNone(result["seconds"])

    def test_prepared_sampling_inventory_and_actual_values_are_required(self):
        for field, value in (
            ("worker_epoch", 9),
            ("request_id", 9),
            ("rpc_id", 0),
            ("prompt_token_sha256", "0" * 64),
            ("seed", 1),
            ("repetition_guard", False),
            ("min_tokens", 1),
            ("max_tokens", 1),
            ("sampling_f32_bits", {}),
        ):
            args = self.fixture()
            args[0][1][field] = value
            with self.subTest(field=field):
                self.assert_blocked(args, "prepared")
        args = self.fixture()
        args[0][1]["rust_intent"]["normalized_sampling_f32_bits"] = {}
        self.assert_blocked(args, "sampling intent")
        args = self.fixture()
        args[0][1]["rust_intent"]["seed"] = None
        self.assert_blocked(args, "prepared sampling seed")
        args = self.fixture()
        args[0].pop(1)
        self.assert_blocked(args, "missing prepared")
        args = self.fixture()
        args[0].insert(1, deepcopy(args[0][1]))
        self.assert_blocked(args, "prepared sampling inventory")

    def test_actual_dispatch_inventory_shape_precision_and_modes_are_required(self):
        for field, value in (
            ("rows", 1),
            ("max_queries", 1),
            ("verify", True),
            ("execution", "invented"),
            ("gdn_ba_precision", "bf16"),
            ("worker_epoch", 9),
            ("step_id", 9),
        ):
            args = self.fixture()
            args[0][2]["dispatches"][0][field] = value
            with self.subTest(field=field):
                self.assert_blocked(args, "dispatch")
        args = self.fixture()
        args[0][2]["dispatches"].append(deepcopy(args[0][2]["dispatches"][0]))
        self.assert_blocked(args, "dispatch")
        args = self.fixture()
        args[0][2]["dispatches"][0]["requests"][0]["lease_id"] = 2
        self.assert_blocked(args, "dispatch")
        args = self.fixture()
        args[0][2].pop("dispatches")
        self.assert_blocked(args, "dispatches")
        args = self.fixture()
        args[0][2]["dispatches"] = []
        self.assert_blocked(args, "dispatch")
