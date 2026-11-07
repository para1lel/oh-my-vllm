import hashlib
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.draft_chain import DraftChain
from oh_my_vllm.models.qwen38.speculation_ops import Draft
from oh_my_vllm.operators.fa_fp8.api import CacheView

from ..cases import load_cases
from ..fixtures import load_fixture
from ..run import PreparedCase
from ..storage import state_rows
from . import mtp
from .mtp_cache import FixtureCache, repage


class CPUDevice:
    def empty(self, shape, dtype):
        return torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")))

    def copy(self, target, source):
        target.view(torch.uint8).reshape(-1).copy_(source.view(torch.uint8).reshape(-1))

    def download(self, tensor, dtype):
        return tensor.clone().view(getattr(torch, dtype))

    def synchronize(self):
        pass


def fixture(case_id):
    case = next(case for case in load_cases() if case.id == case_id)
    return case, load_fixture(case)


def draft_stub(device, weights, batch, first, *, table_width):
    stage = object.__new__(Draft)
    stage.first, stage.previous_hidden = first, None
    stage.hidden = stage.tokens = stage.rows = stage.boundary_rows = object()
    stage.zero = object()
    stage.weights = weights
    stage.ffn = SimpleNamespace(outputs=object())
    stage.operation = SimpleNamespace(project=Mock())
    stage.sampling = SimpleNamespace(
        inputs=SimpleNamespace(),
        operation=SimpleNamespace(
            outputs=SimpleNamespace(
                **{
                    name: object()
                    for name in (
                        "hidden",
                        "logits",
                        "tokens",
                        "selected_probability",
                        "probabilities",
                    )
                }
            )
        ),
        draw=Mock(),
    )
    stage.close = stage.copy = Mock()
    stage.sampling.presence_history = object()
    stage.sampling.operation.advance_presence = Mock()
    stage.table_width = table_width
    return stage


def chain_stub(device, stages, count, greedy):
    chain = object.__new__(DraftChain)
    chain.device, chain.stages = device, stages
    chain.tokens = object()
    chain.token_rows = [object() for _ in stages]
    chain.close = Mock()
    return chain


class MTPAdapter(unittest.TestCase):
    def test_production_draft_rejects_the_old_single_tensor_cache(self):
        stage = draft_stub(None, None, 1, False, table_width=7)
        stage.metadata = tuple(object() for _ in range(5))
        stage.cache = (object(),)
        with self.assertRaisesRegex(
            TypeError, "missing 1 required positional argument: 'tables'"
        ):
            stage.run()

    def test_fixed_fixtures_repage_without_changing_the_reference(self):
        cases = [
            case
            for case in load_cases()
            if case.purpose == "numerics"
            and case.boundary in {"mtp.draft", "mtp.chain"}
        ]
        for case in cases:
            with self.subTest(case=case.id):
                data = load_fixture(case)
                before = {
                    name: hashlib.sha256(value.tobytes()).hexdigest()
                    for name, value in data.expected.items()
                }
                prefixes = (
                    tuple(f"stage{i}." for i in range(4))
                    if case.boundary == "mtp.chain"
                    else ("",)
                )
                converted = repage(data, prefixes)
                self.assertEqual(converted.codes.dtype, np.uint8)
                self.assertEqual(converted.scales.dtype, np.float32)
                self.assertEqual(converted.codes.shape[1:], (2, 128, 4, 256))
                self.assertEqual(converted.scales.shape, converted.codes.shape[:-1])
                self.assertEqual(converted.tables.shape, (case.shape["batch"], 7))
                self.assertTrue(np.all(converted.scales > 0))
                for prefix in prefixes:
                    positions = data.inputs[prefix + "lengths"] - 1
                    expected = (
                        converted.tables[np.arange(len(positions)), positions // 128]
                        * 128
                        + positions % 128
                    )
                    np.testing.assert_array_equal(converted.slots[prefix], expected)
                written = converted.written
                if case.boundary == "mtp.chain":
                    expected = np.stack(
                        [converted.slots[p] for p in prefixes[1:]], axis=1
                    ).reshape(-1)
                elif case.options["variant"] == "query_only":
                    expected = []
                else:
                    expected = converted.slots[""]
                np.testing.assert_array_equal(written, expected)
                self.assertEqual(
                    before,
                    {
                        name: hashlib.sha256(value.tobytes()).hexdigest()
                        for name, value in data.expected.items()
                    },
                )

    def test_missing_context_and_inconsistent_write_metadata_fail(self):
        _, data = fixture("mtp.chain.numerics.batch-1")
        prefixes = tuple(f"stage{i}." for i in range(4))
        for changed, error in (
            ("tables", "lacks a stored KV page"),
            ("slots", "slots differ"),
            ("sequence_rows", "one query per request"),
        ):
            inputs = dict(data.inputs)
            name = "stage0." + changed
            inputs[name] = inputs[name].copy()
            inputs[name].fill(0)
            with self.subTest(field=changed), self.assertRaisesRegex(ValueError, error):
                repage(replace(data, inputs=inputs), prefixes)

    def test_draft_binds_the_production_cache_view_with_fp8_and_new_pages(self):
        case, data = fixture("mtp.draft.numerics.draft-2")
        with PreparedCase(case, CPUDevice(), data) as prepared:
            prepared.weights = object()
            with patch.object(mtp, "Draft", side_effect=draft_stub):
                mtp.draft(prepared)
            prepared.run()
            stage = prepared.owners[-1]
            binding = stage.operation.project.call_args.args[1]
            self.assertIsInstance(binding.state, CacheView)
            self.assertEqual(binding.state.codes.dtype, torch.uint8)
            self.assertEqual(binding.state.scales.dtype, torch.float32)
            self.assertEqual(binding.state.tables.shape, (1, 7))
            self.assertEqual(binding.state.slots.tolist(), [911])
            self.assertEqual(stage.table_width, 7)
            self.assertIs(prepared.fixture, data)

    def test_chain_shares_codes_and_scales_and_uses_each_stage_metadata(self):
        case, data = fixture("mtp.chain.numerics.batch-2")
        with PreparedCase(case, CPUDevice(), data) as prepared:
            prepared.weights = object()
            with (
                patch.object(mtp, "Draft", side_effect=draft_stub),
                patch.object(mtp, "DraftChain", side_effect=chain_stub),
            ):
                mtp.chain(prepared)
            prepared.run()
            stages = prepared.owners[-1].stages
            for index, stage in enumerate(stages):
                state = stage.operation.project.call_args.args[1].state
                self.assertIsInstance(state, CacheView)
                self.assertIs(state.codes, stages[0].cache[0])
                self.assertIs(state.scales, stages[0].cache[1])
                self.assertEqual(state.tables.shape, (2, 7))
                self.assertEqual(state.slots.tolist(), [911 + index, 1807 + index])
            self.assertIs(prepared.fixture, data)

    def test_cache_observation_reset_and_both_guards_use_fp8_storage(self):
        case, data = fixture("mtp.draft.numerics.draft-2")
        case = replace(case, outputs=("kv", "rng", "guards"))
        with PreparedCase(case, CPUDevice(), data) as prepared:
            cache = prepared.own(FixtureCache(prepared, ("",)))
            selected = int(cache.initial.written[0])
            codes, scales = cache.cache
            page, offset = divmod(selected, 128)
            codes[page, :, offset].fill_(0x7E)
            scales[page, :, offset].fill_(np.float32(1 / 448))
            observed = prepared.observe()
            np.testing.assert_array_equal(observed["kv"], np.ones((1, 2, 4, 256)))
            self.assertTrue(observed["guards"].all())
            codes[0, 0, 0, 0, 0] ^= 1
            self.assertFalse(prepared.observe()["guards"].all())
            prepared.reset()
            self.assertTrue(prepared.observe()["guards"].all())
            scales[0, 0, 0, 0] *= 2
            self.assertFalse(prepared.observe()["guards"].all())
            prepared.reset()
            np.testing.assert_array_equal(prepared.raw(codes), cache.initial.codes)
            np.testing.assert_array_equal(prepared.raw(scales), cache.initial.scales)
            prepared.stage_resets()
            codes[0, 0, 0, 0, 0] ^= 1
            scales[0, 0, 0, 0] *= 2
            prepared.reset()
            self.assertTrue(prepared.observe()["guards"].all())
            cache.metadata[""][2][0, 0] = 0
            with self.assertRaisesRegex(ValueError, "read-only"):
                prepared.observe()

    def test_repage_preserves_rows_across_both_page_boundaries(self):
        _, data = fixture("mtp.chain.numerics.batch-1")
        prefixes = tuple(f"stage{i}." for i in range(4))
        bits = data.state["kv"].copy()
        table = data.inputs["stage0.tables"][0]
        for position, value in (
            (127, 0x3F00),
            (128, 0x3F80),
            (783, 0x4000),
            (784, 0x4080),
        ):
            bits[table[position // 784], :, position % 784] = value
        converted = repage(replace(data, state={**data.state, "kv": bits}), prefixes)
        from ..qwen_fp8_reference import attention_bf16_vectors

        restored = state_rows(
            attention_bf16_vectors(converted.codes, converted.scales), "kv"
        )
        for position, value in (
            (127, 0x3F00),
            (128, 0x3F80),
            (783, 0x4000),
            (784, 0x4080),
        ):
            slot = converted.tables[0, position // 128] * 128 + position % 128
            np.testing.assert_array_equal(
                restored[slot], np.full((2, 4, 256), value, np.uint16)
            )
