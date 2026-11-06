import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38.request import Request

from tests.constraints_fixtures import ChainDevice, ChainDraft, ChainSample


class ConstraintsPackedMasks(unittest.TestCase):
    def test_packed_grammar_masks_match_existing_sampler_rules(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer, request = self._packed_mask_request()
        for thinking, generated, history, prefix in (
            (False, 1, [4, 5], []),
            (False, 70, [4] + [7] * 70, []),
            (False, 70, [4] + [7] * 70, [8]),
            (False, 62, [4] + [7] * 62, [7]),
            (False, 63, [4] + [7] * 63, [7]),
            (False, 64, [4] + [7] * 63 + [8], [8, 8, 8]),
            (True, 2, [4, 5, 6], [8]),
        ):
            request.history = history
            request.options["thinking_token_budget"] = 1
            for valid in (258, 259, 264):
                tokenizer.valid_tokens = valid
                mask = np.ones(248320, np.uint8)
                mask[10] = 0
                request.draft_guard_position = None
                expected = request.mask(
                    request.constraint, thinking, generated, 2, prefix, mask.copy()
                )
                request.prepare_draft_guard()
                raw = np.packbits(mask, bitorder="little")
                for _ in range(2):
                    packed = request.packed_mask(raw, thinking, generated, 2, prefix)
                    np.testing.assert_array_equal(
                        np.unpackbits(packed, bitorder="little"), expected
                    )
        self.assertLessEqual(len(request.filtered_masks), 32)
        request.history = [4, 5, 6]
        request.prepare_draft_guard()
        self.assertTrue(request.draft_guard_safe)
        self.assertFalse(request.needs_repetition_guard([7, 8, 9]))
        self.assertTrue(request.needs_repetition_guard([7, 8, 9, 10]))
        request.append([7])
        self.assertTrue(request.needs_repetition_guard([]))
        request.options["sampling"]["repetition_guard"] = False
        for blocked in (*range(40), 0):
            tokenizer.eos = {256 if blocked % 2 else 258}
            mask = np.ones(248320, np.uint8)
            mask[blocked] = 0
            expected = request.mask(thinking=False, generated=0, mask=mask.copy())
            packed = request.packed_mask(
                np.packbits(mask, bitorder="little"), False, 0, 0, []
            )
            np.testing.assert_array_equal(
                np.unpackbits(packed, bitorder="little"), expected
            )
        self.assertEqual(len(request.filtered_masks), 32)
        for eos in ({256}, {258}, {256}):
            tokenizer.eos = eos
            expected = request.mask(thinking=False, generated=0, mask=mask.copy())
            packed = request.packed_mask(
                np.packbits(mask, bitorder="little"), False, 0, 0, []
            )
            np.testing.assert_array_equal(
                np.unpackbits(packed, bitorder="little"), expected
            )
        single = SamplingInputs(5)
        with patch("oh_my_vllm.models.qwen38.sampling.batch_masks") as masks:
            request.constraint.matcher = None
            single.update([request] * 5, [[]] * 5, "target")
            self.assertFalse(masks.called)

    def test_sampler_packed_upload_reuses_and_restores_bindings(self):
        from oh_my_vllm.models.qwen38.sampling import Sampling

        class Device(ChainDevice):
            def empty(self, shape, dtype):
                dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
                return torch.empty(shape, dtype=dtype)

            def scratch(self, shape, dtype):
                return self.empty(shape, dtype)

            def zeros(self, shape, dtype):
                return self.empty(shape, dtype).zero_()

            def view(self, value, shape, offset=0):
                return value.reshape(-1)[offset : offset + int(np.prod(shape))].reshape(
                    shape
                )

            def copy(self, destination, source):
                destination.copy_(source)

        device = Device()
        with patch("oh_my_vllm.models.qwen38.sampling.prepare", ChainSample):
            sampler = Sampling(device, None, 2)
        original = sampler.separate_inputs
        sampler.inputs.update([], [], "target")
        sampler.upload_packed(True)
        first = sampler.packed_input_key
        sampler.inputs.parameters_view[:] = 3
        sampler.inputs.random_view[:] = 7
        sampler.upload_packed(True)
        self.assertTrue(torch.all(sampler.parameters == 3))
        sampler.inputs.compact = False
        sampler.inputs.history = sampler.inputs.full_history
        sampler.upload_packed(False)
        sampler.inputs.compact = True
        sampler.inputs.history = sampler.inputs.presence_history
        sampler.inputs.parameters_view[:] = 9
        sampler.upload_packed(True)
        self.assertEqual(sampler.packed_input_key, first)
        self.assertTrue(torch.all(sampler.parameters == 9))
        sampler.inputs.mask_view[:] = 0
        sampler.inputs.mask_changed = False
        sampler.restore_inputs()
        self.assertTrue(sampler.inputs.mask_changed)
        sampler.uploads.copy(sampler.mask, sampler.inputs.mask)
        self.assertFalse(sampler.mask.any())
        for before, after in zip(original, sampler.separate_inputs, strict=True):
            self.assertIs(before, after)
        self.assertIs(sampler.mask, original[2])
        self.assertIsNone(sampler.packed_input_key)
        sampler.close()

    def test_draft_row_views_follow_storage_bindings(self):
        from oh_my_vllm.models.qwen38.speculation import Draft

        device = ChainDevice()
        draft = Draft.__new__(Draft)
        draft.device = device
        draft._row_bindings = draft._row_views = None
        for rows in (2, 4, 2):
            draft.hidden = np.full((rows, 5120), rows, np.float32)
            draft.sampling = SimpleNamespace(
                operation=SimpleNamespace(
                    outputs=SimpleNamespace(
                        probabilities=np.zeros((rows, 248320), np.float32),
                        hidden=np.full((rows, 5120), rows + 1, np.float32),
                    )
                )
            )
            cached = draft.row_views()
            self.assertIs(draft.row_views(), cached)
            for source, views in zip(draft._row_bindings, cached, strict=True):
                self.assertEqual(len(views), rows)
                for index, view in enumerate(views):
                    self.assertTrue(np.shares_memory(source, view))
                    np.testing.assert_array_equal(view, source[index : index + 1])
            old = cached
            draft.sampling.operation.outputs.hidden = np.full(
                (rows, 5120), rows + 2, np.float32
            )
            cached = draft.row_views()
            self.assertIsNot(cached, old)
            self.assertFalse(np.shares_memory(cached[2][0], old[2][0]))
            np.testing.assert_array_equal(cached[2][0], rows + 2)

    def test_draft_row_cache_retains_sources_until_rebind(self):
        from oh_my_vllm.models.qwen38.speculation import Draft

        draft = Draft.__new__(Draft)
        draft.device = ChainDevice()
        draft._row_bindings = draft._row_views = None
        draft.hidden = np.zeros((2, 5120), np.float32)
        outputs = SimpleNamespace(
            probabilities=np.zeros((2, 248320), np.float32),
            hidden=np.zeros((2, 5120), np.float32),
        )
        draft.sampling = SimpleNamespace(operation=SimpleNamespace(outputs=outputs))
        old_views = draft.row_views()
        source = weakref.ref(draft.hidden)
        draft.hidden = None
        self.assertIsNotNone(source())
        draft.hidden = np.full((4, 5120), 7, np.float32)
        outputs.probabilities = np.zeros((4, 248320), np.float32)
        outputs.hidden = np.zeros((4, 5120), np.float32)
        new_views = draft.row_views()
        self.assertIsNot(new_views, old_views)
        self.assertEqual(len(new_views[0]), 4)
        self.assertFalse(np.shares_memory(new_views[0][0], old_views[0][0]))
        np.testing.assert_array_equal(new_views[0][0], 7)
        del old_views
        self.assertIsNone(source())

    def test_draft_cached_rows_track_batch_requests_and_page_tables(self):
        from oh_my_vllm.models.qwen38.speculation import Speculation

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: bytes(tokens),
        )

        def make_request():
            value = Request(
                tokenizer,
                [4, 4],
                {
                    "max_tokens": 32,
                    "min_tokens": 16,
                    "sampling": {"temperature": 0, "repetition_guard": False},
                },
            )
            value.append([5])
            value.state_position = 2
            return value

        device = ChainDevice()
        speculation = Speculation(
            device,
            None,
            SimpleNamespace(fa=[None], blocks=8),
            None,
            SimpleNamespace(record=lambda *args, **kwargs: None),
        )
        with (
            patch("oh_my_vllm.models.qwen38.speculation.Draft", ChainDraft),
            patch("oh_my_vllm.models.qwen38.speculation.can_chain", return_value=False),
            patch("oh_my_vllm.models.qwen38.sampling.prepare", ChainSample),
            patch("oh_my_vllm.models.qwen38.sampling.Sampling.upload_packed"),
        ):
            for count, table, hidden in ((2, 1, 2), (4, 2, 4), (2, 3, 6), (1, 4, 8)):
                requests = [make_request() for _ in range(count)]
                batch = SimpleNamespace(
                    requests=requests,
                    scheduled=[
                        {"request_id": i + 1, "fa_block_table": [table]}
                        for i in range(count)
                    ],
                    metadata=(None, None, np.full((count, 335), table, np.int32)),
                )
                endpoints = {
                    i: (np.full((1, 5120), hidden, np.float32), None)
                    for i in range(count)
                }
                if count == 1:
                    with patch.object(
                        ChainDraft, "row_views", side_effect=AssertionError
                    ):
                        speculation.propose(batch, endpoints, False)
                else:
                    speculation.propose(batch, endpoints, False)
                expected = [hidden + 1]
                for _ in range(3):
                    expected.append(2 * expected[-1] + 1)
                for value in requests:
                    self.assertEqual(value.drafts, expected)
                for first in (True, False):
                    operation = speculation.drafts[count, first]
                    np.testing.assert_array_equal(operation.metadata_arrays[2], table)
                    stage = 0 if first else 3
                    np.testing.assert_array_equal(
                        operation.metadata_arrays[1], table * 784 + 2 + stage
                    )

    def _packed_mask_request(self):
        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: bytes(tokens),
        )
        request = Request(
            tokenizer,
            [4],
            {
                "max_tokens": 128,
                "min_tokens": 64,
                "sampling": {"repetition_guard": True},
            },
        )
        return tokenizer, request
