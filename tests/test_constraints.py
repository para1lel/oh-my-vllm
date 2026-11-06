import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import llguidance
import numpy as np
import torch
from oh_my_vllm.models.qwen38.request import Request
from oh_my_vllm.worker.constraints import Constraint
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tests.constraints_draft_chain import ConstraintsDraftChain as ConstraintsDraftChain
from tests.constraints_fixtures import AcceptDevice as AcceptDevice
from tests.constraints_fixtures import ChainDevice, ChainSample, sampling_mask
from tests.constraints_fixtures import ChainDraft as ChainDraft
from tests.constraints_fixtures import PackedAcceptance as PackedAcceptance
from tests.constraints_packed_masks import (
    ConstraintsPackedMasks as ConstraintsPackedMasks,
)
from tests.constraints_schema import ConstraintsSchema as ConstraintsSchema


class Constraints(unittest.TestCase):
    def test_sampling_inputs_use_contiguous_cpu_torch_tensors(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        inputs = SamplingInputs(2)
        for value in (
            inputs.full_history,
            inputs.presence_history,
            inputs.mask,
            inputs.parameters,
            inputs.random,
        ):
            with self.subTest(dtype=value.dtype):
                self.assertIsInstance(value, torch.Tensor)
                self.assertEqual(value.device.type, "cpu")
                self.assertTrue(value.is_contiguous())
        empty = SamplingInputs(2)
        empty.update([], [], "target")
        self.assertEqual(empty.parameters.shape, (2, 7))
        self.assertEqual(torch.count_nonzero(empty.parameters).item(), 0)
        self.assertTrue(empty.random.eq(0).all())
        self.assertFalse(empty.history_needed)
        self.assertTrue(empty.compact)
        zero = SamplingInputs(0)
        zero.update([], [], "target")
        self.assertEqual(zero.parameters.numel(), 0)

    def test_sampling_inputs_ignore_parameters_outside_the_requests(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=258,
            think_end=257,
            generation={},
            decode_bytes=bytes,
        )
        plain = Request(tokenizer, [], {"max_tokens": 8})
        guarded = Request(
            tokenizer, [], {"max_tokens": 8, "sampling": {"repetition_penalty": 2}}
        )
        inputs = SamplingInputs(2)
        inputs.update([plain, guarded], [[], []], "target")
        inputs.update([plain], [[]], "target")
        self.assertTrue(inputs.compact)
        self.assertFalse(inputs.history_needed)

    def test_sampling_inputs_preserve_uint64_seed_bits(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=258,
            think_end=257,
            generation={},
            decode_bytes=bytes,
        )

        def update(seed, stream):
            request = Request(
                tokenizer,
                [],
                {"max_tokens": 8, "sampling": {"seed": seed}},
            )
            inputs = SamplingInputs(1)
            inputs.update([request], [[]], stream)
            return inputs.random[0].tolist()

        proposal_seed = 7
        self.assertEqual(
            update(proposal_seed, "proposal"),
            [proposal_seed ^ 0x9E3779B97F4A7C15, 0],
        )
        target_seed = (1 << 63) + 5
        self.assertEqual(update(target_seed, "target"), [target_seed, 0])

    def test_tool_constraints_require_a_model_grammar(self):
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
        vocabulary["<eos>"] = 256
        vocabulary["<tool_call>"] = 257
        vocabulary["</tool_call>"] = 258
        tokenizer = Tokenizer(models.BPE(vocabulary, []))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(["<eos>", "<tool_call>", "</tool_call>"])
        grammar_tokenizer = llguidance.LLTokenizer(tokenizer.to_str(), eos_token=256)
        tools = [{"function": {"name": "choose", "parameters": {"type": "object"}}}]
        for choice in ("auto", "required", {"function": {"name": "choose"}}):
            with self.subTest(choice=choice):
                options = {"tools": tools, "tool_choice": choice}
                with self.assertRaisesRegex(ValueError, "model tool grammar"):
                    Constraint.for_request(grammar_tokenizer, options)
                constraint = Constraint.for_request(
                    grammar_tokenizer,
                    options,
                    tool_grammar=lambda request: (
                        'start: "selected_'
                        + request["tools"][0]["function"]["name"]
                        + '"'
                    ),
                )
                self.assertFalse(constraint.accepting)
                constraint.accept(tokenizer.encode("selected_choose").ids)
                self.assertTrue(constraint.accepting)
        for options in ({}, {"tools": tools, "tool_choice": "none"}):
            self.assertTrue(
                Constraint.for_request(grammar_tokenizer, options).accepting
            )
        request = Request(
            SimpleNamespace(grammar=grammar_tokenizer),
            [],
            {"max_tokens": 8, "tools": tools, "tool_choice": "required"},
        )
        self.assertFalse(request.constraint.accepting)
        request.constraint.accept(
            tokenizer.encode(
                '<tool_call>{"name":"choose","arguments":{}}</tool_call>'
            ).ids
        )
        self.assertTrue(request.constraint.accepting)

    def test_ignore_eos_keeps_sampling_masks_stop_strings_and_output_bounds(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={256, 258},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: b"".join(
                b"<eos>" if token in {256, 258} else bytes([token]) for token in tokens
            ),
        )
        for ignore in (None, False, True):
            with self.subTest(ignore_eos=ignore):
                options = {
                    "max_tokens": 8,
                    "stop": ["bc"],
                    "sampling": {"repetition_guard": False},
                }
                if ignore is not None:
                    options["ignore_eos"] = ignore
                request = Request(tokenizer, [], options)
                inputs = SamplingInputs(1)
                inputs.update([request], [[]], "target")
                self.assertEqual(
                    set(np.flatnonzero(sampling_mask(inputs))), set(range(259))
                )
                self.assertEqual(request.stops_after([97, 256, 98]), not ignore)
                self.assertTrue(request.stops_after([97, 256, 98, 258, 99]))
                kept, text = request.append([97, 256, 98])
                self.assertEqual(text, "a")
                if ignore:
                    self.assertEqual(kept, [97, 256, 98])
                    self.assertIsNone(request.finish)
                    self.assertTrue(request.stops_after([258, 99]))
                    self.assertEqual(request.append([258, 99, 100]), ([258, 99], ""))
                else:
                    self.assertEqual(kept, [97, 256])
                self.assertEqual(request.finish, "stop")
        request = Request(tokenizer, [], {"max_tokens": 4, "ignore_eos": True})
        self.assertFalse(request.stops_after([256, 258]))
        self.assertTrue(request.stops_after([256, 258, 97, 256]))
        self.assertEqual(
            request.append([256, 258, 97, 256, 98]), ([256, 258, 97, 256], "a")
        )
        self.assertEqual(request.finish, "length")
        request = Request(
            tokenizer, [], {"max_tokens": 4, "min_tokens": 2, "ignore_eos": True}
        )
        self.assertFalse(request.mask()[256])

    def test_verification_distribution_does_not_reuse_sampling_graph(self):
        from oh_my_vllm.models.qwen38.sampling import Sampling

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
        )
        request = Request(tokenizer, [4], {"max_tokens": 16, "sampling": {"seed": 7}})
        calls = []

        class Sample(ChainSample):
            def draw(self, settings, rng):
                calls.append("draw")
                return super().draw(settings, rng)

            def distribution(self, settings):
                calls.append("distribution")
                self.outputs.probabilities[:] = 0.5
                return self.outputs

        with patch("oh_my_vllm.models.qwen38.sampling.prepare", Sample):
            sample = Sampling(ChainDevice(), None, 1)
            sample.operation.outputs.hidden[:] = 3
            sample.rng[:] = 91
            for graph in (False, True, True):
                for draw in (False, True, False):
                    before = request.random_offset
                    tokens, probabilities = sample.select_from_logits(
                        [request], graph, sample=draw
                    )
                    self.assertEqual(calls[-1], "draw" if draw else "distribution")
                    self.assertEqual(request.random_offset, before + draw)
                    if draw:
                        self.assertEqual(tokens.tolist(), [3])
                    else:
                        self.assertIsNone(tokens)
                        self.assertTrue(np.all(probabilities == 0.5))
            request.options["sampling"]["temperature"] = 0
            before = request.random_offset
            sample.select_from_logits([request], True, sample=False)
            self.assertEqual(calls[-1], "draw")
            self.assertEqual(sample.operation.outputs.tokens.tolist(), [3])
            self.assertEqual(request.random_offset, before)
            sample.close()

    def test_repetition_mask_requires_complete_cycles(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={248319},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=248320,
            think_end=248318,
            decode_bytes=bytes,
            generation={},
        )
        broken = [1, 2, 3, 4, 5, 6, 7] * 11
        broken[-20] = 9
        cases = [
            ([7] * 63, set()),
            ([7] * 64, {7}),
            ([3, 5] * 32, {3}),
            ([7] * 64 + [8], set()),
            (broken, set()),
            (list(range(256)) * 3, {0}),
            (list(range(257)) * 3, set()),
        ]
        for history, blocked in cases:
            with self.subTest(length=len(history), final_token=history[-1]):
                request = Request(
                    tokenizer, [], {"max_tokens": 4096, "sampling": {"seed": 1}}
                )
                request.history = history
                inputs = SamplingInputs(1)
                inputs.update([request], [[]], "proposal")
                inputs.update([request], [[]], "target")
                self.assertEqual(
                    set(np.flatnonzero(sampling_mask(inputs) == 0)), blocked
                )
                request.append([8])
                inputs.update([request], [[]], "target")
                self.assertFalse(np.any(sampling_mask(inputs) == 0))

    def test_sampling_inputs_do_not_retain_speculative_history(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={"presence_penalty": 0.5},
            decode_bytes=lambda tokens: bytes(tokens),
        )
        request = Request(
            tokenizer,
            [4, 4],
            {"max_tokens": 16, "sampling": {"seed": 7, "repetition_guard": False}},
        )
        request.append([5])
        request.random_offset, request.proposal_offset = 13, 17
        inputs = SamplingInputs(2)
        inputs.update([request, request], [[6, 6], [7, 8]], "target")
        self.assertTrue(inputs.history_needed)
        self.assertTrue(inputs.compact)
        self.assertEqual(inputs.history.dtype, torch.uint8)
        self.assertEqual(inputs.history.shape, (2, 248320 // 8))
        self.assertEqual(inputs.history[:, 0].tolist(), [0b01100000, 0b10100000])
        self.assertEqual(inputs.history[:, 1].tolist(), [0, 1])
        self.assertFalse(inputs.history[:, -1].any())
        self.assertFalse(request.counts[1, 6:8].any())
        np.testing.assert_array_equal(inputs.random, [[7, 13], [7, 13]])
        self.assertFalse(
            any(sampling_mask(inputs, row)[259:].any() for row in range(2))
        )

        request.options["sampling"]["presence_penalty"] = 0
        inputs.update([request, request], [[], []], "proposal")
        self.assertFalse(inputs.history_needed)
        self.assertEqual(inputs.random[0, 0], 7 ^ 0x9E3779B97F4A7C15)
        self.assertEqual(inputs.random[0, 1], 17)
        request.options["sampling"]["frequency_penalty"] = 0.25
        inputs.update([request, request], [[6, 6], []], "target")
        self.assertTrue(inputs.history_needed)
        self.assertFalse(inputs.compact)
        self.assertEqual(inputs.history[:, 0, 4].tolist(), [2, 2])
        self.assertEqual(inputs.history[:, 1, 6].tolist(), [2, 0])
        request.options["sampling"].update(frequency_penalty=0, repetition_penalty=1.1)
        inputs.update([request, request], [[], []], "target")
        self.assertFalse(inputs.compact)
        self.assertEqual(request.history, [4, 4, 5])
        self.assertEqual((request.random_offset, request.proposal_offset), (13, 17))

    def test_sampling_mask_cache_tracks_output_and_thinking_boundaries(self):
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
        )
        request = Request(
            tokenizer,
            [],
            {
                "max_tokens": 128,
                "min_tokens": 2,
                "sampling": {"repetition_guard": False},
            },
        )
        inputs = SamplingInputs(1)

        def update(prefix, changed):
            inputs.update([request], [prefix], "target")
            self.assertEqual(inputs.mask_changed, changed)
            return set(np.flatnonzero(sampling_mask(inputs)))

        self.assertNotIn(256, update([], True))
        self.assertNotIn(256, update([8], False))
        self.assertIn(256, update([8, 9], True))
        self.assertIn(256, update([9, 8], False))
        request.thinking = True
        request.options["thinking_token_budget"] = 1
        self.assertEqual(update([8], True), {257})
        self.assertEqual(update([9], False), {257})
        self.assertEqual(update([257], True), set(range(259)) - {256})
        request.thinking = False
        request.options["sampling"]["repetition_guard"] = True
        request.history = [7] * 64
        self.assertNotIn(7, update([], True))
        self.assertIn(7, update([8], True))
        request.options["sampling"]["repetition_guard"] = False
        tokenizer.valid_tokens = 258
        self.assertNotIn(258, update([], True))
        self.assertNotIn(258, update([], False))
        request.options["sampling"]["repetition_guard"] = True
        update([], True)
        constraint = weakref.ref(request.constraint)
        request = None
        self.assertIsNone(constraint())
