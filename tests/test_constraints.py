import unittest
import weakref
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import llguidance
import numpy as np
from oh_my_vllm.models.qwen38.request import Request
from oh_my_vllm.models.qwen38.tool_grammar import tool_grammar
from oh_my_vllm.worker.constraints import Constraint, batch_masks
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def sampling_mask(inputs, index=0):
    return np.unpackbits(inputs.mask[index], bitorder="little")


class ChainDevice:
    def __init__(self):
        self.downloads = 0
        self.module = SimpleNamespace(copy_bytes=self.copy, gather_rows=self.gather)

    def empty(self, shape, dtype):
        return np.empty(shape, np.float32 if dtype == "bfloat16" else dtype)

    def scratch(self, shape, dtype):
        return self.empty(shape, dtype)

    def zeros(self, shape, dtype):
        return np.zeros(shape, np.float32 if dtype == "bfloat16" else dtype)

    def upload(self, value):
        return value.copy()

    def view(self, value, shape, offset=0):
        return value.reshape(-1)[offset : offset + int(np.prod(shape))].reshape(shape)

    def copy(self, destination, source):
        np.copyto(
            destination,
            source if isinstance(source, np.ndarray) else np.from_dlpack(source),
        )

    def uploads(self):
        return SimpleNamespace(copy=self.copy, close=lambda: None)

    def gather(self, destination, source, indices):
        destination[:] = source[indices]

    def download(self, value, dtype):
        self.downloads += 1
        return value.astype(dtype, copy=True)

    def synchronize(self):
        pass

    @contextmanager
    def graph(self, operation):
        yield operation


class ChainSample:
    def __init__(self, plan, device, weights):
        rows, allocate = plan.rows, device.empty
        self.outputs = SimpleNamespace(
            hidden=allocate((rows, 5120), "bfloat16"),
            probabilities=allocate((rows, 248320), "float32"),
            tokens=allocate((rows,), "int64"),
        )

    def project(self, weights, inputs):
        self.outputs.hidden[:] = inputs.value + inputs.residual + 1

    def draw(self, settings, rng):
        rng[:, 1] += (settings.parameters[:, 0] > 0).astype(np.uint64)
        self.outputs.tokens[:] = self.outputs.hidden[:, 0].astype(np.int64)
        self.outputs.probabilities[:] = 0
        self.outputs.probabilities[
            np.arange(len(self.outputs.hidden)), self.outputs.tokens
        ] = 1
        return self.outputs

    def advance_presence(self, history):
        for row, token in enumerate(self.outputs.tokens):
            history[row, token // 8] |= 1 << (token % 8)

    def close(self):
        pass

    @contextmanager
    def graph_lease(self):
        yield


class ChainDraft:
    def __init__(self, device, weights, batch, first):
        from oh_my_vllm.models.qwen38.sampling import Sampling

        self.device, self.first = device, first
        self.hidden = device.zeros((batch, 5120), "bfloat16")
        self.tokens = device.zeros((batch,), "int64")
        self.rows = device.upload(np.arange(batch, dtype=np.int32))
        self.ffn = SimpleNamespace(outputs=(self.hidden.copy(), self.hidden.copy()))
        self.output = self.ffn.outputs
        self.sampling = Sampling(device, None, batch, hidden=self.output)
        self.metadata_inputs = SimpleNamespace(
            host_views=tuple(
                np.zeros(shape, dtype=dtype)
                for shape, dtype in (
                    ((3, batch), "int64"),
                    ((batch,), "int32"),
                    ((batch, 335), "int32"),
                    ((batch + 1,), "int32"),
                    ((batch,), "int32"),
                    ((batch,), "int64"),
                )
            ),
            update=lambda arrays: self.tokens.__setitem__(slice(None), arrays[-1]),
            close=lambda: None,
        )

    def run(self):
        self.output[0][:] = self.hidden + (0 if self.first else self.tokens[:, None])
        self.output[1][:] = 0
        self.sampling.project()
        return self.output

    def execute(self, graph):
        self.run()

    def copy(self, destination, source):
        self.device.copy(destination, source)

    @contextmanager
    def graph_lease(self):
        yield

    def close(self):
        self.sampling.close()


class Constraints(unittest.TestCase):
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
        self.assertEqual(inputs.history.dtype, np.uint8)
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

    def test_draft_chain_keeps_stage_order_private_probabilities_and_fallback(
        self,
    ):
        from oh_my_vllm.models.qwen38.draft_chain import can_chain
        from oh_my_vllm.models.qwen38.speculation import Speculation

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
            decode_bytes=lambda tokens: bytes(tokens),
        )
        options = {
            "max_tokens": 32,
            "min_tokens": 16,
            "stop": [],
            "sampling": {"temperature": 0, "repetition_guard": False},
        }

        def request(sampling=None):
            configured = deepcopy(options)
            configured["sampling"].update(sampling or {})
            value = Request(tokenizer, [4, 4], configured)
            value.append([5])
            value.state_position = 2
            return value

        d = ChainDevice()
        cache = SimpleNamespace(fa=[None], blocks=8)
        events = []
        trace = SimpleNamespace(
            record=lambda *args, **kwargs: events.append((args, kwargs))
        )
        speculation = Speculation(d, None, cache, None, trace)
        with (
            patch("oh_my_vllm.models.qwen38.speculation.Draft", ChainDraft),
            patch("oh_my_vllm.models.qwen38.sampling.prepare", ChainSample),
        ):
            first = request()
            batch = SimpleNamespace(
                requests=[first],
                scheduled=[{"request_id": 1, "fa_block_table": [1]}],
                metadata=(None, None, np.ones((1, 335), np.int32)),
            )
            endpoints = {0: (np.full((1, 5120), 2, np.float32), None)}
            first.proposal_offset = 17
            speculation.propose(batch, endpoints, True)
            self.assertEqual(first.drafts, [3, 7, 15, 31])
            self.assertEqual(d.downloads, 1)
            self.assertEqual(first.proposal_offset, 17)
            self.assertEqual(
                events,
                [
                    (
                        ("draft",),
                        {"stage": stage, "requests": [{"id": 1, "position": position}]},
                    )
                    for stage, position in enumerate((2, 3, 4, 5))
                ],
            )
            self.assertTrue(first.drafts_greedy)
            self.assertIsNone(first.proposals)
            saved = first.drafts.copy()
            second = request()
            batch.requests = [second]
            endpoints[0] = (np.full((1, 5120), 4, np.float32), None)
            speculation.propose(batch, endpoints, True)
            self.assertEqual(second.drafts, [5, 11, 23, 47])
            self.assertEqual(d.downloads, 2)
            self.assertEqual(first.drafts, saved)
            for seed, offset in ((7, 17), (19, 31)):
                stochastic = request(
                    dict(temperature=0.7, presence_penalty=0.5, seed=seed)
                )
                stochastic.proposal_offset = offset
                batch.requests = [stochastic]
                before = d.downloads
                speculation.propose(batch, endpoints, True)
                self.assertEqual(d.downloads - before, 1)
                self.assertEqual(stochastic.proposal_offset, offset + 4)
                self.assertFalse(stochastic.drafts_greedy)
                expected = np.zeros((4, 248320), np.float32)
                expected[np.arange(4), [5, 11, 23, 47]] = 1
                np.testing.assert_array_equal(stochastic.proposals, expected)
                endpoints[0] = (np.full((1, 5120), 2, np.float32), None)
                batch.requests = [request()]
                speculation.propose(batch, endpoints, True)
                np.testing.assert_array_equal(stochastic.proposals, expected)
                endpoints[0] = (np.full((1, 5120), 4, np.float32), None)
            for update in (
                {"frequency_penalty": 0.2},
                {"repetition_penalty": 1.2},
            ):
                rejected = request()
                rejected.options["sampling"].update(update)
                batch.requests = [rejected]
                before = d.downloads
                speculation.propose(batch, endpoints, True)
                with self.subTest(update=update):
                    self.assertEqual(d.downloads - before, 4)
            for field, value in (("thinking", True), ("minimum", 4), ("limit", 5)):
                rejected = request()
                setattr(rejected, field, value)
                batch.requests = [rejected]
                before = d.downloads
                speculation.propose(batch, endpoints, True)
                with self.subTest(field=field):
                    self.assertGreater(d.downloads - before, 1)
            rejected = request()
            batch.requests = [rejected]
            before = d.downloads
            speculation.propose(batch, endpoints, False)
            self.assertEqual(d.downloads - before, 4)
            rejected.options["stop"] = ["x"]
            before = d.downloads
            speculation.propose(batch, endpoints, True)
            self.assertEqual(d.downloads - before, 4)
            rejected.options["stop"] = []
            rejected.constraint.matcher = object()
            self.assertFalse(can_chain(batch, [0], True, cache.blocks))
            rejected.constraint.matcher = None
            for table in ([], [0], [8]):
                batch.scheduled[0]["fa_block_table"] = table
                with self.subTest(table=table):
                    self.assertFalse(can_chain(batch, [0], True, cache.blocks))
            speculation.close()

    def test_stochastic_chain_requires_prefix_independent_masks(self):
        from oh_my_vllm.models.qwen38.draft_chain import can_chain

        tokenizer = SimpleNamespace(
            eos={256},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=259,
            think_end=257,
            generation={},
        )
        request = Request(
            tokenizer,
            [4],
            {
                "max_tokens": 4096,
                "ignore_eos": True,
                "sampling": {"temperature": 1, "presence_penalty": 0.5},
            },
        )
        batch = SimpleNamespace(requests=[request], scheduled=[{"fa_block_table": [1]}])
        request.state_position = 1
        self.assertTrue(can_chain(batch, [0], True, 8))
        for history in (
            [1] * 60,
            [1] * 61,
            [1, 2] * 31,
            [1, 2] * 30 + [3],
            list(range(128)),
            list(range(23)) * 3,
        ):
            request.history = [4, *history]
            eligible = can_chain(batch, [0], True, 8)
            if history == [1] * 61 or history == [1, 2] * 31:
                self.assertFalse(eligible)
            if eligible:
                initial = request.mask()
                for length in range(4):
                    for prefix in product((1, 2, 3), repeat=length):
                        np.testing.assert_array_equal(
                            request.mask(
                                generated=request.generated + length, prefix=prefix
                            ),
                            initial,
                        )
        request.history = [4, *range(128)]
        self.assertTrue(can_chain(batch, [0], True, 8))
        request.minimum = request.generated + 2
        self.assertFalse(can_chain(batch, [0], True, 8))
        request.minimum = 0
        request.ignore_eos = False
        self.assertFalse(can_chain(batch, [0], True, 8))

    def test_thinking_cannot_end_before_the_constraint_starts(self):
        tokenizer = SimpleNamespace(
            eos={256}, grammar=None, valid_tokens=259, think_end=257
        )
        request = Request(
            tokenizer,
            [112],
            {"max_tokens": 8, "effort": "high", "sampling": {"seed": 1}},
        )
        self.assertFalse(request.mask()[256])
        self.assertTrue(request.mask()[257])

    def test_minimum_length_does_not_reactivate_an_old_stop_match(self):
        tokenizer = SimpleNamespace(
            eos={256},
            grammar=None,
            decode_bytes=lambda tokens: bytes(tokens),
        )
        for stop, tokens, expected, finished in [
            ("x", b"xyz", "xyz", None),
            ("xy", b"axy", "a", "stop"),
        ]:
            request = Request(
                tokenizer,
                [112],
                {
                    "max_tokens": 8,
                    "min_tokens": 3,
                    "effort": "off",
                    "sampling": {"seed": 1},
                    "stop": [stop],
                },
            )
            chunks = []
            for token in tokens:
                predicted = request.stops_after([token])
                _, text = request.append([token])
                chunks.append(text)
                self.assertEqual(predicted, bool(request.finish))
            self.assertEqual("".join(chunks), expected)
            self.assertEqual(request.finish, finished)
        truncated = Request(
            tokenizer, [112], {"max_tokens": 1, "sampling": {"seed": 1}}
        )
        self.assertEqual(truncated.append([228]), ([228], "\ufffd"))
        self.assertEqual(truncated.finish, "length")

    def test_schema_masks_and_speculative_branches_keep_separate_state(self):
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
        grammar_tokenizer = llguidance.LLTokenizer(
            tokenizer.to_str(), n_vocab=248320, eos_token=256
        )
        constraint = Constraint.for_request(
            grammar_tokenizer,
            {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"enum": [1, 2]}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                },
                "tools": [],
                "tool_choice": "none",
            },
        )
        constraint.accept(tokenizer.encode('{"x":').ids)
        self.assertFalse(constraint.mask()[tokenizer.encode("3").ids[0]])
        from oh_my_vllm.models.qwen38.sampling import SamplingInputs

        request = Request(
            SimpleNamespace(
                eos={256},
                grammar=grammar_tokenizer,
                valid_tokens=259,
                think_end=257,
                generation={},
                decode_bytes=lambda tokens: tokenizer.decode(tokens).encode(),
            ),
            [],
            {
                "max_tokens": 16,
                "ignore_eos": True,
                "sampling": {"repetition_guard": False},
            },
        )
        request.constraint = constraint
        inputs = SamplingInputs(1)
        for prefix in ([], tokenizer.encode("1}").ids, []):
            inputs.update([request], [prefix], "target")
            self.assertTrue(inputs.mask_changed)
            self.assertEqual(bool(sampling_mask(inputs)[256]), bool(prefix))
        thinking = Request(request.tokenizer, [], {"max_tokens": 16, "effort": "high"})
        thinking.constraint = constraint.fork()
        plain = Request(request.tokenizer, [], {"max_tokens": 16})
        inputs = SamplingInputs(4)
        inputs.update(
            [request, thinking, plain, request],
            [[], [], [], tokenizer.encode("2}").ids],
            "target",
        )
        self.assertFalse(sampling_mask(inputs, 0)[tokenizer.encode("3").ids[0]])
        self.assertTrue(sampling_mask(inputs, 1)[tokenizer.encode("3").ids[0]])
        self.assertFalse(sampling_mask(inputs, 1)[256])
        self.assertTrue(sampling_mask(inputs, 2)[256])
        self.assertTrue(sampling_mask(inputs, 3)[256])
        self.assertFalse(request.constraint.accepting)
        before = constraint.mask()
        for paths in (("", "1", "1}"), ("1}", "2}"), ("1}", "")):
            prefixes = [tokenizer.encode(path).ids for path in paths]
            inputs = SamplingInputs(len(paths))
            inputs.update([request] * len(paths), prefixes, "target")
            for index, prefix in enumerate(prefixes):
                branch = constraint.fork()
                branch.accept(prefix)
                np.testing.assert_array_equal(
                    sampling_mask(inputs, index), request.mask(branch)
                )
            np.testing.assert_array_equal(constraint.mask(), before)
        for text in ("3", "1}3"):
            inputs = SamplingInputs(1)
            for _ in range(2):
                with self.assertRaisesRegex(
                    ValueError, "violates the output constraint"
                ):
                    inputs.update([request], [tokenizer.encode(text).ids], "target")
            inputs.update([request], [[]], "target")
            np.testing.assert_array_equal(sampling_mask(inputs), request.mask())
            np.testing.assert_array_equal(constraint.mask(), before)
        inputs = SamplingInputs(1)
        inputs.update([request], [[*tokenizer.encode("1}").ids, 256, 256]], "target")
        self.assertTrue(sampling_mask(inputs)[256])
        np.testing.assert_array_equal(constraint.mask(), before)
        constrained = Request(request.tokenizer, [], request.options)
        constrained.constraint = constraint.fork()
        tokens = [*tokenizer.encode("1}").ids, 256, 256]
        self.assertEqual(constrained.append(tokens), (tokens, "1}"))
        self.assertIsNone(constrained.finish)
        inputs.update([constrained], [[256, 256]], "target")
        self.assertTrue(sampling_mask(inputs)[256])
        self.assertFalse(sampling_mask(inputs)[tokenizer.encode("3").ids[0]])
        proposal = constraint.fork()
        proposal.accept(tokenizer.encode("1}").ids)
        self.assertTrue(proposal.accepting)
        self.assertFalse(constraint.accepting)
        constraint.accept(tokenizer.encode("2}").ids)
        self.assertTrue(constraint.accepting)
        self.assertTrue(constraint.mask()[256])
        constraint.accept([256])
        self.assertTrue(constraint.accepting)
        tool_request = {
            "tools": [
                {
                    "function": {
                        "name": "choose",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"enum": [1, 2]}},
                            "required": ["x"],
                            "additionalProperties": False,
                        },
                    }
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
        tool = Constraint.for_request(
            grammar_tokenizer, tool_request, tool_grammar=tool_grammar
        )
        tool.accept(
            tokenizer.encode('<tool_call>{"name":"choose","arguments":{"x":').ids
        )
        self.assertFalse(tool.mask()[tokenizer.encode("3").ids[0]])
        branch = tool.fork()
        branch.accept(tokenizer.encode("1}}</tool_call>").ids)
        self.assertTrue(branch.accepting)
        self.assertFalse(tool.accepting)

        grammars = ('start: "a" | "ab" | "ac"', 'start: "ab" | "ad"')
        self.assertTrue(
            callable(getattr(Constraint, "all_of", None)),
            "generic grammar intersection factory is missing",
        )
        single = Constraint.all_of(grammar_tokenizer, grammars[:1])
        self.assertTrue(single.native_batch)
        np.testing.assert_array_equal(
            single.mask(), Constraint(grammar_tokenizer, grammars[0]).mask()
        )
        with self.assertRaises(ValueError):
            Constraint.all_of(grammar_tokenizer, [])
        with self.assertRaises(ValueError):
            Constraint.all_of(grammar_tokenizer, [None, grammars[0]])
        with self.assertRaises(ValueError):
            Constraint.all_of(grammar_tokenizer, [grammars[0], "start: missing"])
        combined = Constraint.all_of(grammar_tokenizer, grammars)
        self.assertFalse(combined.native_batch)
        self.assertIsNotNone(combined.matcher)
        self.assertFalse(combined.accepting)
        a, b, c, d = (tokenizer.encode(text).ids[0] for text in "abcd")
        before = combined.mask()
        self.assertEqual(np.flatnonzero(before).tolist(), [a])
        batch_masks([combined], [[]])
        saved_cache = dict(combined.prefix_masks)
        for invalid in ([a, c], [a, 256], [a, b, c]):
            with self.assertRaises(ValueError):
                combined.accept(iter(invalid))
            np.testing.assert_array_equal(combined.mask(), before)
            self.assertEqual(combined.prefix_masks, saved_cache)
            self.assertFalse(combined.accepting)
        with combined.at_prefix([a]):
            self.assertFalse(combined.accepting)
            self.assertEqual(np.flatnonzero(combined.mask()).tolist(), [b])
            self.assertFalse(combined.mask()[256])
            with self.assertRaises(ValueError):
                combined.accept([256])
            with (
                self.assertRaisesRegex(RuntimeError, "body failure"),
                combined.at_prefix([b]),
            ):
                self.assertTrue(combined.accepting)
                raise RuntimeError("body failure")
            self.assertFalse(combined.accepting)
            combined.accept([b])
            self.assertTrue(combined.accepting)
        np.testing.assert_array_equal(combined.mask(), before)
        self.assertEqual(combined.prefix_masks, saved_cache)
        for _ in range(2):
            with self.assertRaises(ValueError), combined.at_prefix([a, c]):
                self.fail("an invalid prefix entered the context")
            np.testing.assert_array_equal(combined.mask(), before)
            self.assertEqual(combined.prefix_masks, saved_cache)
        proposal = combined.fork()
        self.assertIsNot(proposal.cache_identity, combined.cache_identity)
        self.assertEqual(proposal.prefix_masks, {})
        self.assertIsNot(proposal.prefix_masks, combined.prefix_masks)
        proposal.accept([a, b, 256, 256])
        self.assertTrue(proposal.accepting)
        self.assertTrue(proposal.mask()[256])
        self.assertFalse(combined.accepting)
        self.assertEqual(proposal.prefix_masks, {})
        self.assertEqual(combined.prefix_masks, saved_cache)

        # The second grammar removes c and d; native batching must not use only one.
        prefixes = [[], [a], [a, b], [a, b, 256, 256]]
        for _ in range(2):
            constraints = [single, combined] * len(prefixes)
            paths = [path for path in prefixes for _ in range(2)]
            masks = batch_masks(constraints, paths)
            for member, path, mask in zip(constraints, paths, masks, strict=True):
                independent = member.fork()
                independent.accept(path)
                np.testing.assert_array_equal(mask, independent.mask())
            self.assertTrue(masks[2][c])
            self.assertFalse(masks[3][c])
            self.assertFalse(masks[3][d])
            np.testing.assert_array_equal(combined.mask(), before)
        with patch(
            "oh_my_vllm.worker.constraints.mask_executor",
            side_effect=AssertionError("composite reached the native executor"),
        ):
            batch_masks([combined] * len(prefixes), prefixes)
        with self.assertRaises(ValueError):
            batch_masks([combined, combined], [[a], [a, c]])
        np.testing.assert_array_equal(batch_masks([combined], [[]])[0], before)
        branching = Constraint.all_of(
            grammar_tokenizer,
            ['start: "ab" | "ba" | "ac"', 'start: "ab" | "ba" | "bd"'],
        )
        for paths in (([a, b], [b, a]), ([a], [], [b])):
            for path, mask in zip(
                paths, batch_masks([branching] * len(paths), paths), strict=True
            ):
                independent = branching.fork()
                independent.accept(path)
                np.testing.assert_array_equal(mask, independent.mask())
        self.assertEqual(np.flatnonzero(branching.mask()).tolist(), [a, b])
        unicode_tokens = tokenizer.encode("\u4e2da").ids
        unicode_constraint = Constraint.all_of(
            grammar_tokenizer,
            ['start: "\u4e2da" | "\u4e2db"', 'start: "\u4e2da" | "\u4e2dc"'],
        )
        for index, mask in enumerate(
            batch_masks(
                [unicode_constraint] * (len(unicode_tokens) + 1),
                [unicode_tokens[:index] for index in range(len(unicode_tokens) + 1)],
            )
        ):
            self.assertEqual(
                np.flatnonzero(mask).tolist(),
                [unicode_tokens[index]] if index < len(unicode_tokens) else [256],
            )
        self.assertFalse(unicode_constraint.accepting)

        repeated = Constraint.all_of(
            grammar_tokenizer, ['start: "a"* "b"', 'start: "a"* ("b" | "c")']
        )
        paths = [[a] * count for count in range(70)]
        for mask in batch_masks([repeated] * len(paths), paths):
            self.assertEqual(np.flatnonzero(mask).tolist(), [a, b])
        self.assertLessEqual(len(repeated.prefix_masks), 64)
        masks = batch_masks([repeated, repeated], [[a] * 300, []])
        for mask in masks:
            self.assertEqual(np.flatnonzero(mask).tolist(), [a, b])
        self.assertTrue(all(len(path) <= 256 for path in repeated.prefix_masks))
        masks[1][:] = 0
        self.assertEqual(
            np.flatnonzero(batch_masks([repeated], [[]])[0]).tolist(), [a, b]
        )
        repeated.accept([a, b])
        self.assertEqual(repeated.prefix_masks, {})
        self.assertTrue(repeated.accepting)
