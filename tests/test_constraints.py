import unittest
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import llguidance
import numpy as np
from oh_my_vllm.worker.constraints import Constraint
from oh_my_vllm.worker.request import Request
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


class ChainDevice:
    def __init__(self):
        self.downloads = 0
        self.module = SimpleNamespace(copy_bytes=self.copy, gather_rows=self.gather)

    def empty(self, shape, dtype):
        return np.empty(shape, np.float32 if dtype == "bfloat16" else dtype)

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
    def __init__(self, weights, rows, allocate):
        self.buffers = [None] * 9
        self.buffers[0] = allocate((rows, 5120), "bfloat16")
        self.buffers[6] = allocate((rows, 248320), "float32")
        self.buffers[7] = allocate((rows,), "int64")

    def __call__(self, value, residual, history, mask, parameters, rng, *, greedy):
        self.buffers[0][:] = value + residual + 1
        self.buffers[7][:] = self.buffers[0][:, 0].astype(np.int64)
        self.buffers[6][:] = 0
        self.buffers[6][np.arange(len(value)), self.buffers[7]] = 1
        return self.buffers[7], None, self.buffers[6]


class ChainDraft:
    def __init__(self, device, weights, batch, first):
        from oh_my_vllm.worker.sampling import Sampling

        self.device, self.first = device, first
        self.hidden = device.zeros((batch, 5120), "bfloat16")
        self.tokens = device.zeros((batch,), "int64")
        self.rows = device.upload(np.arange(batch, dtype=np.int32))
        self.ffn = SimpleNamespace(buffers=(self.hidden.copy(), self.hidden.copy()))
        self.output = self.ffn.buffers
        self.sampling = Sampling(device, None, batch)
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
        )

    def run(self):
        self.output[0][:] = self.hidden + (0 if self.first else self.tokens[:, None])
        self.output[1][:] = 0
        return self.output

    def execute(self, graph):
        self.run()

    def copy(self, destination, source):
        self.device.copy(destination, source)

    def close(self):
        self.sampling.close()


class Constraints(unittest.TestCase):
    def test_repetition_mask_requires_complete_cycles(self):
        tokenizer = SimpleNamespace(
            eos={248319},
            grammar=SimpleNamespace(vocab_size=248320),
            valid_tokens=248320,
            think_end=248318,
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
                self.assertEqual(set(np.flatnonzero(request.mask() == 0)), blocked)

    def test_sampling_inputs_do_not_retain_speculative_history(self):
        from oh_my_vllm.worker.sampling import SamplingInputs

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
        inputs.update([request, request], [[6, 6], [7]], "target")
        self.assertTrue(inputs.history_needed)
        self.assertEqual(inputs.history[:, 0, 4].tolist(), [2, 2])
        self.assertEqual(inputs.history[:, 1, 5].tolist(), [1, 1])
        self.assertEqual(inputs.history[:, 1, 6].tolist(), [2, 0])
        self.assertEqual(inputs.history[:, 1, 7].tolist(), [0, 1])
        self.assertFalse(request.counts[1, 6:8].any())
        np.testing.assert_array_equal(inputs.random, [[7, 13], [7, 13]])
        self.assertFalse(inputs.mask[:, 259:].any())

        request.options["sampling"]["presence_penalty"] = 0
        inputs.update([request, request], [[], []], "proposal")
        self.assertFalse(inputs.history_needed)
        self.assertEqual(inputs.random[0, 0], 7 ^ 0x9E3779B97F4A7C15)
        self.assertEqual(inputs.random[0, 1], 17)
        request.options["sampling"]["frequency_penalty"] = 0.25
        inputs.update([request, request], [[], []], "target")
        self.assertTrue(inputs.history_needed)
        self.assertFalse(inputs.history[:, 1, 6:8].any())
        self.assertEqual(request.history, [4, 4, 5])
        self.assertEqual((request.random_offset, request.proposal_offset), (13, 17))

    def test_sampling_mask_cache_tracks_output_and_thinking_boundaries(self):
        from oh_my_vllm.worker.sampling import SamplingInputs

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
            return set(np.flatnonzero(inputs.mask[0]))

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

    def test_greedy_draft_chain_keeps_stage_order_private_probabilities_and_fallback(
        self,
    ):
        from oh_my_vllm.worker.draft_chain import can_chain
        from oh_my_vllm.worker.speculation import Speculation

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

        def request():
            value = Request(tokenizer, [4, 4], deepcopy(options))
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
            patch("oh_my_vllm.worker.speculation.Draft", ChainDraft),
            patch("oh_my_vllm.worker.sampling.Sample", ChainSample),
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
            for update in (
                {"temperature": 0.5},
                {"frequency_penalty": 0.2},
                {"repetition_guard": True},
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
        from oh_my_vllm.worker.sampling import SamplingInputs

        request = Request(
            SimpleNamespace(
                eos={256},
                grammar=grammar_tokenizer,
                valid_tokens=259,
                think_end=257,
                generation={},
            ),
            [],
            {"max_tokens": 16, "sampling": {"repetition_guard": False}},
        )
        request.constraint = constraint
        inputs = SamplingInputs(1)
        for prefix in ([], tokenizer.encode("1}").ids, []):
            inputs.update([request], [prefix], "target")
            self.assertTrue(inputs.mask_changed)
            self.assertEqual(bool(inputs.mask[0, 256]), bool(prefix))
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
        tool = Constraint.for_request(grammar_tokenizer, tool_request)
        tool.accept(
            tokenizer.encode('<tool_call>{"name":"choose","arguments":{"x":').ids
        )
        self.assertFalse(tool.mask()[tokenizer.encode("3").ids[0]])
        branch = tool.fork()
        branch.accept(tokenizer.encode("1}}</tool_call>").ids)
        self.assertTrue(branch.accepting)
        self.assertFalse(tool.accepting)
