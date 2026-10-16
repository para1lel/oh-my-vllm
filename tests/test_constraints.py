import unittest
from types import SimpleNamespace

import llguidance
import numpy as np
from oh_my_vllm.worker.constraints import Constraint
from oh_my_vllm.worker.request import Request
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


class Constraints(unittest.TestCase):
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
            tokenizer.to_str(), n_vocab=259, eos_token=256
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
