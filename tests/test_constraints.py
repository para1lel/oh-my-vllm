import unittest
from types import SimpleNamespace

import llguidance
from oh_my_vllm.worker.constraints import Constraint
from oh_my_vllm.worker.request import Request
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


class Constraints(unittest.TestCase):
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
