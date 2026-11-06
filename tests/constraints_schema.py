import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llguidance
import numpy as np
from oh_my_vllm.models.qwen38.request import Request
from oh_my_vllm.models.qwen38.tool_grammar import tool_grammar
from oh_my_vllm.worker.constraints import Constraint, batch_masks
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tests.constraints_fixtures import sampling_mask


class ConstraintsSchema(unittest.TestCase):
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
        tokenizer, grammar_tokenizer, constraint, request, SamplingInputs = (
            self._schema_fixture()
        )
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
        self._check_schema_prefix_masks(constraint, request, SamplingInputs, tokenizer)
        self._check_schema_and_tool_branches(constraint, grammar_tokenizer, tokenizer)

        single, combined, a, b, c, d, before, saved_cache = self._intersection_fixture(
            grammar_tokenizer, tokenizer
        )
        self._check_intersection_prefix_state(combined, a, b, c, before, saved_cache)

        self._check_intersection_batch_masks(single, combined, a, b, c, d, before)
        self._check_intersection_branches_and_unicode(
            grammar_tokenizer, tokenizer, a, b
        )

        self._check_intersection_prefix_cache(grammar_tokenizer, a, b)

    def _check_intersection_prefix_cache(self, grammar_tokenizer, a, b):
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

    def _check_intersection_branches_and_unicode(
        self, grammar_tokenizer, tokenizer, a, b
    ):
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

    def _check_intersection_batch_masks(self, single, combined, a, b, c, d, before):
        # The second grammar removes c and d; native batching must not use only one.
        prefixes = [[], [a], [a, b], [a, b, 256, 256]]
        for _ in range(2):
            constraints = [single, combined] * len(prefixes)
            paths = [path for path in prefixes for _ in range(2)]
            masks = batch_masks(constraints, paths)
            packed = batch_masks(constraints, paths, packed=True)
            for dense, bits in zip(masks, packed, strict=True):
                np.testing.assert_array_equal(
                    np.unpackbits(bits, bitorder="little")[: dense.size], dense
                )
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

    def _check_intersection_prefix_state(self, combined, a, b, c, before, saved_cache):
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

    def _intersection_fixture(self, grammar_tokenizer, tokenizer):
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
        return single, combined, a, b, c, d, before, saved_cache

    def _check_schema_and_tool_branches(self, constraint, grammar_tokenizer, tokenizer):
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

    def _check_schema_prefix_masks(
        self, constraint, request, SamplingInputs, tokenizer
    ):
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

    def _schema_fixture(self):
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
        return tokenizer, grammar_tokenizer, constraint, request, SamplingInputs
