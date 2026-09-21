"""CPU regressions on the actual tokenizer and XGrammar speculative-mask path."""

import copy
import unittest
from types import SimpleNamespace

from oh_my_vllm.worker.serving import (
    ServingAdapter,
    validate_schema,
    validate_xml_parameters,
)

MODEL = "/data0/shared/Qwen3.8-27B-FP8"


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = ServingAdapter(MODEL, 248320, 65536)

    def prepare(self, **overrides):
        request = {
            "messages": [{"role": "user", "content": "Return an object."}],
            "tools": [],
            "effort": "off",
            "max_tokens": 128,
            "format": {
                "type": "json_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
        }
        request.update(overrides)
        return self.adapter.prepare(1, copy.deepcopy(request))

    def test_template_ids_and_thinking(self):
        ids, params = self.prepare()
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 10)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_k, 20)
        plain = self.adapter.tokenizer.decode(ids)
        self.assertTrue(plain.endswith("<think>\n\n</think>\n\n"))
        ids, _ = self.prepare(effort="low")
        self.assertIn("Keep your thinking brief", self.adapter.tokenizer.decode(ids))

    def test_masks_rollback_valid_and_invalid_drafts(self):
        self.prepare()
        scheduled = SimpleNamespace(
            num_scheduled_tokens={"1": 1}, scheduled_spec_decode_tokens={}
        )
        initial = self.adapter.masks(scheduled).grammar_bitmask.copy()
        valid = self.adapter.tokenizer.encode('{"n":', add_special_tokens=False)
        scheduled.scheduled_spec_decode_tokens = {"1": valid}
        speculative = self.adapter.masks(scheduled).grammar_bitmask
        self.assertEqual(len(speculative), len(valid) + 1)
        scheduled.scheduled_spec_decode_tokens = {}
        self.assertTrue(
            (initial == self.adapter.masks(scheduled).grammar_bitmask).all()
        )
        scheduled.scheduled_spec_decode_tokens = {
            "1": self.adapter.tokenizer.encode("invalid", add_special_tokens=False)
        }
        self.adapter.masks(scheduled)
        scheduled.scheduled_spec_decode_tokens = {}
        self.assertTrue(
            (initial == self.adapter.masks(scheduled).grammar_bitmask).all()
        )
        generation = self.adapter.generations[1]
        ids = self.adapter.tokenizer.encode('{"n":123}', add_special_tokens=False)
        accepted, text = generation.consume([*ids, 248046, 123], self.adapter.tokenizer)
        self.assertEqual(accepted, [*ids, 248046])
        self.assertEqual(text, '{"n":123}')
        self.assertEqual(generation.finished, "stop")

    def test_masks_rollback_eos_draft_and_bonus(self):
        self.prepare()
        scheduled = SimpleNamespace(
            num_scheduled_tokens={"1": 1}, scheduled_spec_decode_tokens={}
        )
        initial = self.adapter.masks(scheduled).grammar_bitmask.copy()
        drafts = self.adapter.tokenizer.encode('{"n":1}', add_special_tokens=False)
        scheduled.scheduled_spec_decode_tokens = {"1": [*drafts, 248046, 123]}
        mask = self.adapter.masks(scheduled).grammar_bitmask
        self.assertEqual(len(mask), len(drafts) + 3)
        self.assertTrue((mask[-2:] == -1).all())
        self.assertFalse(self.adapter.generations[1].matcher.is_terminated())
        scheduled.scheduled_spec_decode_tokens = {}
        self.assertTrue(
            (initial == self.adapter.masks(scheduled).grammar_bitmask).all()
        )

    def test_stop_and_utf8_across_steps(self):
        self.prepare(format={"type": "text"}, stop=["END"])
        generation = self.adapter.generations[1]
        ids = self.adapter.tokenizer.encode("你好ENDignored", add_special_tokens=False)
        text = ""
        for token in ids:
            _, part = generation.consume([token], self.adapter.tokenizer)
            text += part
        self.assertEqual(text, "你好")
        self.assertEqual(generation.finished, "stop")

    def test_speculative_window_crosses_reasoning_boundary(self):
        self.prepare(effort="medium")
        scheduled = SimpleNamespace(
            num_scheduled_tokens={"1": 1}, scheduled_spec_decode_tokens={}
        )
        before = self.adapter.masks(scheduled).grammar_bitmask.copy()
        prefix = 'Checking.</think>\n\n{"n":'
        scheduled.scheduled_spec_decode_tokens = {
            "1": self.adapter.tokenizer.encode(prefix, add_special_tokens=False)
        }
        self.adapter.masks(scheduled)
        scheduled.scheduled_spec_decode_tokens = {}
        self.assertTrue((before == self.adapter.masks(scheduled).grammar_bitmask).all())
        generation = self.adapter.generations[1]
        ids = self.adapter.tokenizer.encode(
            'Checking.</think>\n\n{"n":1}', add_special_tokens=False
        )
        _, text = generation.consume([*ids, 248046], self.adapter.tokenizer)
        self.assertEqual(text, 'Checking.</think>\n\n{"n":1}')
        self.assertGreater(generation.reasoning_tokens, 0)
        self.assertLess(generation.reasoning_tokens, len(ids))

    def test_required_tools_and_parallel_limit(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        call = (
            "<tool_call>\n<function=read>\n"
            "<parameter=path>README.md</parameter>\n"
            "</function>\n</tool_call>"
        )
        for parallel in [False, True]:
            self.prepare(
                format={"type": "text"},
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=parallel,
            )
            matcher = self.adapter.generations[1].matcher
            self.assertFalse(matcher.accept_token(248046))
            self.assertTrue(matcher.accept_string(call))
            self.assertEqual(matcher.accept_string("\n" + call), parallel)

    def test_xml_ambiguities_are_explicit_errors(self):
        schemas = [
            {"$ref": "#/$defs/Args", "$defs": {}},
            {"type": "object", "additionalProperties": True},
            {"type": "object", "properties": {"s": {"type": ["string", "null"]}}},
            {
                "type": "object",
                "properties": {"s": {"type": "string", "enum": ["abc", " abc"]}},
            },
        ]
        for schema in schemas:
            with self.assertRaises(ValueError):
                validate_xml_parameters(schema)
        with self.assertRaises(ValueError):
            validate_xml_parameters(
                {
                    "type": "object",
                    "properties": {"s": {"anyOf": [{"type": "string"}], "const": "a"}},
                }
            )
        with self.assertRaises(ValueError):
            validate_schema({"type": "string", "format": "bogus"})
        with self.assertRaises(ValueError):
            validate_schema({"type": "string", "pattern": "a+", "maxLength": 3})

    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_schema({"type": "object", "unevaluatedProperties": False})
        with self.assertRaises(ValueError):
            validate_schema(
                {"type": "object", "properties": {"x": {"type": "string"}}}, strict=True
            )


if __name__ == "__main__":
    unittest.main()
