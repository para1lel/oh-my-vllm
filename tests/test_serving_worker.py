"""CPU regressions on the actual tokenizer and XGrammar speculative-mask path."""

import copy
import os
import threading
import time
import unittest
from unittest.mock import patch

from oh_my_vllm.worker.serving import (
    RequestValidationError,
    ServingAdapter,
    validate_schema,
    validate_xml_parameters,
)

MODEL = os.environ.get("OH_MY_VLLM_MODEL")


class ValidationTests(unittest.TestCase):
    def test_pattern_parser_distinguishes_syntax_and_internal_failure(self):
        for pattern in ("[", "("):
            with (
                self.subTest(pattern=pattern),
                self.assertRaises(RequestValidationError),
            ):
                validate_schema({"type": "string", "pattern": pattern})
        with (
            patch(
                "oh_my_vllm.worker.serving.xgr.Grammar.from_regex",
                side_effect=RuntimeError("native compiler failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "native compiler failed"),
        ):
            validate_schema({"type": "string", "pattern": "ok"})

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


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL:
            raise unittest.SkipTest(
                "set OH_MY_VLLM_MODEL for tokenizer integration tests"
            )
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

    def test_native_streaming_unicode_and_stop(self):
        from oh_my_vllm.worker.serving import Generation

        text = "北京🙂 café\n<tool_call>"
        ids = self.adapter.tokenizer.encode(text, add_special_tokens=False)
        generation = Generation(None, 1024, [], set())
        parts = [
            generation.consume([token], self.adapter.tokenizer)[1] for token in ids
        ]
        self.assertEqual("".join(parts), text)
        self.assertNotIn("\ufffd", "".join(parts))
        generation = Generation(None, 1024, ["🙂 café"], set())
        parts = [
            generation.consume([token], self.adapter.tokenizer)[1] for token in ids
        ]
        self.assertEqual("".join(parts), "北京")
        self.assertEqual(generation.finished, "stop")

    def test_background_compiler_and_tokenizer_with_live_decode(self):
        from oh_my_vllm.worker.serving import Generation

        self.prepare()
        started = threading.Event()
        errors = []
        generated = []

        def prepare_many():
            try:
                started.set()
                for value in range(16):
                    request = {
                        "messages": [{"role": "user", "content": "Return an object."}],
                        "tools": [],
                        "effort": "off",
                        "max_tokens": 16,
                        "format": {
                            "type": "json_schema",
                            "schema": {
                                "type": "object",
                                "properties": {"n": {"const": value}},
                                "required": ["n"],
                            },
                        },
                    }
                    _, _, generation = self.adapter.prepare_inputs(request)
                    generated.append((value, generation))
            except Exception as exc:
                errors.append(exc)

        background = threading.Thread(target=prepare_many, daemon=True)
        background.start()
        self.assertTrue(started.wait(1))
        decoder = Generation(None, 1024, [], set())
        token = self.adapter.tokenizer.encode("x", add_special_tokens=False)
        progressed = 0
        deadline = time.monotonic() + 10
        while background.is_alive() and time.monotonic() < deadline:
            self.adapter.masks({1: []})
            decoder.consume(token, self.adapter.tokenizer)
            progressed += 1
        background.join(0.1)
        self.assertFalse(background.is_alive(), "CPU preparation deadlocked")
        self.assertFalse(errors, errors)
        self.assertGreater(progressed, 0)
        value, generation = generated[-1]
        self.adapter.generations[9] = generation
        self.assertIsNotNone(self.adapter.masks({9: []}))
        ids = self.adapter.tokenizer.encode(
            f'{{"n":{value}}}', add_special_tokens=False
        )
        _, text = generation.consume([*ids, 248046], self.adapter.tokenizer)
        self.assertEqual(text, f'{{"n":{value}}}')

    def test_sampling_fields_and_invalid_parameters(self):
        _, params = self.prepare(
            sampling={
                "frequency_penalty": 0.5,
                "presence_penalty": -0.5,
                "repetition_penalty": 1.1,
                "seed": 123,
            }
        )
        self.assertEqual(params.frequency_penalty, 0.5)
        self.assertEqual(params.presence_penalty, -0.5)
        self.assertEqual(params.repetition_penalty, 1.1)
        self.assertEqual(params.seed, 123)
        for sampling in [
            {"temperature": -1},
            {"top_p": 0},
            {"top_k": 1.5},
            {"seed": 1.5},
            {"presence_penalty": 3},
            {"repetition_penalty": 0},
            {"repetition_penalty": 1e-39},
        ]:
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                self.prepare(sampling=sampling)

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
        drafts = {1: []}
        initial = self.adapter.masks(drafts).grammar_bitmask.copy()
        valid = self.adapter.tokenizer.encode('{"n":', add_special_tokens=False)
        drafts[1] = valid
        speculative = self.adapter.masks(drafts).grammar_bitmask
        self.assertEqual(len(speculative), len(valid) + 1)
        drafts[1] = []
        self.assertTrue((initial == self.adapter.masks(drafts).grammar_bitmask).all())
        drafts[1] = self.adapter.tokenizer.encode("invalid", add_special_tokens=False)
        self.adapter.masks(drafts)
        drafts[1] = []
        self.assertTrue((initial == self.adapter.masks(drafts).grammar_bitmask).all())
        generation = self.adapter.generations[1]
        ids = self.adapter.tokenizer.encode('{"n":123}', add_special_tokens=False)
        accepted, text = generation.consume([*ids, 248046, 123], self.adapter.tokenizer)
        self.assertEqual(accepted, [*ids, 248046])
        self.assertEqual(text, '{"n":123}')
        self.assertEqual(generation.finished, "stop")

    def test_masks_rollback_eos_draft_and_bonus(self):
        self.prepare()
        drafts = {1: []}
        initial = self.adapter.masks(drafts).grammar_bitmask.copy()
        drafts = self.adapter.tokenizer.encode('{"n":1}', add_special_tokens=False)
        mask = self.adapter.masks({1: [*drafts, 248046, 123]}).grammar_bitmask
        self.assertEqual(len(mask), len(drafts) + 3)
        self.assertTrue((mask[-2:] == -1).all())
        self.assertFalse(self.adapter.generations[1].matcher.is_terminated())
        self.assertTrue((initial == self.adapter.masks({1: []}).grammar_bitmask).all())

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
        drafts = {1: []}
        before = self.adapter.masks(drafts).grammar_bitmask.copy()
        prefix = 'Checking.</think>\n\n{"n":'
        drafts[1] = self.adapter.tokenizer.encode(prefix, add_special_tokens=False)
        self.adapter.masks(drafts)
        drafts[1] = []
        self.assertTrue((before == self.adapter.masks(drafts).grammar_bitmask).all())
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


if __name__ == "__main__":
    unittest.main()
