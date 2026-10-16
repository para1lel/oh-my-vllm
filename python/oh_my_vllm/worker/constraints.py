import copy
import json

import llguidance
import numpy as np


def relocate_schema(value, name):
    if isinstance(value, list):
        return [relocate_schema(item, name) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: relocate_schema(item, name) for key, item in value.items()}
    if "$ref" in result and result["$ref"].startswith("#"):
        result["$ref"] = f"#/$defs/{name}" + result["$ref"][1:]
    return result


def tool_schema(tools):
    definitions, alternatives = {}, []
    for index, tool in enumerate(tools):
        name = f"tool_{index}"

        definitions[name] = relocate_schema(
            tool["function"].get("parameters", {"type": "object"}), name
        )
        alternatives.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": tool["function"]["name"]},
                    "arguments": {"$ref": f"#/$defs/{name}"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": alternatives, "$defs": definitions}


def tool_grammar(request):
    choice = request.get("tool_choice", "auto")
    tools = request["tools"]
    if isinstance(choice, dict):
        tools = [
            tool
            for tool in tools
            if tool["function"]["name"] == choice["function"]["name"]
        ]
    schema = tool_schema(tools)
    parallel = request.get("parallel_tool_calls", True)
    if choice == "auto":
        grammar = llguidance.StructTag.to_grammar(
            [
                llguidance.StructTag(
                    trigger="<tool_call>",
                    begin="<tool_call>",
                    grammar=schema,
                    end="</tool_call>",
                )
            ],
            assume_special=True,
        )
        grammar = grammar.replace('"</tool_call>"', "</tool_call>")
        return grammar if parallel else grammar.replace("(tag_0)*", "tag_0?")
    count = "+" if parallel else ""
    return (
        "%llguidance {}\nstart: call"
        + count
        + "\ncall: <tool_call> %json "
        + json.dumps(schema)
        + " </tool_call> /[ \\t\\r\\n]*/\n"
    )


class Constraint:
    def __init__(self, tokenizer, grammar=None):
        self.tokenizer = tokenizer
        self.matcher = (
            llguidance.LLMatcher(tokenizer, grammar, log_level=0)
            if grammar is not None
            else None
        )
        self.check()

    @classmethod
    def for_request(cls, tokenizer, request):
        output = request.get("format", {"type": "text"})
        if output["type"] in {"json_object", "json_schema"}:
            schema = output.get("schema", {"type": "object"})
            grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
            return cls(tokenizer, grammar)
        if request.get("tools") and request.get("tool_choice") != "none":
            return cls(tokenizer, tool_grammar(request))
        return cls(tokenizer)

    def check(self):
        if self.matcher is not None and self.matcher.is_error():
            raise ValueError(self.matcher.get_error())

    @property
    def accepting(self):
        return self.matcher is None or self.matcher.is_accepting()

    def mask(self):
        if self.matcher is None:
            return np.ones(self.tokenizer.vocab_size, dtype=np.uint8)
        self.check()
        return (
            np.frombuffer(self.matcher.compute_logit_bias(), dtype=np.uint8)
            .astype(bool)
            .astype(np.uint8)
        )

    def accept(self, tokens):
        if self.matcher is None:
            return
        for token in tokens:
            if token in self.tokenizer.eos_tokens and self.matcher.is_accepting():
                continue
            if not self.matcher.consume_token(token):
                self.check()
                raise ValueError("token violates the output constraint")

    def fork(self):
        result = copy.copy(self)
        if self.matcher is not None:
            result.matcher = self.matcher.deep_copy()
        return result
