"""Model-specific preparation and grammar masks; no scheduling or KV ownership."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import xgrammar as xgr
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)

from oh_my_vllm.worker.sampling import GrammarOutput, SamplingParams

# Deliberately bounded subset: unknown keywords must never silently weaken strictness.
_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "$defs",
    "$ref",
    "description",
    "title",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "format",
}


class RequestValidationError(ValueError):
    """A rejected client request, distinct from tokenizer or compiler failures."""


def _resolve_local_schema_ref(root: dict, ref: str) -> dict:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise RequestValidationError("only local schema references are supported")
    target = root
    for encoded in unquote(ref[2:]).split("/"):
        if any(
            encoded[index] == "~"
            and (index + 1 == len(encoded) or encoded[index + 1] not in "01")
            for index in range(len(encoded))
        ):
            raise RequestValidationError("invalid schema reference escape")
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict) and key in target:
            target = target[key]
        elif isinstance(target, list) and key.isascii() and key.isdecimal():
            if (len(key) > 1 and key.startswith("0")) or len(key) > len(
                str(len(target))
            ):
                raise RequestValidationError("invalid schema reference array index")
            index = int(key)
            if index >= len(target):
                raise RequestValidationError("unresolved schema reference")
            target = target[index]
        else:
            raise RequestValidationError("unresolved schema reference")
    if not isinstance(target, dict):
        raise RequestValidationError("schema reference must resolve to an object")
    return target


def validate_schema(
    schema: dict, *, strict: bool = False, root: dict | None = None
) -> None:
    if not isinstance(schema, dict):
        raise RequestValidationError("schema must be an object")
    if root is None:
        root = schema
    for key in ("properties", "$defs"):
        if key in schema and not isinstance(schema[key], dict):
            raise RequestValidationError(f"schema {key} must be an object")
    if "required" in schema and (
        not isinstance(schema["required"], list)
        or any(not isinstance(name, str) for name in schema["required"])
    ):
        raise RequestValidationError("schema required must be an array of strings")
    if "anyOf" in schema and not isinstance(schema["anyOf"], list):
        raise RequestValidationError("schema anyOf must be an array")
    if "$ref" in schema and not isinstance(schema["$ref"], str):
        raise RequestValidationError("schema $ref must be a string")
    if "$ref" in schema:
        _resolve_local_schema_ref(root, schema["$ref"])
    if "format" in schema and not isinstance(schema["format"], str):
        raise RequestValidationError("schema format must be a string")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise RequestValidationError("schema enum must be an array")
    kinds = {"null", "boolean", "integer", "number", "string", "array", "object"}
    if "type" in schema:
        value = schema["type"]
        names = value if isinstance(value, list) else [value]
        if not names or any(
            not isinstance(name, str) or name not in kinds for name in names
        ):
            raise RequestValidationError("unsupported schema type")
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise RequestValidationError(f"schema {key} must be a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and (
            isinstance(schema[key], bool) or not isinstance(schema[key], (int, float))
        ):
            raise RequestValidationError(f"schema {key} must be numeric")
    if "pattern" in schema and not isinstance(schema["pattern"], str):
        raise RequestValidationError("schema pattern must be a string")
    if "pattern" in schema:
        try:
            xgr.Grammar.from_regex(schema["pattern"])
        except RuntimeError as exc:
            # XGrammar has no dedicated parse exception. Its parser tags syntax
            # diagnostics; other native failures remain internal errors.
            if "Regex parsing error" not in str(exc):
                raise
            raise RequestValidationError("invalid schema pattern") from exc
    # Keep the documented supported subset even if a newer compiler accepts
    # additional keywords. Mixed string constraints must not silently weaken.
    if ("pattern" in schema or "format" in schema) and (
        "minLength" in schema or "maxLength" in schema
    ):
        raise RequestValidationError(
            "cannot combine pattern/format with string length bounds"
        )
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise RequestValidationError(f"unsupported schema keywords: {sorted(unknown)}")
    if "format" in schema and schema["format"] not in {
        "date",
        "time",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
    }:
        raise RequestValidationError("unsupported schema format")
    if strict and schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise RequestValidationError(
                "strict objects require additionalProperties=false"
            )
        if set(schema.get("required", [])) != set(schema.get("properties", {})):
            raise RequestValidationError(
                "strict objects require every property to be required"
            )
    for key in ("properties", "$defs"):
        for child in schema.get(key, {}).values():
            validate_schema(child, strict=strict, root=root)
    if "items" in schema:
        validate_schema(schema["items"], strict=strict, root=root)
    if "additionalProperties" in schema and not isinstance(
        schema["additionalProperties"], (bool, dict)
    ):
        raise RequestValidationError(
            "schema additionalProperties must be boolean or object"
        )
    if isinstance(schema.get("additionalProperties"), dict):
        validate_schema(schema["additionalProperties"], strict=strict, root=root)
    for child in schema.get("anyOf", []):
        validate_schema(child, strict=strict, root=root)


def validate_xml_parameters(schema: dict) -> None:
    """Reject XML encodings that cannot preserve strict JSON argument types."""

    if schema.get("type") != "object" or "$ref" in schema or "anyOf" in schema:
        raise RequestValidationError("XML tools require a direct root object schema")
    if schema.get("additionalProperties", False) is not False:
        raise RequestValidationError("XML tools require closed parameter objects")

    def resolve(node, seen=()):
        ref = node.get("$ref")
        if ref is None:
            return node
        if ref in seen:
            raise RequestValidationError("unsupported XML parameter reference")
        target = _resolve_local_schema_ref(schema, ref)
        return resolve(target, (*seen, ref))

    def kinds(node):
        node = resolve(node)
        if "anyOf" in node:
            if "const" in node or "enum" in node:
                raise RequestValidationError("XML anyOf cannot have sibling const/enum")
            return set().union(*(kinds(child) for child in node["anyOf"]))
        kind = node.get("type")
        result = set(kind) if isinstance(kind, list) else {kind}
        if "string" in result:
            if any(
                key in node for key in ("pattern", "minLength", "maxLength", "format")
            ):
                raise RequestValidationError(
                    "XML strings with pattern/length/format are unsupported; "
                    "use JSON output"
                )
            for value in [*node.get("enum", []), node.get("const", "")]:
                if isinstance(value, str) and "</parameter>" in value:
                    raise RequestValidationError(
                        "XML string enum/const cannot contain </parameter>"
                    )
        return result

    def string_candidates(node):
        node = resolve(node)
        if "anyOf" in node:
            groups = [string_candidates(child) for child in node["anyOf"]]
            if any(group is None for group in groups):
                return None
            return [value for group in groups for value in group]
        if "const" in node:
            return [node["const"]]
        return node.get("enum")

    for node in schema.get("properties", {}).values():
        types = kinds(node)
        if None in types or ("string" in types and len(types) > 1):
            raise RequestValidationError(
                "XML parameters need unambiguous types; "
                "string/non-string unions are unsupported"
            )
        if types == {"string"}:
            candidates = string_candidates(node)
            seen = {}
            for value in candidates or []:
                if not isinstance(value, str):
                    raise RequestValidationError(
                        "XML string enum/const must contain strings"
                    )
                normalized = value.strip(" \n\t")
                if normalized in seen and seen[normalized] != value:
                    raise RequestValidationError(
                        "ambiguous XML string enum/const whitespace"
                    )
                seen[normalized] = value


@dataclass
class Generation:
    matcher: object
    max_tokens: int
    stop: list[str]
    eos: set[int]
    reasoning_end: int | None = None
    reasoning_tokens: int = 0
    ids: list[int] = field(default_factory=list)
    decoder: DecodeStream = field(default_factory=DecodeStream)
    pending: str = ""
    finished: str | None = None

    def consume(self, tokens: list[int], tokenizer) -> tuple[list[int], str]:
        accepted, text = [], ""
        for token in tokens:
            if self.finished:
                break
            if self.matcher is not None and not self.matcher.accept_token(token):
                raise RuntimeError("sample violated the grammar mask")
            accepted.append(token)
            self.ids.append(token)
            if token in self.eos:
                self.finished = "stop"
                break
            if self.reasoning_end is not None:
                self.reasoning_tokens += 1
                if token == self.reasoning_end:
                    self.reasoning_end = None
            delta = self.decoder.step(tokenizer.backend_tokenizer, token) or ""
            self.pending += delta
            matches = [
                self.pending.find(stop) for stop in self.stop if stop in self.pending
            ]
            if matches:
                text += self.pending[: min(matches)]
                self.pending = ""
                self.finished = "stop"
                break
            # Hold only an actual partial stop prefix, not the entire stop length.
            hold = max(
                (
                    n
                    for stop in self.stop
                    for n in range(1, len(stop))
                    if self.pending.endswith(stop[:n])
                ),
                default=0,
            )
            end = len(self.pending) - hold
            text += self.pending[:end]
            self.pending = self.pending[end:]
            if len(self.ids) >= self.max_tokens:
                self.finished = "length"
        if self.finished:
            text += self.pending
            self.pending = ""
        return accepted, text


class ServingAdapter:
    def __init__(self, model_path: str, vocab_size: int, max_model_len: int):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        # DecodeStream uses the Rust tokenizer directly, avoiding repeated
        # Python vocabulary enumeration and retaining partial UTF-8 bytes.
        if not self.tokenizer.is_fast:
            raise ValueError("serving requires a fast tokenizer for streaming decode")
        self.defaults = json.loads(
            (Path(model_path) / "generation_config.json").read_text()
        )
        self.eos = set(self.defaults["eos_token_id"])
        info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=vocab_size, stop_token_ids=list(self.eos)
        )
        self.compiler = xgr.GrammarCompiler(
            info, max_threads=2, cache_limit_bytes=256 << 20
        )
        self.vocab_size = vocab_size
        self.max_model_len = max_model_len
        self.generations: dict[int, Generation] = {}
        self._mask = None

    def prepare(self, request_id: int, request: dict):
        thinking = request["effort"] != "off"
        tools = request.get("tools", [])
        choice = request.get("tool_choice", "auto" if tools else "none")
        fmt = request.get("format", {"type": "text"})
        for tool in tools:
            function = tool["function"]
            function.setdefault(
                "parameters",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )
            validate_schema(
                function.get("parameters", {"type": "object"}),
                strict=function.get("strict", False),
            )
            validate_xml_parameters(function["parameters"])
            function["parameters"].setdefault("additionalProperties", False)
        grammar = None
        if fmt["type"] != "text":
            schema = fmt.get("schema", {"type": "object"})
            validate_schema(schema, strict=fmt.get("strict", False))
            suffix = JSONSchemaFormat(json_schema=schema)
            if thinking:
                suffix = SequenceFormat(
                    elements=[
                        TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                        ConstStringFormat(value="\n\n"),
                        suffix,
                    ]
                )
            grammar = self.compiler.compile_structural_tag(StructuralTag(format=suffix))
        elif tools:
            tag = xgr.get_model_structural_tag(
                "qwen_3_5",
                tools=tools,
                tool_choice=choice,
                reasoning=thinking,
            )
            suffix = tag.format.elements[-1] if thinking else tag.format
            if not request.get("parallel_tool_calls", True) and hasattr(
                suffix, "stop_after_first"
            ):
                suffix.stop_after_first = True
            grammar = self.compiler.compile_structural_tag(tag)
        # Template accepts dict arguments, while the API transmits JSON strings.
        messages = request["messages"]
        for message in messages:
            for call in message.get("tool_calls", []):
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        call["function"]["arguments"] = json.loads(args)
                    except json.JSONDecodeError as exc:
                        raise RequestValidationError(
                            "tool call arguments must contain valid JSON"
                        ) from exc
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools if choice != "none" else None,
            tokenize=False,
            add_generation_prompt=True,
            preserve_thinking=request.get("preserve_thinking", True),
            enable_thinking=thinking,
            reasoning_effort=request["effort"],
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_tokens = request["max_tokens"]
        if not ids or len(ids) + max_tokens > self.max_model_len:
            raise RequestValidationError(
                "prompt plus output budget exceeds context limit"
            )
        sampling = {
            key: self.defaults[key]
            for key in ("temperature", "top_p", "top_k")
            if key in self.defaults
        }
        sampling.update(request.get("sampling", {}))
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=False, **sampling)
        matcher = xgr.GrammarMatcher(grammar) if grammar else None
        self.generations[request_id] = Generation(
            matcher,
            max_tokens,
            request.get("stop", []),
            self.eos,
            self.tokenizer.convert_tokens_to_ids("</think>") if thinking else None,
        )
        return ids, params

    def masks(self, scheduled):
        ids = [
            rid
            for rid in scheduled.num_scheduled_tokens
            if int(rid) in self.generations
            and self.generations[int(rid)].matcher is not None
        ]
        if not ids:
            return None
        rows = sum(
            1 + len(scheduled.scheduled_spec_decode_tokens.get(rid, [])) for rid in ids
        )
        if self._mask is None or self._mask.shape[0] < rows:
            self._mask = xgr.allocate_token_bitmask(rows, self.vocab_size)
        mask = self._mask[:rows]
        row = 0
        for rid in ids:
            matcher = self.generations[int(rid)].matcher
            advanced = 0
            valid = True
            # Masks follow each speculative prefix. An invalid draft is rejected at
            # its first bad position; subsequent rows cannot affect accepted output.
            try:
                for token in [
                    *scheduled.scheduled_spec_decode_tokens.get(rid, []),
                    None,
                ]:
                    valid = valid and not matcher.is_terminated()
                    if valid:
                        matcher.fill_next_token_bitmask(mask, row)
                    else:
                        # Rows after EOS cannot contribute: consume stops at EOS.
                        mask[row].fill_(-1)
                    row += 1
                    if token is not None and valid:
                        valid = matcher.accept_token(token)
                        advanced += int(valid)
            finally:
                if advanced:
                    matcher.rollback(advanced)
        return GrammarOutput(ids, mask.numpy())
