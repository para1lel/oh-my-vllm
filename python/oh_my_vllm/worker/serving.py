"""Model-specific preparation and grammar masks; no scheduling or KV ownership."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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


def validate_schema(schema: dict, *, strict: bool = False) -> None:
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    # Keep the documented supported subset even if a newer compiler accepts
    # additional keywords. Mixed string constraints must not silently weaken.
    if ("pattern" in schema or "format" in schema) and (
        "minLength" in schema or "maxLength" in schema
    ):
        raise ValueError("cannot combine pattern/format with string length bounds")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
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
        raise ValueError("unsupported schema format")
    if strict and schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError("strict objects require additionalProperties=false")
        if set(schema.get("required", [])) != set(schema.get("properties", {})):
            raise ValueError("strict objects require every property to be required")
    for key in ("properties", "$defs"):
        for child in schema.get(key, {}).values():
            validate_schema(child, strict=strict)
    if "items" in schema:
        validate_schema(schema["items"], strict=strict)
    if isinstance(schema.get("additionalProperties"), dict):
        validate_schema(schema["additionalProperties"], strict=strict)
    for child in schema.get("anyOf", []):
        validate_schema(child, strict=strict)


def validate_xml_parameters(schema: dict) -> None:
    """Reject XML encodings that cannot preserve strict JSON argument types."""

    if schema.get("type") != "object" or "$ref" in schema or "anyOf" in schema:
        raise ValueError("XML tools require a direct root object schema")
    if schema.get("additionalProperties", False) is not False:
        raise ValueError("XML tools require closed parameter objects")

    def resolve(node, seen=()):
        ref = node.get("$ref")
        if ref is None:
            return node
        if ref in seen or not ref.startswith("#/"):
            raise ValueError("unsupported XML parameter reference")
        target = schema
        for key in ref[2:].split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        return resolve(target, (*seen, ref))

    def kinds(node):
        node = resolve(node)
        if "anyOf" in node:
            if "const" in node or "enum" in node:
                raise ValueError("XML anyOf cannot have sibling const/enum")
            return set().union(*(kinds(child) for child in node["anyOf"]))
        kind = node.get("type")
        result = set(kind) if isinstance(kind, list) else {kind}
        if "string" in result:
            if any(
                key in node for key in ("pattern", "minLength", "maxLength", "format")
            ):
                raise ValueError(
                    "XML strings with pattern/length/format are unsupported; "
                    "use JSON output"
                )
            for value in [*node.get("enum", []), node.get("const", "")]:
                if isinstance(value, str) and "</parameter>" in value:
                    raise ValueError(
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
            raise ValueError(
                "XML parameters need unambiguous types; "
                "string/non-string unions are unsupported"
            )
        if types == {"string"}:
            candidates = string_candidates(node)
            seen = {}
            for value in candidates or []:
                if not isinstance(value, str):
                    raise ValueError("XML string enum/const must contain strings")
                normalized = value.strip(" \n\t")
                if normalized in seen and seen[normalized] != value:
                    raise ValueError("ambiguous XML string enum/const whitespace")
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
                    call["function"]["arguments"] = json.loads(args)
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
            raise ValueError("prompt plus output budget exceeds context limit")
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
