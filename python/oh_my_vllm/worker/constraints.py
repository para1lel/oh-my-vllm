import copy
from contextlib import ExitStack, contextmanager
from functools import cache

import llguidance
import numpy as np


@cache
def mask_executor():
    return llguidance.LLExecutor(4)


def batch_masks(constraints, prefixes):
    width = constraints[0].tokenizer.vocab_size
    groups = {}
    for constraint, prefix in zip(constraints, prefixes, strict=True):
        groups.setdefault(constraint, []).append(tuple(prefix))
    pending, fallback, rows = [], {}, 0
    for constraint, paths in groups.items():
        constraint.check()
        missing = [path for path in paths if path not in constraint.prefix_masks]
        if not missing:
            continue
        longest = max(missing, key=len)
        if any(token in constraint.tokenizer.eos_tokens for token in longest) or any(
            longest[: len(path)] != path for path in missing
        ):
            for path in missing:
                branch = constraint.fork()
                branch.accept(path)
                fallback[constraint, path] = branch.mask()
            continue
        skip = next(
            index
            for index in range(len(longest) + 1)
            if longest[:index] not in constraint.prefix_masks
        )
        for index, token in enumerate(longest[:skip]):
            mask = constraint.prefix_masks[longest[:index]]
            if not (mask[token // 8] >> (token % 8)) & 1:
                raise ValueError("token violates the output constraint")
        pending.append((constraint, rows, longest, skip))
        rows += len(longest) - skip + 1
    packed = np.zeros((rows, (width + 31) // 32), dtype=np.uint32)
    # The draft API leaves rows after termination unwritten.
    for constraint, start, path, skip in pending:
        for token in constraint.tokenizer.eos_tokens:
            packed[start : start + len(path) - skip + 1, token // 32] |= 1 << (
                token % 32
            )
    with ExitStack() as states:
        for constraint, _start, path, skip in pending:
            states.enter_context(constraint.at_prefix(path[:skip]))
        for speculative in (False, True):
            entries = [
                (constraint.matcher, start, list(path[skip:]))
                if speculative
                else (constraint.matcher, start)
                for constraint, start, path, skip in pending
                if bool(path[skip:]) == speculative
            ]
            if entries:
                operation = (
                    mask_executor().unsafe_compute_mask_ptr_with_draft_token
                    if speculative
                    else mask_executor().unsafe_compute_mask_ptr
                )
                operation(entries, packed.ctypes.data, packed.shape[1] * 4, rows)
        for constraint, start, path, skip in pending:
            constraint.check()
            bits = packed[start : start + len(path) - skip + 1].view(np.uint8)
            for index, token in enumerate(path[skip:]):
                if not (bits[index, token // 8] >> (token % 8)) & 1:
                    raise ValueError("token violates the output constraint")
            for index, mask in enumerate(bits, skip):
                constraint.prefix_masks[path[:index]] = mask.tobytes()
    return [
        fallback[constraint, tuple(prefix)]
        if (constraint, tuple(prefix)) in fallback
        else np.unpackbits(
            np.frombuffer(constraint.prefix_masks[tuple(prefix)], np.uint8),
            bitorder="little",
        )[:width]
        for constraint, prefix in zip(constraints, prefixes, strict=True)
    ]


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


class Constraint:
    def __init__(self, tokenizer, grammar=None):
        self.tokenizer = tokenizer
        self.cache_identity = object()
        self.prefix_masks = {}
        self.matcher = (
            llguidance.LLMatcher(tokenizer, grammar, log_level=0)
            if grammar is not None
            else None
        )
        self.check()

    @classmethod
    def for_request(cls, tokenizer, request, *, tool_grammar=None):
        output = request.get("format", {"type": "text"})
        if output["type"] in {"json_object", "json_schema"}:
            schema = output.get("schema", {"type": "object"})
            grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
            return cls(tokenizer, grammar)
        if request.get("tools") and request.get("tool_choice") != "none":
            if tool_grammar is None:
                raise ValueError("tool constraints require a model tool grammar")
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
        self.prefix_masks = {}
        for token in tokens:
            if token in self.tokenizer.eos_tokens and self.matcher.is_accepting():
                continue
            if not self.matcher.consume_token(token):
                self.check()
                raise ValueError("token violates the output constraint")

    def fork(self):
        result = copy.copy(self)
        result.cache_identity = object()
        if self.matcher is not None:
            result.matcher = self.matcher.deep_copy()
        return result

    @contextmanager
    def at_prefix(self, tokens):
        consumed = self.matcher.try_consume_tokens(list(tokens)) if tokens else 0
        try:
            if consumed != len(tokens):
                self.check()
                raise ValueError("token violates the output constraint")
            yield
        finally:
            if consumed and not self.matcher.rollback(consumed):
                self.check()
                raise ValueError("constraint rollback failed")
