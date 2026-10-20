import json

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_reference",
            "description": "Find reference material about a research topic.",
            "parameters": TOOL_SCHEMA,
        },
    }
]
PROFILE = {
    "version": "serving-tools-v1",
    "sampling": "checkpoint and service defaults",
    "seed": "client ordinal",
    "tools": TOOLS,
    "tool_choice": "auto",
    "reasoning_effort": "off",
}
SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "repetition_guard",
)


def request_body(messages, output, seed):
    return {
        "model": "qwen3.8-27b-fp8",
        "messages": messages,
        "max_tokens": output,
        "min_tokens": output,
        "reasoning_effort": PROFILE["reasoning_effort"],
        "tools": TOOLS,
        "tool_choice": PROFILE["tool_choice"],
        "seed": seed,
    }


def validate_tools(response, tokenizer, *, required=False):
    choice = response["choices"][0]
    calls = choice["message"].get("tool_calls", [])
    complete = 0
    for index, call in enumerate(calls):
        function = call["function"]
        name, arguments = function["name"], function["arguments"]
        truncated = choice["finish_reason"] == "length" and index == len(calls) - 1
        if call["type"] != "function":
            raise ValueError("unexpected benchmark tool type")
        if name != "lookup_reference":
            if truncated and "lookup_reference".startswith(name) and not arguments:
                continue
            raise ValueError("unexpected benchmark tool call")
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError:
            if not truncated:
                raise ValueError("incomplete benchmark tool arguments") from None
            import llguidance

            matcher = llguidance.LLMatcher(
                tokenizer.grammar,
                llguidance.LLMatcher.grammar_from_json_schema(TOOL_SCHEMA),
            )
            tokens = tokenizer.tokenizer.encode(arguments, add_special_tokens=False).ids
            if any(not matcher.consume_token(token) for token in tokens):
                raise ValueError("invalid benchmark tool argument prefix") from None
            continue
        if (
            not isinstance(value, dict)
            or set(value) != {"topic"}
            or not isinstance(value["topic"], str)
        ):
            raise ValueError("benchmark tool arguments violate the schema")
        complete += 1
    if required and not complete:
        raise ValueError("tool check requires a complete valid call")
    return dict(complete=complete, partial=len(calls) - complete)
