import json

import llguidance

from ...worker.constraints import tool_schema


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
