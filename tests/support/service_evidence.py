def agent_evidence(api="chat"):
    events, history = [], []
    for index, (path, text) in enumerate(
        (
            ("crates/scheduler/src/lib.rs", "pub fn schedule() {\n  real_source\n}"),
            (
                "python/oh_my_vllm/worker/model_runner.py",
                "class OhMyVllmWorker:\n  real_source",
            ),
        )
    ):
        call = f"call_{index}"
        events.extend(
            [
                {
                    "type": "tool_execution_start",
                    "toolName": "read",
                    "toolCallId": call,
                    "args": {"path": path},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": call,
                    "isError": False,
                    "result": {"content": [{"type": "text", "text": text}]},
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": call,
                        "isError": False,
                        "content": [{"type": "text", "text": text}],
                    },
                },
            ]
        )
        history.append(
            {"role": "tool", "tool_call_id": call, "content": text}
            if api == "chat"
            else {"type": "function_call_output", "call_id": call, "output": text}
        )
    rid = "chatcmpl_test_17" if api == "chat" else "resp_test_17"
    events.append(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "api": "openai-completions" if api == "chat" else "openai-responses",
                "model": "qwen3.8-27b-fp8",
                "responseId": rid,
                "stopReason": "stop",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "See crates/scheduler/src/lib.rs and "
                            "python/oh_my_vllm/worker/model_runner.py."
                        ),
                    }
                ],
            },
        }
    )
    requests = [
        {
            "response_status": 200,
            "response_ids": [rid],
            "request": {
                "model": "qwen3.8-27b-fp8",
                "messages" if api == "chat" else "input": history,
            },
        }
    ]
    return events, requests
