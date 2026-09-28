"""Scripted CPU worker for HTTP integration tests, never inference evidence."""

import json
import os
import sys
import time
from pathlib import Path

import msgpack
import zmq
from oh_my_vllm.worker.serving import ServingAdapter


def main():
    address = sys.argv[sys.argv.index("--socket") + 1]
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(address)
    requests = {}
    adapter = None
    while True:
        message = msgpack.unpackb(socket.recv(), raw=False)
        kind = message["type"]
        if kind == "init":
            adapter = ServingAdapter(
                message["model_path"], 248320, message["max_model_len"]
            )
            reply = {"type": "ready", "logical_num_blocks": 100}
        elif kind == "prepare":
            rid = message["request_id"]
            request = message["request"]
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_CAPTURE"):
                with Path(path).open("a") as output:
                    output.write(json.dumps(request) + "\n")
            try:
                prompt, _ = adapter.prepare(rid, request)
                history = request["messages"]
                tool_results = any(item["role"] == "tool" for item in history)
                text = history[-1].get("content") or ""
                if request["format"]["type"] != "text":
                    answer = '{"n":123}'
                elif (
                    request["tools"]
                    and request["tool_choice"] != "none"
                    and not tool_results
                ):
                    name = request["tools"][0]["function"]["name"]
                    intent = (
                        "<parameter=i>Read project documentation</parameter>\n"
                        if "i"
                        in request["tools"][0]["function"]["parameters"]["properties"]
                        else ""
                    )
                    answer = (
                        f"<tool_call>\n<function={name}>\n{intent}"
                        "<parameter=path>\nREADME.md\n</parameter>\n"
                        "</function>\n</tool_call>"
                    )
                elif "long" in text:
                    answer = "hello " * 1000
                else:
                    answer = "Fixture response."
                if request["effort"] != "off":
                    answer = "Checking.</think>\n\n" + answer
                tokens = [
                    *adapter.tokenizer.encode(answer, add_special_tokens=False),
                    248046,
                ]
                requests[rid] = {
                    "tokens": tokens,
                    "prompt_len": len(prompt),
                    "fail": "worker-fail" in text,
                    "fatal": "worker-fatal" in text,
                }
                reply = {"type": "prepared", "prompt_token_ids": prompt}
                if "slow-prepare" in text:
                    time.sleep(0.7)
            except Exception as exc:
                reply = {"type": "error", "message": str(exc)}
            reply["rpc_id"] = message["rpc_id"]
        elif kind == "execute":
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_OVERLAP"):
                scheduled = message["scheduled"]
                if len(scheduled) > 1 and any(
                    requests[r["request_id"]]["fail"] for r in scheduled
                ):
                    with Path(path).open("a") as output:
                        output.write("overlap\n")
            for rid in message["finished_request_ids"]:
                requests.pop(rid, None)
                adapter.generations.pop(rid, None)
                if path := os.environ.get("OH_MY_VLLM_FIXTURE_CLEANUP"):
                    with Path(path).open("a") as output:
                        output.write(str(rid) + "\n")
            outputs = []
            fatal = False
            for scheduled in message["scheduled"]:
                rid = scheduled["request_id"]
                state = requests[rid]
                if state["fatal"]:
                    fatal = True
                    break
                if state["fail"]:
                    outputs.append(
                        {
                            "request_id": rid,
                            "token_ids": [],
                            "error": "fixture request failure",
                        }
                    )
                    continue
                ids = []
                if (
                    scheduled["num_computed_tokens"] + len(scheduled["token_ids"])
                    >= state["prompt_len"]
                ):
                    ids = [state["tokens"].pop(0)]
                ids, text = adapter.generations[rid].consume(ids, adapter.tokenizer)
                outputs.append(
                    {
                        "request_id": rid,
                        "token_ids": ids,
                        "new_draft_token_ids": [],
                        "num_accepted_draft_tokens": 0,
                        "text": text,
                        "finish_reason": adapter.generations[rid].finished,
                        "reasoning_tokens": adapter.generations[rid].reasoning_tokens,
                    }
                )
            time.sleep(0.005)
            reply = (
                {"type": "error", "message": "fixture worker failure"}
                if fatal
                else {"type": "execute_result", "outputs": outputs}
            )
            reply["rpc_id"] = message["rpc_id"]
        elif kind == "shutdown":
            break
        elif kind == "abort":
            requests.pop(message["request_id"], None)
            adapter.generations.pop(message["request_id"], None)
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_ABORTS"):
                with Path(path).open("a") as output:
                    output.write(str(message["request_id"]) + "\n")
            continue
        else:
            raise RuntimeError(kind)
        socket.send(msgpack.packb(reply, use_bin_type=True))
    socket.close()
    context.term()


if __name__ == "__main__":
    main()
