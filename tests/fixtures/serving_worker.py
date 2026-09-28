"""Scripted CPU worker for HTTP integration tests, never inference evidence."""

import json
import os
import sys
import time
from pathlib import Path

import msgpack
import zmq
from oh_my_vllm.worker.batch_plan import plan_request
from oh_my_vllm.worker.protocol import ScheduledRequest
from oh_my_vllm.worker.serving import RequestValidationError, ServingAdapter


def main():
    address = sys.argv[sys.argv.index("--socket") + 1]
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(address)
    socket.RCVTIMEO = 10
    requests = {}
    block_owners = {}
    adapter = None
    delayed_replies = []

    def publish_blocks():
        if not (path := os.environ.get("OH_MY_VLLM_FIXTURE_BLOCKS")):
            return
        fa = set().union(*(owner[0] for owner in block_owners.values()))
        mamba = set().union(*(owner[1] for owner in block_owners.values()))
        snapshot = {
            "owners": sorted(block_owners),
            "free_fa": 99 - len(fa),
            "free_mamba": 99 - len(mamba),
        }
        temporary = Path(path + ".tmp")
        temporary.write_text(json.dumps(snapshot))
        temporary.replace(path)

    def send_reply(reply):
        if path := os.environ.get("OH_MY_VLLM_FIXTURE_RPC_ORDER"):
            with Path(path).open("a") as output:
                output.write(
                    json.dumps({"type": reply["type"], "rpc_id": reply.get("rpc_id")})
                    + "\n"
                )
        socket.send(msgpack.packb(reply, use_bin_type=True))

    while True:
        due = [entry for entry in delayed_replies if entry[0] <= time.monotonic()]
        for entry in due:
            send_reply(entry[1])
            delayed_replies.remove(entry)
        try:
            message = msgpack.unpackb(socket.recv(), raw=False)
        except zmq.Again:
            continue
        kind = message["type"]
        if kind == "init":
            adapter = ServingAdapter(
                message["model_path"], 248320, message["max_model_len"]
            )
            reply = {
                "type": "ready",
                "logical_num_blocks": 100,
                "mamba_blocks": 100,
            }
            publish_blocks()
        elif kind == "prepare":
            rid = message["request_id"]
            request = message["request"]
            text = ""
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_CAPTURE"):
                with Path(path).open("a") as output:
                    output.write(json.dumps(request) + "\n")
            try:
                content = request["messages"][-1].get("content") or ""
                if "prepare-validation" in content:
                    raise RequestValidationError("fixture invalid request")
                if "prepare-internal" in content:
                    raise RuntimeError("fixture tokenizer failure")
                prompt, _ = adapter.prepare(rid, request)
                history = request["messages"]
                tool_results = any(item["role"] == "tool" for item in history)
                text = history[-1].get("content") or ""
                mtp_slot_plan = "mtp-slot-plan-783" in text
                if mtp_slot_plan:
                    # Force an exact page boundary in the scripted worker. Rust
                    # still owns every physical slot in the execute frame.
                    prompt = (prompt + [1] * 783)[:783]
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
                elif "slow-stream" in text:
                    answer = "hello " * 1500
                elif "long" in text or "flood-active" in text:
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
                    "hold": "hold-active" in text,
                    "slow_execute": "slow-execute" in text,
                    "flood": "flood-active" in text,
                    "slow_stream": "slow-stream" in text,
                    "stream_step": 0,
                    "mtp_slot_plan": mtp_slot_plan,
                    "accepted_history": prompt.copy(),
                    "source": None,
                    "draft_sent": False,
                }
                reply = {"type": "prepared", "prompt_token_ids": prompt}
            except Exception as exc:
                reply = {
                    "type": "error",
                    "kind": (
                        "validation"
                        if isinstance(exc, RequestValidationError)
                        else "internal"
                    ),
                    "message": str(exc),
                }
            reply["rpc_id"] = message["rpc_id"]
            if "rpc-wrong-prepare" in text and reply["type"] == "prepared":
                reply = {
                    "type": "execute_result",
                    "rpc_id": message["rpc_id"],
                    "outputs": [],
                }
            if "rpc-wrong-late" in text and reply["type"] == "prepared":
                reply = {
                    "type": "execute_result",
                    "rpc_id": message["rpc_id"],
                    "outputs": [],
                }
            if "rpc-duplicate-prepare" in text and reply["type"] == "prepared":
                send_reply(reply)
            if "rpc-buffer-cancel" in text and reply["type"] == "prepared":
                delayed_replies.append((time.monotonic() + 0.25, reply))
            if "slow-prepare" in text and (
                reply["type"] == "prepared" or "rpc-wrong-late" in text
            ):
                delayed_replies.append((time.monotonic() + 0.7, reply))
                continue
        elif kind == "execute":
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_OVERLAP"):
                scheduled = message["scheduled"]
                if len(scheduled) > 1 and any(
                    requests[r["request_id"]]["fail"] for r in scheduled
                ):
                    with Path(path).open("a") as output:
                        output.write("overlap\n")
            for rid in message["finished_request_ids"]:
                block_owners.pop(rid, None)
                requests.pop(rid, None)
                adapter.generations.pop(rid, None)
                if path := os.environ.get("OH_MY_VLLM_FIXTURE_CLEANUP"):
                    with Path(path).open("a") as output:
                        output.write(str(rid) + "\n")
            for rid in message["preempted_request_ids"]:
                block_owners.pop(rid, None)
                if requests[rid]["mtp_slot_plan"]:
                    requests[rid]["source"] = None
            outputs = []
            fatal = False
            for scheduled in message["scheduled"]:
                rid = scheduled["request_id"]
                if path := os.environ.get("OH_MY_VLLM_FIXTURE_STEPS"):
                    with Path(path).open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "rid": rid,
                                    "fa": len(set(scheduled["fa_block_table"]) - {0}),
                                    "mamba": len(
                                        set(scheduled["mamba_block_table"]) - {0}
                                    ),
                                }
                            )
                            + "\n"
                        )
                block_owners[rid] = (
                    set(scheduled["fa_block_table"]) - {0},
                    set(scheduled["mamba_block_table"]) - {0},
                )
                state = requests[rid]
                plan = None
                if state["mtp_slot_plan"]:
                    plan = plan_request(
                        ScheduledRequest(**scheduled),
                        state["accepted_history"],
                        state["source"],
                        100,
                        100,
                        4,
                    )
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
                new_drafts = []
                if plan is not None:
                    _, next_source, copies = plan.commit(int(bool(ids)))
                    state["source"] = next_source
                    state["accepted_history"].extend(ids)
                    if ids and not state["draft_sent"]:
                        new_drafts = [90, 91, 92, 93]
                        state["draft_sent"] = True
                    if path := os.environ.get("OH_MY_VLLM_FIXTURE_MTP_PLAN"):
                        with Path(path).open("a") as output:
                            output.write(
                                json.dumps(
                                    {
                                        "request_id": rid,
                                        "start": scheduled["num_computed_tokens"],
                                        "scheduled": len(scheduled["token_ids"]),
                                        "source": plan.source,
                                        "writes": plan.writes,
                                        "drafts": plan.drafts,
                                        "copies": copies,
                                        "next_source": next_source,
                                        "table": scheduled["mamba_block_table"],
                                    }
                                )
                                + "\n"
                            )
                if state["flood"]:
                    text = "x" * (1 << 20)
                    if path := os.environ.get("OH_MY_VLLM_FIXTURE_FLOOD"):
                        with Path(path).open("a") as output:
                            output.write("step\n")
                if state["slow_stream"]:
                    state["stream_step"] += 1
                    text = f"{state['stream_step']:06d}|" + "x" * 8185
                outputs.append(
                    {
                        "request_id": rid,
                        "token_ids": ids,
                        "new_draft_token_ids": new_drafts,
                        "num_accepted_draft_tokens": 0,
                        "text": text,
                        "finish_reason": adapter.generations[rid].finished,
                        "reasoning_tokens": adapter.generations[rid].reasoning_tokens,
                    }
                )
            time.sleep(
                0.15
                if any(
                    requests[r["request_id"]]["slow_execute"]
                    for r in message["scheduled"]
                )
                else (
                    0.08
                    if any(
                        requests[r["request_id"]]["hold"] for r in message["scheduled"]
                    )
                    else 0.005
                )
            )
            reply = (
                {"type": "error", "message": "fixture worker failure"}
                if fatal
                else {"type": "execute_result", "outputs": outputs}
            )
            reply["rpc_id"] = message["rpc_id"]
            publish_blocks()
        elif kind == "shutdown":
            publish_blocks()
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_SHUTDOWN"):
                Path(path).write_text("shutdown received\n")
            break
        elif kind == "abort":
            block_owners.pop(message["request_id"], None)
            requests.pop(message["request_id"], None)
            adapter.generations.pop(message["request_id"], None)
            if path := os.environ.get("OH_MY_VLLM_FIXTURE_ABORTS"):
                with Path(path).open("a") as output:
                    output.write(str(message["request_id"]) + "\n")
            publish_blocks()
            continue
        else:
            raise RuntimeError(kind)
        send_reply(reply)
    socket.close()
    context.term()


if __name__ == "__main__":
    main()
