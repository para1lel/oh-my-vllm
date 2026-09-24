"""oh_my_vllm.worker.zmq_bridge — ZMQ process entry point for the Python worker.

This module is the Python side of the Rust ↔ Python ZMQ boundary.  The Rust
scheduler forks this process, then sends msgpack-encoded messages over a DEALER
socket.  We decode each message, call OhMyVllmWorker.execute_model, encode the
result, and send it back.

Message envelope (msgpack dict):

  Rust → Python:
    {"type": "init", "model_path": str, "num_gpu_blocks": int,
     "block_size": int, "tensor_parallel_size": int, "max_model_len": int,
     "num_speculative_tokens": int}
    {"type": "register", "request_id": int, "prompt_token_ids": list[int]}
    {"type": "prepare", "request_id": int, "request": dict}
    {"type": "execute", "step_id": int, "scheduled": [
       {"request_id": int, "token_ids": list[int],
        "num_computed_tokens": int,
        "fa_block_table": list[int], "mamba_block_table": list[int],
        "prefill_token_ids": list[int] | None,
        "new_block_ids_to_zero": list[int]}],
     "finished_request_ids": list[int],
     "preempted_request_ids": list[int],
     "num_batched_tokens": int}
    {"type": "abort", "request_id": int}
    {"type": "shutdown"}

  Python → Rust:
    {"type": "ready", "logical_num_blocks": int}   # after init completes
    {"type": "prepared", "prompt_token_ids": list[int]}  # after prepare
    {"type": "execute_result",
     "outputs": [{"request_id": int, "token_ids": list[int],
                  "num_accepted_draft_tokens": int,
                  "new_draft_token_ids": list[int],
                  "text": str | None,
                  "finish_reason": str | None,
                  "reasoning_tokens": int}, ...]}
    {"type": "error", "message": str}

Run with::

    python -m oh_my_vllm.worker.zmq_bridge --socket ipc:///tmp/oh-my-vllm.ipc
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback

import msgpack
import zmq

from oh_my_vllm.worker.logging_utils import configure_logging
from oh_my_vllm.worker.model_runner import OhMyVllmWorker, RuntimeConfig
from oh_my_vllm.worker.protocol import ScheduledRequest, SchedulerOutput, WorkerOutput

logger = logging.getLogger("oh_my_vllm.worker.zmq_bridge")


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _decode_scheduled_request(d: dict) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=d["request_id"],
        token_ids=list(d["token_ids"]),
        num_computed_tokens=d["num_computed_tokens"],
        fa_block_table=list(d["fa_block_table"]),
        mamba_block_table=list(d["mamba_block_table"]),
        prefill_token_ids=d.get("prefill_token_ids"),
        new_block_ids_to_zero=d["new_block_ids_to_zero"],
    )


def _decode_scheduler_output(d: dict) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled=[_decode_scheduled_request(r) for r in d.get("scheduled", [])],
        finished_request_ids=list(d.get("finished_request_ids", [])),
        preempted_request_ids=list(d.get("preempted_request_ids", [])),
        num_batched_tokens=d.get("num_batched_tokens", 0),
    )


def _encode_worker_output(wo: WorkerOutput) -> dict:
    return {
        "type": "execute_result",
        "outputs": [
            {
                "request_id": o.request_id,
                "token_ids": o.token_ids,
                "num_accepted_draft_tokens": o.num_accepted_draft_tokens,
                "new_draft_token_ids": o.new_draft_token_ids,
                **(
                    {
                        "text": o.text,
                        "finish_reason": o.finish_reason,
                        "reasoning_tokens": o.reasoning_tokens,
                    }
                    if o.text or o.finish_reason or o.reasoning_tokens
                    else {}
                ),
            }
            for o in wo.outputs
        ],
    }


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def serve(socket_addr: str) -> None:
    configure_logging()
    logger.info("Connecting to Rust scheduler at %s", socket_addr)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.connect(socket_addr)
    # 5-second receive timeout so we can check if the parent process is still
    # alive and exit cleanly instead of blocking forever when the Rust side
    # crashes without sending a shutdown message.
    sock.RCVTIMEO = 5000
    parent_pid = os.getppid()

    worker: OhMyVllmWorker | None = None

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                if os.getppid() != parent_pid:
                    logger.info("Parent process died, exiting")
                    break
                continue
            msg = msgpack.unpackb(raw, raw=False)
            msg_type = msg.get("type")

            if msg_type == "init":
                try:
                    worker = _handle_init(msg)
                    sock.send(
                        msgpack.packb(
                            {
                                "type": "ready",
                                "logical_num_blocks": worker.logical_num_blocks,
                            },
                            use_bin_type=True,
                        )
                    )
                    logger.info("Worker ready")
                except Exception:
                    err = traceback.format_exc()
                    logger.error("Init failed:\n%s", err)
                    sock.send(
                        msgpack.packb(
                            {"type": "error", "message": err}, use_bin_type=True
                        )
                    )

            elif msg_type == "prepare":
                try:
                    if worker is None:
                        raise RuntimeError("worker not initialised")
                    if msg["request_id"] in worker.histories:
                        reply = {"type": "error", "message": "duplicate request_id"}
                    else:
                        ids = worker.prepare_request(msg["request_id"], msg["request"])
                        reply = {"type": "prepared", "prompt_token_ids": ids}
                except Exception as exc:
                    if worker is not None and msg["request_id"] in worker.histories:
                        worker.unregister_request(msg["request_id"])
                    reply = {"type": "error", "message": str(exc)}
                sock.send(msgpack.packb(reply, use_bin_type=True))

            elif msg_type == "register":
                if worker is None:
                    continue
                worker.register_request(
                    request_id=msg["request_id"],
                    prompt_token_ids=list(msg["prompt_token_ids"]),
                )

            elif msg_type == "execute":
                if worker is None:
                    sock.send(
                        msgpack.packb(
                            {"type": "error", "message": "worker not initialised"},
                            use_bin_type=True,
                        )
                    )
                    continue
                try:
                    started = time.perf_counter_ns()
                    sched_out = _decode_scheduler_output(msg)
                    decoded = time.perf_counter_ns()
                    worker_out = worker.execute_model(sched_out)
                    executed = time.perf_counter_ns()
                    sock.send(
                        msgpack.packb(
                            _encode_worker_output(worker_out), use_bin_type=True
                        )
                    )
                    logger.debug(
                        "execute_step",
                        extra={
                            "fields": {
                                "step_id": msg.get("step_id"),
                                "decode_us": (decoded - started) / 1000,
                                "execute_host_us": (executed - decoded) / 1000,
                                "encode_send_us": (time.perf_counter_ns() - executed)
                                / 1000,
                            }
                        },
                    )
                except Exception:
                    err = traceback.format_exc()
                    logger.error("execute_model failed:\n%s", err)
                    sock.send(
                        msgpack.packb(
                            {"type": "error", "message": err}, use_bin_type=True
                        )
                    )

            elif msg_type == "abort":
                if worker is not None:
                    # Registration cleanup alone leaves native runner/adapter state.
                    worker.execute_model(
                        SchedulerOutput(finished_request_ids=[msg["request_id"]])
                    )

            elif msg_type == "shutdown":
                logger.info("Shutdown received")
                if worker is not None:
                    worker.shutdown()
                break

            else:
                logger.warning("Unknown message type: %s", msg_type)

    finally:
        sock.close()
        ctx.term()


def _handle_init(msg: dict) -> OhMyVllmWorker:
    if msg.get("block_size", 784) != 784:
        raise ValueError("Target model requires block_size=784")
    if msg.get("tensor_parallel_size", 1) != 1:
        raise ValueError("Only single-device inference is implemented")
    worker = OhMyVllmWorker(
        RuntimeConfig(
            msg["model_path"],
            msg.get("max_model_len", 65536),
            msg["num_gpu_blocks"],
            msg.get("num_speculative_tokens", 0),
            msg.get("mamba_blocks"),
        )
    )
    worker.init_device()
    worker.load_model()
    worker.initialize_cache()
    return worker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="oh-my-vllm ZMQ model worker")
    parser.add_argument(
        "--socket",
        default=os.environ.get("OH_MY_VLLM_SOCKET", "ipc:///tmp/oh-my-vllm.ipc"),
        help="ZMQ DEALER socket address to connect to",
    )
    args = parser.parse_args()
    serve(args.socket)


if __name__ == "__main__":
    main()
