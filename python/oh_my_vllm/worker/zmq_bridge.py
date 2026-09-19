"""oh_my_vllm.worker.zmq_bridge — ZMQ process entry point for the Python worker.

This module is the Python side of the Rust ↔ Python ZMQ boundary.  The Rust
scheduler forks this process, then sends msgpack-encoded messages over a PAIR
socket.  We decode each message, call OhMyVllmWorker.execute_model, encode the
result, and send it back.

Message envelope (msgpack dict):

  Rust → Python:
    {"type": "init", "model_path": str, "num_gpu_blocks": int,
     "block_size": int, "tensor_parallel_size": int}
    {"type": "register", "request_id": int, "prompt_token_ids": list[int],
     "max_tokens": int}
    {"type": "execute", "scheduled": [...], "finished_request_ids": [...],
     "preempted_request_ids": [...], "num_batched_tokens": int}
    {"type": "abort", "request_id": int}
    {"type": "shutdown"}

  Python → Rust:
    {"type": "ready"}                          # after init completes
    {"type": "execute_result",
     "outputs": [{"request_id": int, "next_token_id": int,
                  "num_accepted_draft_tokens": int,
                  "new_draft_token_ids": list[int]}, ...]}
    {"type": "error", "message": str}

Run with::

    python -m oh_my_vllm.worker.zmq_bridge --socket ipc:///tmp/oh-my-vllm.ipc
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback

import msgpack
import zmq

from oh_my_vllm.worker.model_runner import (
    OhMyVllmWorker,
    ScheduledRequest,
    SchedulerOutput,
    WorkerOutput,
)

logger = logging.getLogger(__name__)


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
                "next_token_id": o.next_token_id,
                "num_accepted_draft_tokens": o.num_accepted_draft_tokens,
                "new_draft_token_ids": o.new_draft_token_ids,
            }
            for o in wo.outputs
        ],
    }


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def serve(socket_addr: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [zmq_bridge] %(levelname)s %(message)s",
    )
    logger.info("Connecting to Rust scheduler at %s", socket_addr)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.connect(socket_addr)

    worker: OhMyVllmWorker | None = None

    try:
        while True:
            raw = sock.recv()
            msg = msgpack.unpackb(raw, raw=False)
            msg_type = msg.get("type")

            if msg_type == "init":
                try:
                    worker = _handle_init(msg)
                    sock.send(msgpack.packb({"type": "ready"}, use_bin_type=True))
                    logger.info("Worker ready")
                except Exception:
                    err = traceback.format_exc()
                    logger.error("Init failed:\n%s", err)
                    sock.send(
                        msgpack.packb(
                            {"type": "error", "message": err}, use_bin_type=True
                        )
                    )

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
                    sched_out = _decode_scheduler_output(msg)
                    worker_out = worker.execute_model(sched_out)
                    sock.send(
                        msgpack.packb(
                            _encode_worker_output(worker_out), use_bin_type=True
                        )
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
                    worker.unregister_request(msg["request_id"])

            elif msg_type == "shutdown":
                logger.info("Shutdown received")
                break

            else:
                logger.warning("Unknown message type: %s", msg_type)

    finally:
        sock.close()
        ctx.term()


def _handle_init(msg: dict) -> OhMyVllmWorker:
    """Build a VllmConfig and initialise the worker from an 'init' message."""
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_path: str = msg["model_path"]
    num_gpu_blocks: int = msg["num_gpu_blocks"]
    block_size: int = msg.get("block_size", 784)
    tp_size: int = msg.get("tensor_parallel_size", 1)
    max_model_len: int = msg.get("max_model_len", 65536)

    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="auto",
        seed=0,
        max_model_len=max_model_len,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.90,
        swap_space=0,
        cache_dtype="auto",
        num_gpu_blocks_override=num_gpu_blocks,
        enable_prefix_caching=True,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=tp_size,
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=32768,
        max_num_seqs=256,
        max_model_len=max_model_len,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=LoadConfig(load_format="auto"),
    )

    worker = OhMyVllmWorker(vllm_config)
    worker.init_device()
    worker.load_model()
    worker.initialize_cache(num_gpu_blocks)
    return worker


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="oh-my-vllm ZMQ model worker")
    parser.add_argument(
        "--socket",
        default=os.environ.get("OH_MY_VLLM_SOCKET", "ipc:///tmp/oh-my-vllm.ipc"),
        help="ZMQ PAIR socket address to connect to",
    )
    args = parser.parse_args()
    serve(args.socket)


if __name__ == "__main__":
    main()
