"""oh_my_vllm.worker.zmq_bridge — ZMQ process entry point for the Python worker.

This module is the Python side of the Rust ↔ Python ZMQ boundary.  The Rust
scheduler forks this process, then sends msgpack-encoded messages over a DEALER
socket.  We decode each message, call OhMyVllmWorker.execute_model, encode the
result, and send it back.

Message envelope (msgpack dict):

  Rust → Python:
    {"type": "init", "model_path": str, "num_gpu_blocks": int,
     "mamba_blocks": int | None,
     "block_size": int, "tensor_parallel_size": int, "max_model_len": int,
     "num_speculative_tokens": int}
    {"type": "register", "request_id": int, "prompt_token_ids": list[int]}
    {"type": "prepare", "rpc_id": int, "request_id": int, "request": dict}
    {"type": "execute", "rpc_id": int, "step_id": int, "scheduled": [
       {"request_id": int, "token_ids": list[int],
        "num_computed_tokens": int,
        "fa_block_table": list[int], "mamba_block_table": list[int],
        "prefill_token_ids": list[int] | None}],
     "finished_request_ids": list[int],
     "preempted_request_ids": list[int],
     "num_batched_tokens": int}
    {"type": "abort", "request_id": int}
    {"type": "shutdown"}

  Python → Rust:
    {"type": "ready", "logical_num_blocks": int, "mamba_blocks": int}
    {"type": "prepared", "rpc_id": int, "prompt_token_ids": list[int]}
    {"type": "execute_result", "rpc_id": int,
     "outputs": [{"request_id": int, "token_ids": list[int],
                  "num_accepted_draft_tokens": int,
                  "new_draft_token_ids": list[int],
                  "text": str | None,
                  "finish_reason": str | None,
                  "reasoning_tokens": int}, ...]}
    {"type": "error", "rpc_id": int, "kind": "validation" | "internal",
     "message": str}

Run with::

    python -m oh_my_vllm.worker.zmq_bridge --socket ipc:///tmp/oh-my-vllm.ipc
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass

import msgpack
import zmq

from oh_my_vllm.worker.logging_utils import configure_logging
from oh_my_vllm.worker.model_runner import OhMyVllmWorker, RuntimeConfig
from oh_my_vllm.worker.protocol import ScheduledRequest, SchedulerOutput, WorkerOutput
from oh_my_vllm.worker.serving import RequestValidationError

logger = logging.getLogger("oh_my_vllm.worker.zmq_bridge")


@dataclass
class _PrepareJob:
    rpc_id: int
    request_id: int
    request: dict


class _PrepareExecutor:
    """One daemon CPU preparer; only the bridge thread owns live worker state."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.condition = threading.Condition()
        self.queued: _PrepareJob | None = None
        self.running_id: int | None = None
        self.stopping = False
        self.completed = queue.SimpleQueue()
        self.read_fd, self.write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self.thread = threading.Thread(
            target=self._run, name="oh-my-vllm-prepare", daemon=True
        )
        self.thread.start()

    def submit(self, job: _PrepareJob) -> bool:
        with self.condition:
            if self.stopping or self.queued is not None:
                return False
            self.queued = job
            self.condition.notify()
            return True

    def cancel_queued(self, request_id: int) -> bool:
        with self.condition:
            if self.queued is None or self.queued.request_id != request_id:
                return False
            self.queued = None
            return True

    def _run(self):
        while True:
            with self.condition:
                while self.queued is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                job = self.queued
                self.queued = None
                self.running_id = job.request_id
            try:
                result = self.adapter.prepare_inputs(job.request)
                reply = (job, result, None)
            except Exception as exc:
                reply = (job, None, exc)
            with self.condition:
                self.running_id = None
            self.completed.put(reply)
            # A full pipe remains readable; shutdown may have closed it.
            with suppress(BlockingIOError, OSError):
                os.write(self.write_fd, b"x")

    def close(self):
        with self.condition:
            self.stopping = True
            self.queued = None
            self.condition.notify()
        os.close(self.read_fd)
        os.close(self.write_fd)


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
                "error": o.error,
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
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    parent_pid = os.getppid()

    worker: OhMyVllmWorker | None = None
    preparer: _PrepareExecutor | None = None
    preparing: set[int] = set()
    cancelled: set[int] = set()

    def send(reply: dict) -> None:
        sock.send(msgpack.packb(reply, use_bin_type=True))

    def drain_prepared() -> None:
        if preparer is None:
            return
        while True:
            try:
                os.read(preparer.read_fd, 4096)
            except BlockingIOError:
                break
        while True:
            try:
                job, result, error = preparer.completed.get_nowait()
            except queue.Empty:
                break
            request_id = job.request_id
            preparing.discard(request_id)
            if request_id in cancelled:
                cancelled.discard(request_id)
                continue
            if error is not None:
                if not isinstance(error, RequestValidationError):
                    logger.error("prepare failed for request %s: %s", request_id, error)
                send(
                    {
                        "type": "error",
                        "rpc_id": job.rpc_id,
                        "kind": (
                            "validation"
                            if isinstance(error, RequestValidationError)
                            else "internal"
                        ),
                        "message": str(error),
                    }
                )
                continue
            registered = False
            try:
                ids, params, generation = result
                worker.register_request(request_id, ids, params)
                registered = True
                worker.serving.generations[request_id] = generation
                send(
                    {
                        "type": "prepared",
                        "rpc_id": job.rpc_id,
                        "prompt_token_ids": ids,
                    }
                )
            except Exception as exc:
                if registered:
                    worker.unregister_request(request_id)
                logger.exception(
                    "prepare registration failed for request %s", request_id
                )
                send(
                    {
                        "type": "error",
                        "rpc_id": job.rpc_id,
                        "kind": "internal",
                        "message": str(exc),
                    }
                )

    try:
        while True:
            events = dict(poller.poll(5000))
            if preparer is not None and preparer.read_fd in events:
                drain_prepared()
            if sock not in events:
                if os.getppid() != parent_pid:
                    logger.info("Parent process died, exiting")
                    break
                continue
            raw = sock.recv()
            msg = msgpack.unpackb(raw, raw=False)
            msg_type = msg.get("type")

            if msg_type == "init":
                try:
                    worker = _handle_init(msg)
                    fa_blocks, mamba_blocks = worker.cache_capacities()
                    sock.send(
                        msgpack.packb(
                            {
                                "type": "ready",
                                "logical_num_blocks": fa_blocks,
                                "mamba_blocks": mamba_blocks,
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
                request_id = msg["request_id"]
                try:
                    if worker is None:
                        raise RuntimeError("worker not initialised")
                    if request_id in worker.histories or request_id in preparing:
                        raise RuntimeError("duplicate request_id")
                    if preparer is None:
                        if worker.serving is None:
                            from oh_my_vllm.worker.serving import ServingAdapter

                            worker.serving = ServingAdapter(
                                worker.config.model, 248320, worker.config.max_model_len
                            )
                        preparer = _PrepareExecutor(worker.serving)
                        poller.register(preparer.read_fd, zmq.POLLIN)
                    if not preparer.submit(
                        _PrepareJob(msg["rpc_id"], request_id, msg["request"])
                    ):
                        raise RuntimeError("prepare queue is full")
                    preparing.add(request_id)
                except Exception as exc:
                    logger.exception(
                        "prepare dispatch failed for request %s", request_id
                    )
                    send(
                        {
                            "type": "error",
                            "rpc_id": msg["rpc_id"],
                            "kind": "internal",
                            "message": str(exc),
                        }
                    )

            elif msg_type == "register":
                if worker is None:
                    continue
                if msg["request_id"] in preparing:
                    logger.warning(
                        "register rejected during prepare for request %s",
                        msg["request_id"],
                    )
                    continue
                try:
                    worker.register_request(
                        request_id=msg["request_id"],
                        prompt_token_ids=list(msg["prompt_token_ids"]),
                    )
                except Exception:
                    logger.exception(
                        "register failed for request %s", msg["request_id"]
                    )

            elif msg_type == "execute":
                if worker is None:
                    sock.send(
                        msgpack.packb(
                            {
                                "type": "error",
                                "message": "worker not initialised",
                                "rpc_id": msg["rpc_id"],
                            },
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
                    reply = _encode_worker_output(worker_out)
                    reply["rpc_id"] = msg["rpc_id"]
                    sock.send(msgpack.packb(reply, use_bin_type=True))
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
                            {"type": "error", "message": err, "rpc_id": msg["rpc_id"]},
                            use_bin_type=True,
                        )
                    )

            elif msg_type == "abort":
                request_id = msg["request_id"]
                if request_id in preparing:
                    if preparer is not None and preparer.cancel_queued(request_id):
                        preparing.remove(request_id)
                    else:
                        cancelled.add(request_id)
                if worker is not None:
                    worker.unregister_request(request_id)

            elif msg_type == "shutdown":
                logger.info("Shutdown received")
                if preparer is not None:
                    preparer.close()
                    preparer = None
                if worker is not None:
                    worker.shutdown()
                break

            else:
                logger.warning("Unknown message type: %s", msg_type)

    finally:
        if preparer is not None:
            preparer.close()
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
