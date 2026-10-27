"""Four-rank DeepSeek worker bridge.

The Rust worker protocol stays one DEALER stream. This coordinator fans each
model command out to four CUDA processes and publishes rank zero's result.
"""

import contextlib
import ctypes
import multiprocessing
import os
import signal
import sys
import tempfile
import time
from queue import Empty

WORLD_SIZE = 4


def _error_message(error):
    detail = str(error).strip()
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _set_parent_death_signal():
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None)
    libc.prctl(1, signal.SIGTERM, 0, 0, 0)


def _visible_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [item for item in value.split(",") if item] if value else []
    if not devices:
        devices = [str(index) for index in range(WORLD_SIZE)]
    if len(devices) < WORLD_SIZE:
        raise ValueError("DeepSeek needs four visible CUDA devices")
    return devices[:WORLD_SIZE]


def _rank_entry(
    rank,
    config,
    raw_config,
    strategy,
    init_method,
    device,
    commands,
    results,
):
    parent_pid = os.getppid()
    _set_parent_death_signal()
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if rank:
        os.environ.pop("OH_MY_VLLM_EXECUTION_TRACE", None)
    import torch
    import torch.distributed as dist

    worker = None
    try:
        dist.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=WORLD_SIZE,
        )
        from .inference import Worker

        worker = Worker(config, raw_config=raw_config, strategy=strategy)
        results.put((rank, worker.ready()))
        while True:
            message = commands.get()
            if message is None:
                break
            try:
                reply = worker.handle(message)
                results.put((rank, reply if reply is not None else {"type": "ack"}))
            except Exception as error:
                results.put(
                    (
                        rank,
                        {
                            "type": "error",
                            "rpc_id": message.get("rpc_id", 0),
                            "kind": "internal",
                            "message": _error_message(error),
                        },
                    )
                )
                break
    except Exception as error:
        results.put(
            (
                rank,
                {
                    "type": "error",
                    "rpc_id": 0,
                    "kind": "internal",
                    "message": _error_message(error),
                },
            )
        )
    finally:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


class GroupWorker:
    def __init__(self, config, *, raw_config):
        strategy = os.environ.get("OH_MY_VLLM_DEEPSEEK_STRATEGY", "tep4").lower()
        if strategy not in {"tp4", "tep4", "dep4"}:
            raise ValueError("DeepSeek strategy must be TP4, TEP4, or DEP4")
        self.strategy = strategy
        self.context = multiprocessing.get_context("spawn")
        self.commands = [self.context.Queue(maxsize=2) for _ in range(WORLD_SIZE)]
        self.results = self.context.Queue()
        self.processes = []
        self.store_directory = tempfile.TemporaryDirectory(
            prefix=f"oh-my-vllm-deepseek-{os.getpid()}-"
        )
        self.store_path = os.path.join(self.store_directory.name, "store")
        self.init_method = f"file://{self.store_path}"
        self._closed = False
        devices = _visible_devices()
        try:
            for rank, device in enumerate(devices):
                process = self.context.Process(
                    target=_rank_entry,
                    args=(
                        rank,
                        dict(config),
                        raw_config,
                        strategy,
                        self.init_method,
                        device,
                        self.commands[rank],
                        self.results,
                    ),
                )
                process.start()
                self.processes.append(process)
            ready = self._collect()
            self._ready = self._validate_ready(ready)
        except BaseException:
            self.close()
            raise

    def _collect(self):
        replies = {}
        deadline = time.monotonic() + 1800
        for _ in range(WORLD_SIZE):
            if any(
                not process.is_alive() for process in getattr(self, "processes", [])
            ):
                raise RuntimeError("DeepSeek rank exited before reply")
            timeout = min(1, deadline - time.monotonic())
            if timeout <= 0:
                raise RuntimeError("DeepSeek rank group did not reply")
            try:
                rank, reply = self.results.get(timeout=timeout)
            except Empty as error:
                if any(
                    not process.is_alive() for process in getattr(self, "processes", [])
                ):
                    raise RuntimeError("DeepSeek rank exited before reply") from error
                raise RuntimeError("DeepSeek rank group did not reply") from error
            if rank in replies:
                raise RuntimeError("DeepSeek rank group returned a duplicate reply")
            replies[rank] = reply
        if set(replies) != set(range(WORLD_SIZE)):
            raise RuntimeError("DeepSeek rank group returned an incomplete reply")
        errors = [reply for reply in replies.values() if reply.get("type") == "error"]
        if errors:
            raise RuntimeError(errors[0]["message"])
        return replies

    @staticmethod
    def _validate_ready(replies):
        ready = replies[0]
        descriptor = ready["descriptor"]
        for rank in range(1, WORLD_SIZE):
            if replies[rank]["descriptor"] != descriptor:
                raise ValueError(
                    f"DeepSeek rank {rank} descriptor differs from rank zero"
                )
            if (
                replies[rank]["logical_num_blocks"] != ready["logical_num_blocks"]
                or replies[rank]["mamba_blocks"] != ready["mamba_blocks"]
            ):
                raise ValueError(
                    f"DeepSeek rank {rank} capacity differs from rank zero"
                )
        return dict(ready)

    def ready(self):
        return dict(self._ready)

    def handle(self, message):
        for command in self.commands:
            command.put(dict(message))
        try:
            replies = self._collect()
        except BaseException:
            self.close()
            raise
        return replies[0] if replies[0].get("type") != "ack" else None

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for command in getattr(self, "commands", []):
            with contextlib.suppress(Exception):
                command.put_nowait(None)
        for process in getattr(self, "processes", []):
            with contextlib.suppress(BaseException):
                process.join(timeout=30)
            try:
                alive = process.is_alive()
            except BaseException:
                alive = False
            if alive:
                with contextlib.suppress(BaseException):
                    process.terminate()
                with contextlib.suppress(BaseException):
                    process.join(timeout=5)
            try:
                alive = process.is_alive()
            except BaseException:
                alive = False
            if alive:
                with contextlib.suppress(BaseException):
                    stop = getattr(process, "kill", process.terminate)
                    stop()
                with contextlib.suppress(BaseException):
                    process.join(timeout=5)
        for queue in (*getattr(self, "commands", []), getattr(self, "results", None)):
            if queue is not None:
                queue.close()
        store_directory = getattr(self, "store_directory", None)
        if store_directory is not None:
            store_directory.cleanup()
            self.store_directory = None
