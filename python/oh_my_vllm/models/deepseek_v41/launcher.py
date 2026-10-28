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
from multiprocessing.connection import wait as wait_connections
from queue import Empty

WORLD_SIZE = 4
STARTUP_TIMEOUT_SECONDS = 1800
COMMAND_TIMEOUT_SECONDS = 600


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
        results.send((rank, worker.ready()))
        while True:
            message = commands.get()
            if message is None:
                break
            try:
                reply = worker.handle(message)
                results.send((rank, reply if reply is not None else {"type": "ack"}))
            except Exception as error:
                results.send(
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
        message = _error_message(error)
        print(
            f"DeepSeek rank {rank} failed during startup or execution: {message}",
            file=sys.stderr,
            flush=True,
        )
        results.send(
            (
                rank,
                {
                    "type": "error",
                    "rpc_id": 0,
                    "kind": "internal",
                    "message": message,
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
        self.result_readers = []
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
                reader, writer = self.context.Pipe(duplex=False)
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
                        writer,
                    ),
                )
                try:
                    process.start()
                finally:
                    writer.close()
                self.processes.append(process)
                self.result_readers.append(reader)
            ready = self._collect(STARTUP_TIMEOUT_SECONDS)
            self._ready = self._validate_ready(ready)
        except BaseException:
            self.close()
            raise

    def _collect(self, timeout_seconds=STARTUP_TIMEOUT_SECONDS):
        replies = {}
        deadline = time.monotonic() + timeout_seconds
        readers = getattr(self, "result_readers", None)
        while len(replies) < WORLD_SIZE:
            timeout = min(1, deadline - time.monotonic())
            if timeout <= 0:
                raise RuntimeError("DeepSeek rank group did not reply")
            try:
                if readers:
                    ready = wait_connections(readers, timeout)
                    if not ready:
                        self._raise_if_rank_exited()
                        continue
                    for reader in ready:
                        rank = readers.index(reader)
                        try:
                            reported_rank, reply = reader.recv()
                        except (EOFError, OSError) as error:
                            self._raise_if_rank_exited(rank)
                            raise RuntimeError(
                                f"DeepSeek rank pipe closed before reply (rank {rank})"
                            ) from error
                        if reported_rank != rank:
                            raise RuntimeError(
                                "DeepSeek rank result came from the wrong channel "
                                f"(expected {rank}, got {reported_rank})"
                            )
                        self._store_reply(replies, rank, reply)
                else:
                    self._raise_if_rank_exited()
                    rank, reply = self._result(timeout)
                    self._store_reply(replies, rank, reply)
            except Empty:
                self._raise_if_rank_exited()
                continue
        if set(replies) != set(range(WORLD_SIZE)):
            raise RuntimeError("DeepSeek rank group returned an incomplete reply")
        return replies

    def _raise_if_rank_exited(self, selected_rank=None):
        processes = getattr(self, "processes", [])
        ranks = range(len(processes)) if selected_rank is None else (selected_rank,)
        for rank in ranks:
            process = processes[rank]
            exitcode = getattr(process, "exitcode", None)
            if exitcode is not None:
                raise RuntimeError(
                    "DeepSeek rank exited before reply "
                    f"(rank {rank}, exit code {exitcode})"
                )

    @staticmethod
    def _store_reply(replies, rank, reply):
        if rank in replies:
            raise RuntimeError("DeepSeek rank group returned a duplicate reply")
        if reply.get("type") == "error":
            raise RuntimeError(
                f"DeepSeek rank {rank} returned an error: {reply['message']}"
            )
        replies[rank] = reply

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

    def _result(self, timeout):
        reader = getattr(self.results, "_reader", None)
        if reader is None:
            return self.results.get(timeout=timeout)
        if not reader.poll(timeout):
            raise Empty
        try:
            return reader.recv()
        except (EOFError, OSError) as error:
            raise Empty from error

    def handle(self, message):
        for command in self.commands:
            command.put(dict(message))
        try:
            replies = self._collect(COMMAND_TIMEOUT_SECONDS)
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
        for queue in getattr(self, "commands", []):
            queue.close()
        for reader in getattr(self, "result_readers", []):
            reader.close()
        result_queue = getattr(self, "results", None)
        if result_queue is not None:
            result_queue.close()
        store_directory = getattr(self, "store_directory", None)
        if store_directory is not None:
            store_directory.cleanup()
            self.store_directory = None
