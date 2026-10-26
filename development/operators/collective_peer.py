"""Native communication peer for the Rust four-rank correctness example."""

import argparse
import ctypes
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import msgpack
import numpy as np
import tvm_ffi
import zmq
from oh_my_vllm.operators.runtime.collectives import (
    FUNCTIONS,
    Communicator,
    Route,
    local_device,
    unique_id,
    versions,
)
from oh_my_vllm.operators.runtime.device import Device
from oh_my_vllm.operators.runtime.distributed import CollectivePlan
from oh_my_vllm.operators.runtime.native import file_sha256, load

DTYPES = {
    "f32": "float32",
    "bf16": "bfloat16",
    "f16": "float16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


def loaded_native_identity():
    _, source_id, library = load("collectives", FUNCTIONS)
    metadata = json.loads(library.with_name("manifest.json").read_text())
    if (
        metadata["source_id"] != source_id
        or Path(metadata["library"]).resolve() != library.resolve()
        or metadata["binary_id"] != file_sha256(library)
    ):
        raise ValueError("native evidence differs from the loaded collective library")
    return metadata


def wait_ready(communicator, timeout=30):
    end = time.monotonic() + timeout
    while True:
        status = communicator.poll()
        if status == "ready":
            return
        if status not in ("initializing", "pending"):
            raise RuntimeError(f"unexpected NCCL status: {status}")
        if time.monotonic() >= end:
            raise TimeoutError("NCCL did not complete before the local deadline")
        time.sleep(0.001)


def encode(values, dtype):
    if dtype == "bf16":
        return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return np.asarray(values, dtype=DTYPES[dtype])


def input_values(operation, rank, iteration, count):
    if operation.op == "route":
        return np.concatenate(
            [
                np.arange(operation.counts[rank * 4 + peer])
                + rank * 20
                + peer * 3
                + iteration
                for peer in range(4)
            ]
        )
    return np.arange(count) + rank * 3 + iteration * 2


def expected_values(operation, rank, iteration, count):
    if operation.op == "all_reduce":
        return np.arange(count) * 4 + 18 + iteration * 8
    if operation.op == "all_gather":
        return np.concatenate(
            [np.arange(count // 4) + peer * 3 + iteration * 2 for peer in range(4)]
        )
    if operation.op == "reduce_scatter":
        return np.arange(rank * count, (rank + 1) * count) * 4 + 18 + iteration * 8
    return np.concatenate(
        [
            np.arange(operation.counts[peer * 4 + rank])
            + peer * 20
            + rank * 3
            + iteration
            for peer in range(4)
        ]
    )


def clone_capture(device, operation):
    stream, graph, clone, executable, cloned = (ctypes.c_void_p() for _ in range(5))
    runtime = device.runtime
    runtime.cudaGraphClone.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    runtime.cudaGraphClone.restype = ctypes.c_int
    temporary = (
        ("cudaGraphDestroy", clone),
        ("cudaGraphDestroy", graph),
        ("cudaStreamDestroy", stream),
    )
    try:
        device.checked(runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
        with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
            device.checked(runtime.cudaStreamBeginCapture(stream, 1))
            try:
                operation()
            except BaseException:
                runtime.cudaStreamEndCapture(stream, ctypes.byref(graph))
                raise
            device.checked(runtime.cudaStreamEndCapture(stream, ctypes.byref(graph)))
        device.checked(runtime.cudaGraphClone(ctypes.byref(clone), graph))
        for source, target in ((graph, executable), (clone, cloned)):
            device.checked(
                runtime.cudaGraphInstantiateWithFlags(ctypes.byref(target), source, 0)
            )
        for name, handle in temporary:
            device.checked(getattr(runtime, name)(handle))
            handle.value = None
        return executable, cloned
    except BaseException:
        device.destroy_handles(
            (
                ("cudaGraphExecDestroy", executable),
                ("cudaGraphExecDestroy", cloned),
                *temporary,
            ),
            suppress=True,
        )
        raise


class DiscardCapture(Exception):
    pass


class Probe:
    def __init__(self, device, communicator, plan, rank):
        self.device, self.communicator = device, communicator
        self.plan, self.rank = plan, rank
        self.uploads = device.uploads()
        self.buffers, self.prepared = [], []
        self.graph_resources = ExitStack()
        self.replay = None
        for operation, (inputs, outputs) in zip(
            plan.operations, plan.buffer_counts(rank), strict=True
        ):
            source = device.empty((inputs,), DTYPES[operation.dtype])
            destination = device.empty((outputs,), DTYPES[operation.dtype])
            self.buffers.append((source, destination))
            if operation.op == "route":
                routes = []
                sent = received = 0
                for peer in range(4):
                    send_count = operation.counts[rank * 4 + peer]
                    recv_count = operation.counts[peer * 4 + rank]
                    routes.extend(
                        (
                            Route(
                                "send", peer, device.view(source, (send_count,), sent)
                            ),
                            Route(
                                "recv",
                                peer,
                                device.view(destination, (recv_count,), received),
                            ),
                        )
                    )
                    sent += send_count
                    received += recv_count
                prepared = communicator.routing(routes)
            else:
                prepared = communicator.prepare(operation.op, source, destination)
            self.prepared.append(prepared)

    def fill(self, iteration):
        for operation, (source, destination) in zip(
            self.plan.operations, self.buffers, strict=True
        ):
            values = input_values(operation, self.rank, iteration, source.shape[0])
            self.uploads.copy(source, encode(values, operation.dtype))
            self.uploads.copy(
                destination, encode(np.full(destination.shape, 99), operation.dtype)
            )

    def run(self):
        for prepared in self.prepared:
            try:
                prepared.run()
                end = time.monotonic() + 30
                while True:
                    status = self.communicator.poll_enqueue()
                    if status == "ready":
                        break
                    if status != "pending" or time.monotonic() >= end:
                        raise TimeoutError(
                            f"capture enqueue did not complete: {status}"
                        )
                    time.sleep(0.001)
                prepared.finish_enqueue()
            except BaseException as error:
                # Capture unwinding would call CUDA while NCCL can still be pending.
                try:
                    print(
                        f"fatal capture enqueue: {error}", file=sys.stderr, flush=True
                    )
                finally:
                    os._exit(32)

    def execute(self, mode, iteration):
        self.fill(iteration)
        if mode == "eager":
            for prepared in self.prepared:
                prepared.run()
                wait_ready(self.communicator)
        elif mode == "graph":
            if self.replay is None:
                for prepared in self.prepared:
                    prepared.prepare_capture()
                self.replay = self.graph_resources.enter_context(
                    self.device.graph(self.run)
                )
            self.replay()
            self.communicator.record_completion()
            wait_ready(self.communicator)
        else:
            raise ValueError("unexpected execution mode")
        self.check_outputs(iteration)

    def check_outputs(self, iteration):
        for operation, (_, destination) in zip(
            self.plan.operations, self.buffers, strict=True
        ):
            dtype = np.uint16 if operation.dtype == "bf16" else DTYPES[operation.dtype]
            actual = self.device.download(destination, dtype)
            expected = encode(
                expected_values(operation, self.rank, iteration, destination.shape[0]),
                operation.dtype,
            )
            np.testing.assert_array_equal(actual, expected)

    def wait_graph_release(self):
        end = time.monotonic() + 30
        while not self.communicator.abort_ready():
            if time.monotonic() >= end:
                raise TimeoutError("captured communication owners did not release")
            time.sleep(0.001)
        wait_ready(self.communicator)

    def check_graph_lifetime(self):
        self.graph_resources.close()
        self.wait_graph_release()
        for prepared in self.prepared:
            prepared.prepare_capture()

        def discard():
            self.run()
            raise DiscardCapture

        try:
            self.device.capture(discard)
        except DiscardCapture:
            pass
        else:
            raise AssertionError("deliberate capture failure was not raised")
        self.wait_graph_release()
        for prepared in self.prepared:
            prepared.prepare_capture()
        executable, cloned = clone_capture(self.device, self.run)
        try:
            for iteration, handle in ((3, executable), (4, cloned)):
                if self.communicator.abort_ready():
                    raise AssertionError("live graph executable did not retain owners")
                self.fill(iteration)
                stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
                self.device.checked(self.device.runtime.cudaGraphLaunch(handle, stream))
                self.communicator.record_completion()
                wait_ready(self.communicator)
                self.check_outputs(iteration)
                self.device.release_graph(handle)
                handle.value = None
        finally:
            for handle in (executable, cloned):
                if handle:
                    self.device.release_graph(handle)
        self.wait_graph_release()
        return {
            "failed_capture_cleanup": True,
            "clone_replays": 2,
            "owner_guards": True,
        }

    def close(self):
        self.graph_resources.close()
        self.wait_graph_release()
        end = time.monotonic() + 30
        self.communicator.abort()
        while self.communicator.poll() != "aborted":
            if time.monotonic() >= end:
                raise TimeoutError("aborted communication still owns device work")
            time.sleep(0.001)
        self.uploads.close()
        self.prepared.clear()
        self.buffers.clear()


def send(socket, value):
    socket.send(msgpack.packb(value, use_bin_type=True))


def receive(socket):
    frames = socket.recv_multipart()
    if len(frames) != 1 or len(frames[0]) > 1 << 20:
        raise ValueError("invalid controller frame")
    return msgpack.unpackb(frames[0], raw=False)


def run(args, socket):
    init = receive(socket)
    header = init["header"]
    if init["type"] != "init" or init["transport"] != "nccl":
        raise ValueError("native probe requires NCCL initialization")
    if header["version"] != 3 or header["rank"] != args.rank:
        raise ValueError("native probe header differs")
    count, uuid = local_device()
    if count != 1 or uuid != init["device_uuids"][args.rank]:
        raise ValueError("actual local CUDA device differs from group identity")
    record = {
        "pid": os.getpid(),
        "device_uuid": uuid,
        "nccl_versions": versions(),
        "steps": [],
        "native": loaded_native_identity(),
    }
    args.output.joinpath(f"rank-{args.rank}.json").write_text(
        json.dumps(record, indent=2)
    )
    send(
        socket,
        {
            "type": "bootstrap",
            "header": header,
            "device_uuid": uuid,
            "unique_id": list(unique_id()) if args.rank == 0 else None,
        },
    )
    connect = receive(socket)
    if connect["type"] != "connect" or connect["header"] != header:
        raise ValueError("NCCL connect header differs")
    communicator = Communicator(4, args.rank, bytes(connect["unique_id"]))
    wait_ready(communicator, timeout=120)
    device, probe = Device(), None
    send(
        socket,
        {
            "type": "ready",
            "header": header,
            "device_uuid": uuid,
            "descriptor": init["descriptor"],
            "bootstrap_nonce": connect["bootstrap_nonce"],
        },
    )
    while True:
        message = receive(socket)
        if message["type"] == "shutdown":
            if message["header"] != header or len(record["steps"]) != 3:
                raise ValueError("unexpected native probe shutdown")
            record["graceful_shutdown"] = True
            args.output.joinpath(f"rank-{args.rank}.json").write_text(
                json.dumps(record, indent=2)
            )
            socket.setsockopt(zmq.LINGER, 1000)
            send(socket, {"type": "closed", "header": header})
            return
        if message["type"] != "execute":
            raise ValueError("unexpected native probe command")
        if message["version"] != 3 or message["epoch"] != header["epoch"]:
            raise ValueError("stale execute")
        plan = CollectivePlan.from_wire(message["collective_plan"], world_size=4)
        if (
            plan.sha256 != message["collective_plan_sha256"]
            or len(plan.operations) != message["collective_count"]
        ):
            raise ValueError("collective plan digest or count differs")
        if probe is None:
            probe = Probe(device, communicator, plan, args.rank)
        elif plan != probe.plan:
            raise ValueError("probe plan changed across graph replay")
        if args.rank == 2 and message["step_id"] == 1 and args.failure != "none":
            record["injected_failure"] = args.failure
            args.output.joinpath(f"rank-{args.rank}.json").write_text(
                json.dumps(record, indent=2)
            )
            time.sleep(0.25 if args.failure == "exit" else 120)
            os._exit(31)
        try:
            probe.execute(message["mode"], message["step_id"] - 1)
        except BaseException as error:
            send(socket, {"type": "error", "header": header, "message": str(error)})
            raise
        record["steps"].append(
            {
                "step_id": message["step_id"],
                "mode": message["mode"],
                "plan_sha256": plan.sha256,
                "operations": len(plan.operations),
            }
        )
        args.output.joinpath(f"rank-{args.rank}.json").write_text(
            json.dumps(record, indent=2)
        )
        if message["step_id"] == 3:
            record["graph_lifetime"] = probe.check_graph_lifetime()
            probe.close()
            device.close()
            record["allocated_bytes_after_close"] = int(device.module.allocated_bytes())
            if record["allocated_bytes_after_close"] != 0:
                raise RuntimeError("native probe retained device buffers after closure")
            args.output.joinpath(f"rank-{args.rank}.json").write_text(
                json.dumps(record, indent=2)
            )
        send(
            socket,
            {
                "type": "result",
                "header": header,
                "step_id": message["step_id"],
                "collective_plan_sha256": plan.sha256,
                "completed_collectives": len(plan.operations),
                "transaction_ids": [
                    request["transaction_id"]
                    for request in message["ranks"][args.rank]["requests"]
                ],
                "device_work_complete": True,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure", choices=("none", "exit", "hang"), default="none")
    args = parser.parse_args()
    with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(args.socket)
        run(args, socket)


if __name__ == "__main__":
    main()
