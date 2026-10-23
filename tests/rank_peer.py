"""CPU-only peer for Rust rank-supervisor process tests."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import msgpack
import zmq


def ready(init, args):
    header = init["header"].copy()
    if args.rank == 2 and args.failure == "bad_ready":
        header["epoch"] += 1
    return {
        "type": "ready",
        "header": header,
        "device_uuid": init["device_uuids"][args.rank],
        "descriptor": init["descriptor"],
    }


def bootstrap(init, args):
    reply = {
        "type": "bootstrap",
        "header": init["header"].copy(),
        "device_uuid": init["device_uuids"][args.rank],
        "unique_id": list(range(17, 145)) if args.rank == 0 else None,
    }
    if args.rank == 0:
        if args.failure == "bootstrap_missing_id":
            reply["unique_id"] = None
        if args.failure == "bootstrap_short_id":
            reply["unique_id"].pop()
    if args.rank == 2:
        if args.failure in ("bootstrap_epoch", "bootstrap_rank", "bootstrap_version"):
            reply["header"][args.failure.removeprefix("bootstrap_")] += 1
        if args.failure == "bootstrap_device":
            reply["device_uuid"] = init["device_uuids"][0]
        if args.failure == "bootstrap_extra_id":
            reply["unique_id"] = list(range(17, 145))
        if args.failure == "early_ready":
            return ready(init, args)
        if args.failure == "bootstrap_hang":
            time.sleep(60)
    return reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--failure", default="none")
    args = parser.parse_args()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    args.records.joinpath(f"rank-{args.rank}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "child": child.pid,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    )
    if args.rank == 2 and args.failure == "startup_exit":
        return
    with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IDENTITY, f"rank-{args.rank}".encode())
        socket.connect(args.socket)
        while True:
            message = msgpack.unpackb(socket.recv(), raw=False)
            if message["type"] == "shutdown":
                if args.rank == 2 and args.failure == "shutdown_hang":
                    time.sleep(60)
                child.terminate()
                child.wait()
                header = message["header"].copy()
                if args.rank == 2 and args.failure == "shutdown_stale":
                    header["epoch"] += 1
                if args.rank == 3 and args.failure == "shutdown_duplicate":
                    while not all(
                        args.records.joinpath(f"closed-{rank}.sent").exists()
                        for rank in range(3)
                    ):
                        time.sleep(0.001)
                socket.setsockopt(zmq.LINGER, 1000)
                socket.send(msgpack.packb({"type": "closed", "header": header}))
                args.records.joinpath(f"closed-{args.rank}.sent").touch()
                if args.rank == 3 and args.failure == "shutdown_duplicate":
                    socket.send(msgpack.packb({"type": "closed", "header": header}))
                    time.sleep(0.1)
                if args.rank == 2 and args.failure == "shutdown_nonzero":
                    sys.exit(7)
                return
            if message["type"] == "init":
                init = message
                if args.rank == 3 and args.failure == "premature_ready":
                    while not all(
                        args.records.joinpath(f"bootstrap-{rank}.sent").exists()
                        for rank in range(3)
                    ):
                        time.sleep(0.001)
                reply = (
                    bootstrap(init, args)
                    if init["transport"] == "nccl"
                    else ready(init, args)
                )
                if args.rank == 2 and args.failure == "bootstrap_duplicate":
                    socket.send(msgpack.packb(reply, use_bin_type=True))
            elif message["type"] == "connect":
                if args.rank == 3 and args.failure == "premature_ready":
                    time.sleep(60)
                assert message["header"] == init["header"]
                assert message["unique_id"] == list(range(17, 145))
                reply = ready(init, args)
                reply["bootstrap_nonce"] = message["bootstrap_nonce"]
                if args.rank == 2 and args.failure == "wrong_nonce":
                    reply["bootstrap_nonce"] = "0" * 64
                if args.rank == 2 and args.failure == "missing_nonce":
                    del reply["bootstrap_nonce"]
            else:
                if args.rank == 2 and args.failure == "step_exit":
                    return
                if args.rank == 2 and args.failure == "step_hang":
                    time.sleep(60)
                reply = {
                    "type": "result",
                    "header": {
                        "version": message["version"],
                        "epoch": message["epoch"],
                        "rank": args.rank,
                    },
                    "step_id": message["step_id"],
                    "collective_plan_sha256": message["collective_plan_sha256"],
                    "completed_collectives": message["collective_count"],
                    "transaction_ids": [
                        request["transaction_id"]
                        for request in message["ranks"][args.rank]["requests"]
                    ],
                    "device_work_complete": True,
                }
                if args.rank == 2 and args.failure == "stale_result":
                    reply["step_id"] -= 1
            socket.send(msgpack.packb(reply, use_bin_type=True))
            if message["type"] == "init" and args.failure == "premature_ready":
                args.records.joinpath(f"bootstrap-{args.rank}.sent").touch()
                if args.rank == 3:
                    socket.send(msgpack.packb(ready(init, args), use_bin_type=True))


if __name__ == "__main__":
    main()
