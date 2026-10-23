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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--failure", default="none")
    args = parser.parse_args()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    args.records.joinpath(f"rank-{args.rank}.json").write_text(
        json.dumps({"pid": os.getpid(), "child": child.pid})
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
                return
            if message["type"] == "init":
                header = message["header"].copy()
                if args.rank == 2 and args.failure == "bad_ready":
                    header["epoch"] += 1
                reply = {
                    "type": "ready",
                    "header": header,
                    "device_uuid": message["device_uuids"][args.rank],
                    "descriptor": message["descriptor"],
                }
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


if __name__ == "__main__":
    main()
