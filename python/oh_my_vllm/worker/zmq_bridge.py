import argparse

import msgpack
import zmq

from .placeholder import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    options = parser.parse_args()
    with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(options.socket)
        worker = None
        while True:
            message = msgpack.unpackb(socket.recv(), raw=False)
            if message["type"] == "shutdown":
                return
            try:
                if message["type"] == "init":
                    worker = Worker(message)
                    reply = worker.ready()
                elif worker is None:
                    raise ValueError("worker is not initialized")
                else:
                    reply = worker.handle(message)
            except Exception as error:
                reply = {
                    "type": "error",
                    "rpc_id": message.get("rpc_id", 0),
                    "kind": "validation"
                    if message["type"] == "prepare"
                    else "internal",
                    "message": str(error),
                }
            if reply is not None:
                socket.send(msgpack.packb(reply, use_bin_type=True))


if __name__ == "__main__":
    main()
