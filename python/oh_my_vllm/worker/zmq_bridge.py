import argparse

import msgpack
import zmq

from ..models.descriptor import MODEL_DESCRIPTOR_VERSION
from .wire import validate_execute, validate_init, validate_result


def initialize(config):
    validate_init(config)
    version = config.get("descriptor_version")
    if type(version) is not int or version != MODEL_DESCRIPTOR_VERSION:
        raise ValueError("unsupported or missing model descriptor version")
    return create_worker(config)


def create_worker(config):
    from ..models.registry import create

    return create(config)


def error_kind(message, error):
    text = str(error).lower()
    if "out of memory" in text:
        return "oom"
    if any(
        value in text
        for value in (
            "illegal memory access",
            "illegal instruction",
            "device-side assert",
            "unspecified launch failure",
            "misaligned address",
            "launch timeout",
        )
    ):
        return "device"
    return "validation" if message["type"] == "prepare" else "internal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    options = parser.parse_args()
    with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(options.socket)
        worker = None
        worker_epoch = None
        try:
            while True:
                message = msgpack.unpackb(socket.recv(), raw=False)
                if message["type"] == "shutdown":
                    return
                try:
                    if message["type"] == "init":
                        if worker is not None:
                            previous, worker = worker, None
                            try:
                                getattr(previous, "close", lambda: None)()
                            finally:
                                del previous
                        worker = initialize(message)
                        worker_epoch = message["worker_epoch"]
                        reply = worker.ready()
                        if (
                            reply.get("protocol_version") != message["protocol_version"]
                            or reply.get("worker_epoch") != worker_epoch
                        ):
                            raise ValueError("worker Ready execution identity differs")
                    elif worker is None:
                        raise ValueError("worker is not initialized")
                    else:
                        if message["type"] == "execute":
                            expected = validate_execute(message, worker_epoch)
                            reply = worker.handle(message)
                            validate_result(message, reply, expected)
                        else:
                            reply = worker.handle(message)
                except Exception as error:
                    reply = {
                        "type": "error",
                        "rpc_id": message.get("rpc_id", 0),
                        "kind": error_kind(message, error),
                        "message": str(error),
                    }
                if reply is not None:
                    socket.send(msgpack.packb(reply, use_bin_type=True))
        finally:
            if worker is not None:
                getattr(worker, "close", lambda: None)()


if __name__ == "__main__":
    main()
