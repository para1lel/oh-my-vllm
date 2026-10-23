from dataclasses import dataclass

from .native import load

FUNCTIONS = (
    "collective_unique_id",
    "collective_versions",
    "collective_local_device",
    "collective_images",
    "collective_init",
    "collective_poll",
    "collective_poll_enqueue",
    "collective_abort",
    "collective_abort_ready",
    "collective_record_completion",
    "collective_check",
    "collective_prepare",
    "collective_prepare_capture",
    "collective_submit",
    "collective_finish_enqueue",
)
DTYPES = ("float32", "bfloat16", "float16", "int32", "int64", "uint8")


def _module():
    return load("collectives", FUNCTIONS)[0]


def _integer(value, name, lower, upper):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} is out of bounds")


def unique_id():
    return bytes(_module().collective_unique_id())


def versions():
    return tuple(_module().collective_versions())


def local_device():
    count, uuid = _module().collective_local_device()
    return int(count), str(uuid)


@dataclass(frozen=True)
class Route:
    kind: str
    peer: int
    tensor: object

    def __post_init__(self):
        if self.kind not in ("send", "recv"):
            raise ValueError("route kind must be send or recv")
        _integer(self.peer, "peer", 0, 2**31 - 1)


class PreparedCollective:
    def __init__(self, module, handle):
        self._module, self._handle = module, handle

    def run(self):
        self._module.collective_submit(self._handle)

    def prepare_capture(self):
        self._module.collective_prepare_capture(self._handle)

    def finish_enqueue(self):
        """After capture host polling is ready, enqueue result copies on its stream."""
        self._module.collective_finish_enqueue(self._handle)


class Communicator:
    def __init__(self, world, rank, unique_id):
        _integer(world, "world", 1, 2**31 - 1)
        _integer(rank, "rank", 0, world - 1)
        if type(unique_id) is not bytes:
            raise TypeError("unique ID must be bytes")
        if len(unique_id) != 128:
            raise ValueError("unique ID must contain 128 bytes")
        self._module = _module()
        self._handle = self._module.collective_init(world, rank, unique_id)

    def poll(self):
        return str(self._module.collective_poll(self._handle))

    def poll_enqueue(self):
        """Query NCCL host enqueue only; ready does not prove device completion.

        During capture, poll after every run, then call prepared.finish_enqueue().
        Finish the last operation before the capture callback returns.
        On timeout, cancellation, or error, do not unwind the capture context.
        Keep owners alive, report through bounded CPU/IPC only, and terminate
        the rank without Python/CUDA cleanup. The supervisor must reap the group.
        """
        return str(self._module.collective_poll_enqueue(self._handle))

    def abort(self):
        self._module.collective_abort(self._handle)

    def abort_ready(self):
        return bool(self._module.collective_abort_ready(self._handle))

    def record_completion(self):
        self._module.collective_record_completion(self._handle)

    def _prepare(self, kind, tensors, directions=(), peers=()):
        import tvm_ffi

        from .device import FUNCTIONS as DEVICE_FUNCTIONS

        device = load("device", DEVICE_FUNCTIONS)[0]
        handle = self._module.collective_prepare(
            self._handle,
            kind,
            tvm_ffi.convert(tensors),
            tvm_ffi.Shape(directions),
            tvm_ffi.Shape(peers),
            device.allocate,
        )
        return PreparedCollective(self._module, handle)

    def prepare(self, kind, source, destination):
        if kind not in ("all_reduce", "all_gather", "reduce_scatter"):
            raise ValueError("unknown collective operation")
        return self._prepare(kind, [source, destination])

    def routing(self, routes):
        routes = tuple(routes)
        if any(not isinstance(route, Route) for route in routes):
            raise TypeError("routing requires Route objects")
        return self._prepare(
            "routing",
            [route.tensor for route in routes],
            [0 if route.kind == "send" else 1 for route in routes],
            [route.peer for route in routes],
        )
