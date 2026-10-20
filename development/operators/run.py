from contextlib import ExitStack, contextmanager

import numpy as np
import tvm_ffi

from .storage import decode, equal_bits, state_rows, storage_dtype


class PreparedCase:
    def __init__(self, case, device, fixture):
        self.case, self.device, self.fixture = case, device, fixture
        self.resources = ExitStack()
        self.graphs = ExitStack()
        self.owners, self.host, self.restores = [], [], []
        self.outputs, self.observers = {}, {}
        self.operation = self.replay = None
        self.closed = False
        self.started = False
        self.weights = None
        self.inputs, self.state = {}, {}
        try:
            self.inputs = self.upload_group("inputs", fixture.inputs)
            self.state = self.upload_group("state", fixture.state)
        except BaseException:
            self.close()
            raise

    def upload_group(self, group, arrays):
        result = {}
        for name, initial in arrays.items():
            host = np.require(initial, requirements=["C", "W"])
            source = tvm_ffi.from_dlpack(host)
            target = self.device.empty(host.shape, self.fixture.dtype(group, name))
            self.device.module.copy_bytes(target, source)
            self.host.append((host, source))
            self.restores.append((target, source))
            result[name] = target
        return result

    def own(self, operation):
        self.owners.append(operation)
        close = getattr(operation, "close", None)
        if close is not None:
            self.resources.callback(close)
        return operation

    def require_open(self):
        if self.closed:
            raise RuntimeError("prepared case is closed")

    def reset(self):
        self.require_open()
        for target, source in self.restores:
            self.device.module.copy_bytes(target, source)

    @contextmanager
    def graph_lease(self):
        with ExitStack() as leases:
            for owner in self.owners:
                acquire = getattr(owner, "graph_lease", None)
                if acquire is not None:
                    leases.enter_context(acquire())
            yield

    def run(self):
        self.operation()

    def start(self, *, warmup=None):
        self.require_open()
        if self.started or self.operation is None:
            raise RuntimeError("prepared case has no operation or is already started")
        try:
            if warmup is not None:
                warmup()
            self.reset()
            self.device.synchronize()
            if self.case.mode == "graph":
                self.replay = self.graphs.enter_context(self.device.graph(self.run))
            self.started = True
        except BaseException:
            self.close()
            raise

    def invoke(self):
        self.require_open()
        if not self.started:
            raise RuntimeError("prepared case has not started")
        (self.replay or self.run)()

    def raw(self, tensor):
        return self.device.download(tensor, storage_dtype(str(tensor.dtype)))

    def download(self, tensor):
        return decode(self.raw(tensor), str(tensor.dtype))

    def observe(self):
        self.require_open()
        actual = {name: self.download(value) for name, value in self.outputs.items()}
        actual.update({name: observe() for name, observe in self.observers.items()})
        guards = []
        for name, tensor in self.state.items():
            raw = self.raw(tensor)
            write = self.fixture.manifest["writes"].get(name)
            if write is None:
                guards.append(equal_bits(raw, self.fixture.state[name]))
                continue
            rows = state_rows(raw, write["layout"])
            initial = state_rows(self.fixture.state[name], write["layout"])
            indices = np.asarray(write["indices"], dtype=np.int64)
            actual[write["output"]] = decode(rows[indices], str(tensor.dtype))
            untouched = np.ones(len(rows), dtype=bool)
            untouched[indices] = False
            guards.append(equal_bits(rows[untouched], initial[untouched]))
        for name, tensor in self.inputs.items():
            if not equal_bits(self.raw(tensor), self.fixture.inputs[name]):
                raise ValueError(f"operator changed read-only input: {name}")
        if "guards" in self.case.outputs:
            actual["guards"] = np.asarray(guards, dtype=bool)
        elif not all(guards):
            raise ValueError("operator changed an unassigned state row")
        if set(actual) != set(self.case.outputs):
            raise ValueError("observations differ from the complete boundary contract")
        return actual

    def close(self):
        if self.closed:
            return
        self.closed = True
        with ExitStack() as cleanup:
            cleanup.callback(self.release_references)
            cleanup.callback(self.resources.close)
            cleanup.callback(self.device.synchronize)
            self.graphs.close()

    def release_references(self):
        self.replay = self.operation = None
        self.outputs.clear()
        self.observers.clear()
        self.owners.clear()
        self.inputs.clear()
        self.state.clear()
        self.restores.clear()
        self.host.clear()
        self.weights = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prepare(case, device, fixture, *, model):
    from .adapters import fa, ffn, gdn, mtp, output, vision
    from .weights import CaseWeights

    prepared = PreparedCase(case, device, fixture)
    try:
        prepared.weights = CaseWeights(device, model, case)
        prepared.weights.verify(fixture.manifest["checkpoint"])
        family = case.boundary.split(".")[0]
        adapter = {
            "fa": fa,
            "gdn": gdn,
            "ffn": ffn,
            "output": output,
            "mtp": mtp,
            "vision": vision,
        }[family]
        warmup = adapter.bind(prepared)
        prepared.start(warmup=warmup)
        return prepared
    except BaseException:
        prepared.close()
        raise
