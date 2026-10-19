import ctypes
from contextlib import contextmanager

import numpy as np
import tvm_ffi

from .native import load
from .tile_plan import KINDS, TileExecutionError, compile_graph

FUNCTIONS = ("mega_bind", "mega_max_blocks", "mega_launch")


class Completion:
    def __init__(self, plan, epoch):
        self.plan, self.epoch = plan, epoch
        self._outputs = tuple(plan.buffers[index] for index in plan.outputs)
        self.completed_tasks = 0
        self.error = None
        self.ready = False
        self.metadata = None

    def wait(self):
        if self.error is not None:
            raise self.error
        if self.ready:
            return self
        plan = self.plan
        try:
            plan.device.checked(
                plan.device.runtime.cudaEventSynchronize(plan.completion_event)
            )
            status = plan.device.download(plan.control, np.int64)
            if (
                status[2]
                or status[0] != self.epoch
                or status[3] != len(plan.graph.tasks)
            ):
                raise TileExecutionError(
                    f"tile execution failed: epoch={self.epoch}, "
                    f"status={status.tolist()}"
                )
            self.completed_tasks = int(status[3])
            self.ready = True
        except Exception as error:
            plan.failed = True
            self.error = (
                error
                if isinstance(error, TileExecutionError)
                else TileExecutionError(str(error))
            )
            if self.error is error:
                raise
            raise self.error from error
        finally:
            plan.pending = None
            self.metadata = None
        return self

    @property
    def outputs(self):
        self.wait()
        return self._outputs


class Plan:
    def __init__(self, capacity, bindings):
        self.capacity = capacity
        self.device = bindings["device"]
        self.buffers = tuple(bindings["buffers"])
        self.graph = compile_graph(
            bindings["tasks"], capacity, buffers=len(self.buffers)
        )
        self.outputs = tuple(bindings["outputs"])
        if not self.outputs or any(
            type(index) is not int or not 0 <= index < len(self.buffers)
            for index in self.outputs
        ):
            raise ValueError("invalid output bindings")
        self.module, self.source_id, self.library = load("mega", FUNCTIONS)
        maximum = self.module.mega_max_blocks()
        self.workers = capacity.workers or min(maximum, len(self.graph.tasks))
        if not 0 < self.workers <= maximum:
            raise ValueError("persistent grid exceeds cooperative residency")
        self.closed = self.failed = False
        self.leases = self.previous_epoch = 0
        self.pending = None
        descriptors, edges = [], []
        for task, children in zip(self.graph.tasks, self.graph.successors, strict=True):
            factor = int(np.float32(task.factor).view(np.int32))
            descriptors.append(
                (
                    KINDS[task.kind],
                    task.inputs[0],
                    task.inputs[1] if len(task.inputs) == 2 else -1,
                    task.output,
                    task.start,
                    task.count,
                    len(edges),
                    len(children),
                    factor,
                    len(task.depends),
                )
            )
            edges.extend(children)
        self.descriptors = self.device.upload(np.array(descriptors, dtype=np.int64))
        self.edges = self.device.upload(np.array(edges, dtype=np.int32))
        self.state = self.device.empty((len(descriptors), 2), "int32")
        self.queue = self.device.empty((capacity.queue,), "int32")
        self.control = self.device.upload(np.zeros(9, dtype=np.int64))
        self.metadata = self.device.upload(np.zeros(2, dtype=np.int64))
        self.addresses = self.device.empty((len(self.buffers),), "int64")
        self.module.mega_bind(self.buffers, self.addresses, capacity.rows)
        # Preparation can use a different stream from the first submission.
        self.device.synchronize()
        self.completion_event = ctypes.c_void_p()
        self.device.checked(
            self.device.runtime.cudaEventCreate(ctypes.byref(self.completion_event))
        )

    @contextmanager
    def graph_lease(self):
        self.check_available()
        self.leases += 1
        try:
            yield
        finally:
            self.leases -= 1

    def check_available(self):
        if self.closed or self.failed:
            raise RuntimeError("tile plan is closed or failed; discard its state")

    def launch(self):
        self.check_available()
        self.module.mega_launch(
            self.descriptors,
            self.edges,
            self.state,
            self.queue,
            self.control,
            self.metadata,
            self.addresses,
            self.capacity.rows,
            self.workers,
        )

    def close(self):
        if self.closed:
            return
        if self.leases:
            raise RuntimeError("tile plan has live CUDA Graph leases")
        error = None
        try:
            if self.pending is not None:
                self.pending.wait()
        except Exception as failure:
            error = failure
        finally:
            self.closed = True
            self.buffers = ()
            self.descriptors = self.edges = self.state = self.queue = None
            self.control = self.metadata = self.addresses = None
            try:
                self.device.checked(
                    self.device.runtime.cudaEventDestroy(self.completion_event)
                )
            except Exception:
                if error is None:
                    raise
            finally:
                self.completion_event = None
        if error is not None:
            raise error


def prepare(kind, capacity, bindings):
    if kind != "elementwise":
        raise ValueError("unsupported tile plan kind")
    return Plan(capacity, bindings)


def execute(plan, metadata, *, replay=None):
    plan.check_available()
    if plan.pending is not None:
        raise RuntimeError("observe the previous tile completion before reuse")
    metadata.validate(plan.capacity, plan.previous_epoch)
    plan.previous_epoch = metadata.epoch
    completion = Completion(plan, metadata.epoch)
    plan.pending = completion
    try:
        values = np.array((metadata.epoch, metadata.valid_rows), dtype=np.int64)
        completion.metadata = values
        plan.device.module.copy_bytes(plan.metadata, tvm_ffi.from_dlpack(values))
        (plan.launch if replay is None else replay)()
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        plan.device.checked(
            plan.device.runtime.cudaEventRecord(plan.completion_event, stream)
        )
    except Exception as error:
        plan.failed = True
        completion.error = TileExecutionError(str(error))
        try:
            plan.device.synchronize()
        except Exception:
            pass
        finally:
            plan.pending = None
        raise completion.error from error
    return completion


def close(plan):
    plan.close()
