import ctypes
import os
from contextlib import ExitStack, contextmanager, suppress
from functools import lru_cache
from pathlib import Path

import numpy as np
import tvm_ffi

from .native import load
from .uploads import UploadPool, Uploads

FUNCTIONS = (
    "allocate",
    "allocate_scratch",
    "reserve_scratch",
    "copy_bytes",
    "copy_batch",
    "upload_bytes",
    "synchronize",
    "view",
    "zero",
    "gather_rows",
    "free_bytes",
    "allocated_bytes",
    "reserved_bytes",
)


@lru_cache(maxsize=128)
def tensor_shape(dimensions):
    return tvm_ffi.Shape(dimensions)


class Device:
    def __init__(self):
        self.module, self.source_id, self.library = load("device", FUNCTIONS)
        cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        self.runtime = ctypes.CDLL(str(cuda / "lib64/libcudart.so"))
        for name, arguments in {
            "cudaMemGetInfo": [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cudaHostAlloc": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ],
            "cudaFreeHost": [ctypes.c_void_p],
            "cudaEventCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "cudaStreamIsCapturing": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
            "cudaEventCreate": [ctypes.POINTER(ctypes.c_void_p)],
            "cudaEventDestroy": [ctypes.c_void_p],
            "cudaEventQuery": [ctypes.c_void_p],
            "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaEventSynchronize": [ctypes.c_void_p],
            "cudaEventElapsedTime": [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
            "cudaStreamCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "cudaStreamDestroy": [ctypes.c_void_p],
            "cudaStreamQuery": [ctypes.c_void_p],
            "cudaStreamSynchronize": [ctypes.c_void_p],
            "cudaStreamWaitEvent": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint],
            "cudaStreamBeginCapture": [ctypes.c_void_p, ctypes.c_int],
            "cudaStreamEndCapture": [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            "cudaGraphInstantiateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_ulonglong,
            ],
            "cudaGraphUpload": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaGraphLaunch": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaGraphDestroy": [ctypes.c_void_p],
            "cudaGraphExecDestroy": [ctypes.c_void_p],
            "cudaProfilerStart": [],
            "cudaProfilerStop": [],
        }.items():
            function = getattr(self.runtime, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        self.runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.runtime.cudaGetErrorString.restype = ctypes.c_char_p
        self.transfer_engines = set()
        self.upload_pool = UploadPool(self)
        self.graph_failed = False
        self._failed_graph_resources = ExitStack()

    def close(self):
        self.upload_pool.close()

    def memory_info(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.checked(
            self.runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        )
        return free.value, total.value

    def empty(self, shape, dtype):
        return self.module.allocate(tensor_shape(tuple(shape)), dtype, 0)

    def uploads(self):
        return Uploads(self)

    def scratch(self, shape, dtype):
        return self.module.allocate_scratch(tensor_shape(tuple(shape)), dtype, 0)

    def zeros(self, shape, dtype):
        result = self.empty(shape, dtype)
        self.module.zero(result)
        return result

    def view(self, value, shape, offset=0):
        return self.module.view(value, tensor_shape(tuple(shape)), offset)

    def upload(self, value, dtype=None):
        value = np.require(value, requirements=["C", "W"])
        output = self.empty(value.shape, dtype or str(value.dtype))
        self.module.copy_bytes(output, tvm_ffi.from_dlpack(value))
        return output

    def download(self, value, dtype):
        output = np.empty(value.shape, dtype=dtype)
        self.module.copy_bytes(tvm_ffi.from_dlpack(output), value)
        return output

    def synchronize(self):
        self.module.synchronize(0)

    def synchronize_compute(self):
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        self.checked(self.runtime.cudaStreamSynchronize(stream))

    def pinned_quiescent(self):
        stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
        status = self.runtime.cudaStreamQuery(stream)
        if status == 600:
            return False
        self.checked(status)
        return all(engine.quiescent() for engine in self.transfer_engines)

    def wait_pinned_safe(self):
        self.synchronize_compute()
        for engine in self.transfer_engines:
            for stream in engine.streams.values():
                self.checked(self.runtime.cudaStreamSynchronize(stream))

    def destroy_handles(self, handles, *, suppress=False):
        error = None
        for name, handle in handles:
            if handle:
                try:
                    self.checked(getattr(self.runtime, name)(handle))
                    if isinstance(handle, ctypes.c_void_p):
                        handle.value = None
                except Exception as failure:
                    error = error or failure
        if error is not None and not suppress:
            raise error
        return error

    @staticmethod
    def _pointer(value):
        return value if isinstance(value, ctypes.c_void_p) else ctypes.c_void_p(value)

    def _check_graph_health(self):
        if getattr(self, "graph_failed", False):
            raise RuntimeError("CUDA Graph device is unusable after cleanup failure")

    @staticmethod
    def _combine_errors(label, *errors):
        flattened = []
        for error in errors:
            if error is None:
                continue
            if isinstance(error, BaseExceptionGroup):
                flattened.extend(error.exceptions)
            else:
                flattened.append(error)
        errors = tuple(flattened)
        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        return ExceptionGroup(label, errors)

    def ensure_healthy(self):
        self._check_graph_health()

    def _retain_failed_graph_resources(self, resources):
        holder = getattr(self, "_failed_graph_resources", None)
        if holder is None:
            holder = ExitStack()
            self._failed_graph_resources = holder
        holder.enter_context(resources.pop_all())

    def capture(self, operation):
        self._check_graph_health()
        stream, graph, executable = (ctypes.c_void_p() for _ in range(3))
        events = []
        temporary = (("cudaGraphDestroy", graph), ("cudaStreamDestroy", stream))
        try:
            caller = self._pointer(tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0")))
            capturing = ctypes.c_int()
            self.checked(
                self.runtime.cudaStreamIsCapturing(caller, ctypes.byref(capturing))
            )
            if capturing.value:
                raise RuntimeError("caller stream is already capturing")
            self.checked(
                self.runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
            )
            producers = [caller]
            producers.extend(
                self._pointer(producer)
                for engine in self.transfer_engines
                for producer in engine.streams.values()
            )
            for producer in producers:
                event = ctypes.c_void_p()
                events.append(event)
                self.checked(
                    self.runtime.cudaEventCreateWithFlags(ctypes.byref(event), 2)
                )
                self.checked(self.runtime.cudaEventRecord(event, producer))
                self.checked(self.runtime.cudaStreamWaitEvent(stream, event, 0))
            self.destroy_handles(("cudaEventDestroy", event) for event in events)
            events.clear()
            with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
                self.checked(self.runtime.cudaStreamBeginCapture(stream, 1))
                try:
                    operation()
                except BaseException as failure:
                    try:
                        self.checked(
                            self.runtime.cudaStreamEndCapture(
                                stream, ctypes.byref(graph)
                            )
                        )
                    except BaseException as cleanup:
                        self.graph_failed = True
                        raise failure from cleanup
                    raise
                self.checked(
                    self.runtime.cudaStreamEndCapture(stream, ctypes.byref(graph))
                )
            self.checked(
                self.runtime.cudaGraphInstantiateWithFlags(
                    ctypes.byref(executable), graph, 0
                )
            )
            self.checked(self.runtime.cudaGraphUpload(executable, caller))
        except BaseException as failure:
            cleanup = (
                ("cudaGraphExecDestroy", executable),
                *temporary,
                *(("cudaEventDestroy", event) for event in events),
            )
            cleanup_error = self.destroy_handles(cleanup, suppress=True)
            retry_error = self.destroy_handles(cleanup, suppress=True)
            cause = self._combine_errors(
                "CUDA Graph cleanup failed",
                failure.__cause__,
                cleanup_error,
                retry_error,
            )
            if cause is not None:
                self.graph_failed = True
                raise failure from cause
            raise
        try:
            self.destroy_handles(temporary)
        except BaseException as failure:
            cleanup = (("cudaGraphExecDestroy", executable), *temporary)
            cleanup_error = self.destroy_handles(cleanup, suppress=True)
            retry_error = self.destroy_handles(cleanup, suppress=True)
            cause = self._combine_errors(
                "CUDA Graph cleanup failed",
                failure.__cause__,
                cleanup_error,
                retry_error,
            )
            if cause is not None:
                self.graph_failed = True
                raise failure from cause
            raise
        return executable

    def release_graph(self, executable):
        error = None
        try:
            self.synchronize()
        except BaseException as failure:
            error = failure
        cleanup = (("cudaGraphExecDestroy", executable),)
        try:
            self.destroy_handles(cleanup)
        except BaseException as failure:
            retry_error = self.destroy_handles(cleanup, suppress=True)
            destroy_error = self._combine_errors(
                "CUDA Graph executable cleanup failed",
                failure,
                retry_error,
            )
            if error is None:
                error = destroy_error
            else:
                error = self._combine_errors(
                    "CUDA Graph cleanup failed",
                    error,
                    destroy_error,
                )
        if error is not None:
            self.graph_failed = True
            raise error

    @contextmanager
    def graph(self, operation):
        owner = getattr(operation, "__self__", operation)
        acquire = getattr(owner, "graph_lease", None)
        with ExitStack() as resources:
            if acquire is not None:
                resources.enter_context(acquire())
            try:
                executable = self.capture(operation)
            except BaseException:
                if getattr(self, "graph_failed", False):
                    self._retain_failed_graph_resources(resources)
                raise
            active = True

            def replay():
                self._check_graph_health()
                if not active:
                    raise RuntimeError("CUDA Graph is closed")
                stream = tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))
                try:
                    self.checked(self.runtime.cudaGraphLaunch(executable, stream))
                except BaseException:
                    with suppress(BaseException):
                        self.synchronize()
                    raise

            try:
                yield replay
            except BaseException as failure:
                try:
                    self.release_graph(executable)
                except BaseException as cleanup:
                    self._retain_failed_graph_resources(resources)
                    raise failure from cleanup
                raise
            else:
                try:
                    self.release_graph(executable)
                except BaseException:
                    self._retain_failed_graph_resources(resources)
                    raise
            finally:
                active = False

    def checked(self, status):
        if status:
            message = self.runtime.cudaGetErrorString(status).decode("utf-8", "replace")
            raise RuntimeError(f"CUDA runtime error {status}: {message}")
