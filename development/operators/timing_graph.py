import ctypes
from contextlib import AbstractContextManager, ExitStack
from time import perf_counter

import tvm_ffi


def caller_stream():
    return tvm_ffi.get_raw_stream(tvm_ffi.device("cuda:0"))


class TimingGraph(AbstractContextManager):
    def __init__(self, prepared, events, repeats):
        self.device = prepared.device
        self.resources = ExitStack()
        self.executable = ctypes.c_void_p()
        runtime = self.device.runtime
        pointer = ctypes.c_void_p
        for name, arguments in {
            "cudaGraphCreate": [ctypes.POINTER(pointer), ctypes.c_uint],
            "cudaGraphAddEventRecordNode": [
                ctypes.POINTER(pointer),
                pointer,
                ctypes.POINTER(pointer),
                ctypes.c_size_t,
                pointer,
            ],
            "cudaGraphAddChildGraphNode": [
                ctypes.POINTER(pointer),
                pointer,
                ctypes.POINTER(pointer),
                ctypes.c_size_t,
                pointer,
            ],
            "cudaGraphGetNodes": [
                pointer,
                ctypes.POINTER(pointer),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "cudaGraphUpload": [pointer, pointer],
        }.items():
            function = getattr(runtime, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
        stream, child, outer = (pointer() for _ in range(3))
        handles = (
            ("cudaGraphDestroy", outer),
            ("cudaGraphDestroy", child),
            ("cudaStreamDestroy", stream),
        )
        started = perf_counter()
        prepared.reset()
        self.device.synchronize()
        try:
            self.resources.enter_context(prepared.graph_lease())
            self.device.checked(
                runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
            )
            with tvm_ffi.use_raw_stream(tvm_ffi.device("cuda:0"), stream.value):
                self.device.checked(runtime.cudaStreamBeginCapture(stream, 1))
                try:
                    prepared.run()
                except BaseException:
                    runtime.cudaStreamEndCapture(stream, ctypes.byref(child))
                    raise
                self.device.checked(
                    runtime.cudaStreamEndCapture(stream, ctypes.byref(child))
                )
            count = ctypes.c_size_t()
            self.device.checked(
                runtime.cudaGraphGetNodes(child, None, ctypes.byref(count))
            )
            if not count.value:
                raise ValueError("timing graph contains no production work")
            self.nodes = count.value
            self.device.checked(runtime.cudaGraphCreate(ctypes.byref(outer), 0))
            previous = pointer()
            self.device.checked(
                runtime.cudaGraphAddEventRecordNode(
                    ctypes.byref(previous), outer, None, 0, events.start
                )
            )
            for _ in range(repeats):
                node = pointer()
                self.device.checked(
                    runtime.cudaGraphAddChildGraphNode(
                        ctypes.byref(node), outer, ctypes.byref(previous), 1, child
                    )
                )
                previous = node
            node = pointer()
            self.device.checked(
                runtime.cudaGraphAddEventRecordNode(
                    ctypes.byref(node), outer, ctypes.byref(previous), 1, events.end
                )
            )
            self.device.checked(
                runtime.cudaGraphInstantiateWithFlags(
                    ctypes.byref(self.executable), outer, 0
                )
            )
            self.resources.callback(self.device.release_graph, self.executable)
            self.device.checked(
                runtime.cudaGraphUpload(self.executable, caller_stream())
            )
            self.device.synchronize()
        except BaseException:
            self.device.destroy_handles(handles, suppress=True)
            self.resources.close()
            raise
        try:
            self.device.destroy_handles(handles)
        except BaseException:
            self.resources.close()
            raise
        prepared.setup.setdefault("timing_graphs", []).append(
            {
                "seconds": perf_counter() - started,
                "repeats": repeats,
                "production_nodes": self.nodes,
            }
        )

    def __call__(self):
        if not self.executable:
            raise RuntimeError("timing graph is closed")
        self.device.checked(
            self.device.runtime.cudaGraphLaunch(self.executable, caller_stream())
        )

    def __exit__(self, *_):
        try:
            self.resources.close()
        finally:
            self.executable = None
