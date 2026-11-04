from contextlib import contextmanager, nullcontext

import torch
import tvm_ffi

_LIBRARY = torch.library.Library("oh_my_vllm", "DEF")
_LIBRARY.define(
    "native_empty(SymInt[] shape, ScalarType dtype, Device device) -> Tensor"
)
_LIBRARY.define(
    "native_scratch(SymInt[] shape, ScalarType dtype, Device device) -> Tensor"
)
_LIBRARY.define("native_view(Tensor source, SymInt[] shape, int offset=0) -> Tensor")
_LIBRARY.define("native_copy(Tensor input, Tensor(a!) output) -> Tensor(a!)")
_LIBRARY.define("native_zero(Tensor(a!) output) -> Tensor(a!)")
_LIBRARY.define("native_upload(Tensor input, Tensor(a!) output) -> Tensor(a!)")
_LIBRARY.define("native_download(Tensor input) -> Tensor")
_LIBRARY.define(
    "native_copy_batch(Tensor[] inputs, Tensor[] outputs, int stream) -> ()"
)

_TORCH_TO_NATIVE_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e4m3fnuz: "float8_e4m3fnuz",
    torch.float8_e5m2: "float8_e5m2",
    torch.float8_e5m2fnuz: "float8_e5m2fnuz",
    torch.float8_e8m0fnu: "float8_e8m0fnu",
    torch.float4_e2m1fn_x2: "float4_e2m1fn_x2",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
}


def native_tensor(value, *, require_contiguous=True):
    if isinstance(value, tvm_ffi.Tensor):
        return value
    if not isinstance(value, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported tensor device: {value.device}")
    if require_contiguous and not value.is_contiguous():
        raise ValueError("native tensor conversion requires a contiguous tensor")
    return tvm_ffi.from_dlpack(value, require_contiguous=require_contiguous)


def native_shape(dimensions):
    return tvm_ffi.Shape(tuple(dimensions))


def native_tensors(values):
    return tuple(native_tensor(value) for value in values)


def torch_tensor(value):
    try:
        return torch.from_dlpack(value)
    except (AttributeError, TypeError) as error:
        raise TypeError("expected a TVMFFI DLPack producer") from error


def current_stream(device):
    device = torch.device(device)
    if device.type == "cpu":
        return 0
    if device.type != "cuda":
        raise ValueError(f"unsupported stream device: {device}")
    return torch.cuda.current_stream(device).cuda_stream


@contextmanager
def use_stream(device, stream):
    device = torch.device(device)
    if device.type == "cpu":
        yield
        return
    if device.type != "cuda":
        raise ValueError(f"unsupported stream device: {device}")
    raw_stream = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
    with tvm_ffi.use_raw_stream(tvm_ffi.device(str(device)), raw_stream):
        yield


def _device_module():
    from .device import FUNCTIONS
    from .native import load

    return load("device", FUNCTIONS)[0]


def _native_dtype(dtype):
    try:
        return _TORCH_TO_NATIVE_DTYPE[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported native dtype: {dtype}") from error


def _native_empty(shape, dtype, device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"native allocation requires a CUDA device: {device}")
    native = _device_module().allocate(
        native_shape(shape), _native_dtype(dtype), device.index or 0
    )
    return torch_tensor(native)


def _native_scratch(shape, dtype, device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"native allocation requires a CUDA device: {device}")
    with use_stream(device, torch.cuda.current_stream(device)):
        native = _device_module().allocate_scratch(
            native_shape(shape), _native_dtype(dtype), device.index or 0
        )
        return torch_tensor(native)


def _native_view(source, shape, offset=0):
    native = _device_module().view(native_tensor(source), native_shape(shape), offset)
    return torch_tensor(native)


def _current_stream_scope(*tensors):
    for tensor in tensors:
        if tensor.device.type == "cuda":
            return use_stream(tensor.device, torch.cuda.current_stream(tensor.device))
    return nullcontext()


def _native_copy(input, output):
    with _current_stream_scope(input, output):
        _device_module().copy_bytes(native_tensor(output), native_tensor(input))
    return output


def _native_zero(output):
    with _current_stream_scope(output):
        _device_module().zero(native_tensor(output))
    return output


def _native_upload(input, output):
    if input.device.type != "cpu":
        raise ValueError("native upload input must be a CPU tensor")
    if output.device.type != "cuda":
        raise ValueError("native upload output must be a CUDA tensor")
    # Only pinned upload objects call this operator. A copy kernel reads the
    # host arena on the caller stream, so metadata does not queue behind bulk
    # H2D copies. Ordinary CPU tensors use native_copy.
    with _current_stream_scope(input, output):
        _device_module().upload_bytes(native_tensor(output), native_tensor(input))
    return output


def _native_download(input):
    if input.device.type != "cuda":
        raise ValueError("native download input must be a CUDA tensor")
    output = torch.empty(input.shape, dtype=input.dtype, device="cpu")
    return _native_copy(input, output)


def _native_copy_batch(inputs, outputs, stream):
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError("native copy batch requires matching nonempty tensors")
    _device_module().copy_batch(
        tuple(native_tensor(output) for output in outputs),
        tuple(native_tensor(input) for input in inputs),
        stream,
    )


_IMPLEMENTATIONS = torch.library.Library(
    "oh_my_vllm", "IMPL", "CompositeExplicitAutograd"
)
_IMPLEMENTATIONS.impl("native_empty", _native_empty)
_IMPLEMENTATIONS.impl("native_scratch", _native_scratch)
_IMPLEMENTATIONS.impl("native_view", _native_view)
_IMPLEMENTATIONS.impl("native_copy", _native_copy)
_IMPLEMENTATIONS.impl("native_zero", _native_zero)
_IMPLEMENTATIONS.impl("native_upload", _native_upload)
_IMPLEMENTATIONS.impl("native_download", _native_download)
_IMPLEMENTATIONS.impl("native_copy_batch", _native_copy_batch)
