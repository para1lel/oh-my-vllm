"""Bounded checkpoint preparation in the source weight formats."""

import math
from contextlib import suppress

import numpy as np

from .checkpoint import (
    DTYPE_BYTES,
    MAX_TILE_BYTES,
    RAW_DTYPES,
    EncodedReader,
    read_layout,
)
from .placement import LAYOUT_SHA256, _size, _uint, rank_plan

_DEVICE_DTYPES = {
    "BF16": "bfloat16",
    "F32": "float32",
    "F8_E4M3": "uint8",
    "F8_E8M0": "uint8",
    "I8": "uint8",
}
_FAILED_LOADS = []

_FP8_VALUES = np.array(
    [
        float("nan")
        if code & 127 == 127
        else (-1 if code & 128 else 1)
        * (
            (code & 7) / 512
            if code & 120 == 0
            else (1 + (code & 7) / 8) * 2 ** (((code >> 3) & 15) - 7)
        )
        for code in range(256)
    ],
    dtype=np.float32,
)


def _array(value, dtype):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(dtype)
        or value.ndim not in (1, 2)
        or not value.flags.c_contiguous
        or not value.size
        or value.size * 4 > MAX_TILE_BYTES
    ):
        raise ValueError("weight tile has an invalid type, shape, or byte size")


def promote_bf16(raw):
    _array(raw, np.uint16)
    value = np.left_shift(raw, 16, dtype=np.uint32).view(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("weight promotion requires finite BF16 values")
    return value


def promote_fp8(raw, scales, *, row_offset=0):
    """Return BF16 bits; scales start at the tile's first dense32 row block."""
    _array(raw, np.uint8)
    _array(scales, np.uint8)
    if (
        raw.ndim != 2
        or type(row_offset) is not int
        or not 0 <= row_offset < 32
        or scales.shape
        != ((row_offset + raw.shape[0] + 31) // 32, (raw.shape[1] + 31) // 32)
    ):
        raise ValueError("weight scale blocks differ from the tile")
    if np.any((raw & 127) == 127) or np.any(scales == 255):
        raise ValueError("weight promotion requires finite FP8 values and scales")
    value = _FP8_VALUES[raw]
    with np.errstate(over="ignore", under="ignore"):
        for row in range(raw.shape[0]):
            powers = scales[(row_offset + row) // 32].astype(np.int16) - 127
            np.ldexp(
                value[row],
                np.repeat(powers, 32)[: raw.shape[1]],
                out=value[row],
            )
    if not np.isfinite(value).all():
        raise ValueError("weight promotion overflows FP32")
    bits, result = value.view(np.uint32), np.empty(raw.shape, np.uint16)
    # The retained low bit selects ties-to-even before discarding 16 bits.
    np.right_shift(bits, 16, out=result, casting="unsafe")
    np.bitwise_and(result, 1, out=result)
    np.add(bits, np.uint32(0x7FFF), out=bits)
    np.add(bits, result, out=bits)
    np.right_shift(bits, 16, out=result, casting="unsafe")
    if np.any((result & 0x7F80) == 0x7F80):
        raise ValueError("weight promotion overflows BF16")
    return result


def _shape(spec):
    return tuple(
        last - first for first, last in zip(spec.start, spec.stop, strict=True)
    )


def _scale(plan, name):
    spec = plan.selections[name]
    scale_name = name.removesuffix(".weight") + ".scale"
    scale = plan.selections.get(scale_name)
    factor = 2 if spec.dtype == "I8" else 1
    if len(spec.shape) != 2 or scale is None or scale.dtype != "F8_E8M0":
        raise ValueError("encoded weight requires its scale matrix")
    first = spec.start[0] if factor == 2 else spec.start[0] // 32
    last = spec.stop[0] if factor == 2 else (spec.stop[0] + 31) // 32
    if (
        spec.start[1] * factor % 32
        or spec.stop[1] * factor % 32
        or scale.start != (first, spec.start[1] * factor // 32)
        or scale.stop != (last, spec.stop[1] * factor // 32)
    ):
        raise ValueError("encoded weight scale coordinates differ")
    return scale_name, scale


def _inventory(plan, limit, tile_bytes):
    from ...operators.projection.plan import make_plan, scale_bytes

    growth = scale_bound = resident = 0
    paired = set()
    for name, spec in plan.selections.items():
        if spec.dtype == "F8_E8M0":
            continue
        shape = _shape(spec)
        promotion = plan.promotions.get(name)
        if promotion and (spec.dtype, promotion) not in (
            ("BF16", "F32"),
            ("F8_E4M3", "BF16"),
        ):
            raise ValueError("unsupported startup weight promotion")
        row_elements = math.prod(shape[1:])
        if row_elements * DTYPE_BYTES[spec.dtype] > tile_bytes // 2 or (
            promotion and row_elements * 16 > tile_bytes - tile_bytes // 2
        ):
            raise ValueError("weight tile budget cannot hold one row")
        resident += _size(*shape, DTYPE_BYTES[promotion or spec.dtype])
        if spec.dtype in ("I8", "F8_E4M3"):
            scale_name, scale = _scale(plan, name)
            paired.add(scale_name)
            compact_scale = _size(*_shape(scale))
            if promotion:
                if compact_scale * 4 > MAX_TILE_BYTES:
                    raise ValueError("startup scale buffer exceeds its byte limit")
                scale_bound = max(scale_bound, compact_scale)
            else:
                rows, columns = shape
                bits = 4 if spec.dtype == "I8" else 8
                native = make_plan(
                    rows=0, outputs=rows, width=columns * (8 // bits), bits=bits
                )
                packed = _size(rows, native.padded_width * bits // 8)
                growth += (
                    packed
                    - _size(*shape)
                    + scale_bytes(rows, native.width)
                    - compact_scale
                )
                resident += compact_scale
    if paired != {
        name for name, spec in plan.selections.items() if spec.dtype == "F8_E8M0"
    }:
        raise ValueError("unpaired startup weight scale")
    if resident != plan.resident_bytes:
        raise ValueError("rank resident byte accounting differs")
    if not 0 <= growth <= 4 << 30:
        raise ValueError("native weight layout exceeds its 4 GiB cap")
    if _uint(resident + growth) > limit:
        raise ValueError("rank weight budget cannot hold persistent storage")
    return growth, scale_bound


class RankWeights:
    """Own loaded tensors; limit covers requested GPU bytes, including pack overlap.

    CUDA allocator/context overhead and unrelated device storage need a caller
    reserve. host_bound_bytes bounds bulk buffers, not Python/FFI metadata: reader
    source buffer plus tile_bytes for raw/conversion work plus one wo_a scale array.
    """

    def __init__(self, device, plan, limit, host_bound_bytes):
        self.device, self.plan, self.limit = device, plan, limit
        self.tensors, self.packed, self.identity = {}, {}, {}
        self.persistent_bytes = self.peak_bytes = 0
        self.host_bound_bytes = host_bound_bytes
        self._storage, self._pending, self._buffer = {}, (), None
        self.closed = self._retained = False

    def _allocate(self, shape, dtype):
        size = _size(*shape, {"uint8": 1, "bfloat16": 2, "float32": 4}[dtype])
        if self.persistent_bytes + size > self.limit:
            raise ValueError("rank weight budget exceeded during startup overlap")
        value = self.device.empty(shape, dtype)
        self._storage[id(value)] = value, size
        self.persistent_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.persistent_bytes)
        return value

    def _drop(self, value):
        _, size = self._storage.pop(id(value))
        self.persistent_bytes -= size

    def _synchronize(self):
        try:
            self.device.synchronize_compute()
        except BaseException:
            if not self._retained:
                _FAILED_LOADS.append(self)
                self._retained = True
            raise
        self._pending = ()

    def close(self):
        if self.closed or self._retained:
            return
        self._synchronize()
        self.tensors.clear()
        self.packed.clear()
        self._storage.clear()
        self._buffer = None
        self.persistent_bytes = 0
        self.closed = True

    def __enter__(self):
        if self.closed or self._retained:
            raise RuntimeError("rank weights are closed or failed")
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self.close()
        except BaseException:
            if error_type is None:
                raise


def _upload(owner, reader, name, tile_bytes):
    import tvm_ffi

    spec = owner.plan.selections[name]
    shape, promotion = _shape(spec), owner.plan.promotions.get(name)
    columns = math.prod(shape[1:])
    rows = min(shape[0], len(owner._buffer) // (columns * DTYPE_BYTES[spec.dtype]))
    if promotion:
        rows = min(rows, (tile_bytes - len(owner._buffer)) // (16 * columns))
    scales = output = values = source = destination = None
    try:
        if promotion == "BF16":
            scale_name, scale = _scale(owner.plan, name)
            scales = np.empty(_shape(scale), np.uint8)
            if reader.read_into(scale_name, 0, scales) != scales.shape[0]:
                raise ValueError("incomplete startup weight scale")
        output = owner._allocate(shape, _DEVICE_DTYPES[promotion or spec.dtype])
        raw = np.frombuffer(
            owner._buffer, RAW_DTYPES[spec.dtype], count=rows * columns
        ).reshape((rows, *shape[1:]))
        for first in range(0, shape[0], rows):
            count = min(rows, shape[0] - first)
            if reader.read_into(name, first, raw[:count]) != count:
                raise ValueError("incomplete startup weight tile")
            values = raw[:count]
            if promotion == "F32":
                values = promote_bf16(values)
            elif promotion == "BF16":
                global_row = spec.start[0] + first
                block = global_row // 32 - scale.start[0]
                end = (global_row % 32 + count + 31) // 32
                values = promote_fp8(
                    values, scales[block : block + end], row_offset=global_row % 32
                )
            source = tvm_ffi.from_dlpack(values)
            destination = owner.device.view(output, values.shape, first * columns)
            owner._pending = raw, values, source, destination
            owner.device.module.copy_bytes(destination, source)
            owner._synchronize()
        return output
    finally:
        output = values = source = destination = None


def _load_tensor(owner, reader, name, tile_bytes):
    from ...operators.projection.api import Weights
    from ...operators.projection.workspace import prepare_weights

    value = scales = packed = None
    try:
        value = _upload(owner, reader, name, tile_bytes)
        spec = owner.plan.selections[name]
        if spec.dtype in ("I8", "F8_E4M3") and name not in owner.plan.promotions:
            scale_name, _ = _scale(owner.plan, name)
            scales = _upload(owner, reader, scale_name, tile_bytes)
            packed = prepare_weights(
                owner.device,
                Weights(value, scales),
                bits=4 if spec.dtype == "I8" else 8,
                allocate=owner._allocate,
            )
            owner.packed[name] = packed
            if packed.encoded is not value:
                owner._drop(value)
            owner._drop(scales)
        else:
            owner.tensors[name] = value
    finally:
        value = scales = packed = None


def _load_plan(root, verified_files, device, plan, *, limit, tile_bytes=16 << 20):
    from ...operators.projection.workspace import require_eager

    if type(limit) is not int or not 0 <= limit < 1 << 64:
        raise ValueError("rank weight budget must be an unsigned byte count")
    if type(tile_bytes) is not int or not 2 <= tile_bytes <= MAX_TILE_BYTES:
        raise ValueError("startup tile exceeds its byte budget")
    growth, scale_bound = _inventory(plan, limit, tile_bytes)
    require_eager(device)
    owner = RankWeights(device, plan, limit, MAX_TILE_BYTES + tile_bytes + scale_bound)
    try:
        owner._buffer = bytearray(tile_bytes // 2)
        with EncodedReader(root, verified_files, plan.selections) as reader:
            for name, spec in sorted(plan.selections.items()):
                if spec.dtype != "F8_E8M0":
                    _load_tensor(owner, reader, name, tile_bytes)
            if owner.persistent_bytes != plan.resident_bytes + growth:
                raise ValueError(
                    "native weight storage differs from startup accounting"
                )
            owner.identity = reader.finish()
        owner._buffer = None
        return owner
    except BaseException:
        with suppress(BaseException):
            owner.close()
        raise


def load_rank(
    root, verified_files, device, *, strategy, rank, limit, tile_bytes=16 << 20
):
    """Validate the fixed model before allocating GPU storage; load only owned slices.

    File identities are explicit and trusted by the caller. Payloads are verified
    by EncodedReader before upload and finish is required before publication.
    """
    model, tensors = read_layout(root, verified_files, layout_sha256=LAYOUT_SHA256)
    plan = rank_plan(model, tensors, strategy, rank)
    return _load_plan(
        root, verified_files, device, plan, limit=limit, tile_bytes=tile_bytes
    )
