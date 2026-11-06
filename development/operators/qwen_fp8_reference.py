"""Independent CPU reference for the proposed Qwen KV format.

Proposed, not production-verified: E4M3FN has bias 7, subnormals, signed zero,
and a maximum finite magnitude of 448. Encoding uses nearest-even rounding
and finite saturation. Nonfinite inputs and reserved NaN bytes are errors.
Each 256-element BF16 vector has one dequantization scale. The scale is
RN32(absmax / 448), bounded below by the smallest positive FP32 subnormal;
an all-zero vector has scale 1. Division uses FP32 with gradual underflow.
Reconstruction multiplies the decoded value and stored scale in FP64.
The proposed attention path uses RN32 multiplication followed by RN-even
BF16 conversion. Both new rows and replay rows use these stored bytes and
scales. Do not send original BF16 new rows directly to attention. This
reference does not quantize Q/P or change FP32 recurrent/BF16 conv states.

独立 CPU 参考,格式提案尚未与生产实现核验。E4M3FN 使用最近偶数舍入、
有限饱和与有符号零。非有限输入及 NaN 字节报错。每个 256 维 BF16 向量
有一个 FP32 反量化 scale;零向量的 scale 为 1。FP64 用于重建。
拟议 attention 路径使用 FP32 乘法后转 BF16,新写与重放均从相同存储值
解码,不让原始 BF16 新行绕过量化。不量化 Q/P,不改 recurrent/conv。
"""

from dataclasses import dataclass

import numpy as np

PAGE_ROWS = 128
KV_HEADS = 4
VECTOR_WIDTH = 256
FORMAT_ID = "qwen-kv-e4m3fn-rne-fp32-scale-proposal-v1"


def _positive_values():
    codes = np.arange(127, dtype=np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float64)
    return np.where(
        exponent == 0,
        fraction * 2.0**-9,
        (1.0 + fraction / 8.0) * np.exp2(exponent - 7),
    )


_FINITE_VALUES = _positive_values()


def encode_e4m3fn(values):
    """Encode finite real values; saturate overflow and keep the zero sign."""
    values = np.asarray(values)
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("E4M3FN encoding needs finite floating-point values")
    magnitude = np.minimum(np.abs(values.astype(np.float64)), 448.0)
    upper = np.searchsorted(_FINITE_VALUES, magnitude).clip(0, 126)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _FINITE_VALUES[lower]
    upper_distance = _FINITE_VALUES[upper] - magnitude
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    return codes | (np.signbit(values).astype(np.uint8) << 7)


def decode_e4m3fn(codes):
    """Decode finite E4M3FN bytes to FP64; reject both reserved NaN codes."""
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or np.any((codes & 127) == 127):
        raise ValueError("expected finite E4M3FN uint8 bytes")
    magnitude = _FINITE_VALUES[codes & 127]
    return np.copysign(magnitude, np.where(codes & 128, -1.0, 1.0))


def quantize_bf16_vectors(bits):
    """Take raw BF16 uint16 [..., 256]; return uint8 bytes and FP32 scales."""
    bits = np.asarray(bits)
    if bits.dtype != np.uint16 or bits.ndim < 1 or bits.shape[-1] != VECTOR_WIDTH:
        raise ValueError("expected raw BF16 uint16 vectors with width 256")
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BF16 vectors must be finite")
    maximum = np.max(np.abs(values), axis=-1)
    minimum_scale = np.nextafter(np.float32(0), np.float32(1))
    scales = np.maximum(
        (maximum.astype(np.float64) / 448.0).astype(np.float32), minimum_scale
    )
    scales = np.where(maximum == 0, np.float32(1), scales).astype(np.float32)
    normalized = np.divide(values, scales[..., None], dtype=np.float32)
    return encode_e4m3fn(normalized), scales


def reconstruct_vectors(codes, scales):
    """Use stored FP32 scales; return FP64 without a BF16 output cast."""
    codes, scales = np.asarray(codes), np.asarray(scales)
    if (
        codes.ndim < 1
        or codes.shape[-1] != VECTOR_WIDTH
        or scales.shape != codes.shape[:-1]
    ):
        raise ValueError("payload and scale shapes do not describe 256-element vectors")
    if (
        scales.dtype != np.float32
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("scales must be positive finite FP32 values")
    return decode_e4m3fn(codes) * scales.astype(np.float64)[..., None]


def attention_bf16_vectors(codes, scales):
    """Return raw BF16 bits after RN32 dequantization and nearest-even cast.

    Use this for both new and cached K/V rows. It leaves Q/P unchanged.
    新写与缓存 K/V 行均用此解码语义,返回 BF16 原始位,不改变 Q/P。
    """
    with np.errstate(over="ignore"):
        product = reconstruct_vectors(codes, scales).astype(np.float32)
    if not np.isfinite(product).all():
        raise ValueError("dequantization exceeds finite FP32")
    words = product.view(np.uint32)
    rounded = ((words + np.uint32(0x7FFF) + ((words >> 16) & 1)) >> 16).astype(
        np.uint16
    )
    if np.any((rounded & 0x7F80) == 0x7F80):
        raise ValueError("dequantization exceeds finite BF16")
    return rounded


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass
class KVPages:
    """Logical CPU pages; payload [P,128,2,4,256], scales [P,128,2,4].

    K precedes V. Validity is per token. Updates commit only the valid input
    prefix. A failed update changes no payload, scale, or validity bits.
    Private pages can be copied and discarded without changing shared pages.
    """

    payload: np.ndarray
    scales: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        if self.payload.dtype != np.uint8 or self.payload.ndim != 5:
            raise ValueError("page payload must be a five-dimensional uint8 array")
        if self.payload.shape[1:] != (PAGE_ROWS, 2, KV_HEADS, VECTOR_WIDTH):
            raise ValueError("page payload shape must be [P,128,2,4,256]")
        if (
            self.scales.dtype != np.float32
            or self.scales.shape != self.payload.shape[:-1]
        ):
            raise ValueError("page scales must be FP32 with shape [P,128,2,4]")
        if self.valid.dtype != np.bool_ or self.valid.shape != self.payload.shape[:2]:
            raise ValueError("page validity must be bool with shape [P,128]")
        arrays = (self.payload, self.scales, self.valid)
        if any(not array.flags.writeable for array in arrays):
            raise ValueError("page storage must be writable")
        if any(
            np.shares_memory(a, b)
            for i, a in enumerate(arrays)
            for b in arrays[i + 1 :]
        ):
            raise ValueError("page buffers must not overlap")

    @classmethod
    def allocate(cls, pages):
        pages = _integer(pages, "pages")
        if pages < 1:
            raise ValueError("at least one physical page is required")
        shape = (pages, PAGE_ROWS, 2, KV_HEADS)
        return cls(
            np.zeros((*shape, VECTOR_WIDTH), dtype=np.uint8),
            np.ones(shape, dtype=np.float32),
            np.zeros(shape[:2], dtype=np.bool_),
        )

    def _page(self, page):
        page = _integer(page, "page")
        if not 0 <= page < len(self.payload):
            raise ValueError("physical page is out of range")
        return page

    def _locations(self, page_table, start, count):
        table = np.asarray(page_table)
        start, count = _integer(start, "start"), _integer(count, "count")
        if table.ndim != 1 or table.dtype.kind not in "iu":
            raise ValueError("page table must be a one-dimensional integer array")
        if np.any(table < 0) or np.any(table >= len(self.payload)):
            raise ValueError("page table contains an invalid physical page")
        if len(np.unique(table)) != len(table):
            raise ValueError("logical pages must not alias within one page table")
        if start < 0 or count < 0 or start + count > len(table) * PAGE_ROWS:
            raise ValueError("logical token range exceeds the page table")
        positions = np.arange(start, start + count, dtype=np.int64)
        return table[positions // PAGE_ROWS], positions % PAGE_ROWS

    def update(self, page_table, start, bf16_rows, valid_rows=None):
        """Write a valid prefix of [rows,2,4,256] raw BF16 input rows."""
        rows = np.asarray(bf16_rows)
        if rows.dtype != np.uint16 or rows.ndim != 4 or rows.shape[1:] != (2, 4, 256):
            raise ValueError("updates need raw BF16 rows with shape [rows,2,4,256]")
        count = len(rows) if valid_rows is None else _integer(valid_rows, "valid_rows")
        if not 0 <= count <= len(rows):
            raise ValueError("valid_rows exceeds the input")
        pages, offsets = self._locations(page_table, start, count)
        codes, scales = quantize_bf16_vectors(rows[:count])
        self.payload[pages, offsets] = codes
        self.scales[pages, offsets] = scales
        self.valid[pages, offsets] = True

    def read(self, page_table, start, count):
        """Read valid logical rows in order as [rows,2,4,256] FP64."""
        pages, offsets = self._locations(page_table, start, count)
        if not self.valid[pages, offsets].all():
            raise ValueError("logical read contains an unwritten or discarded token")
        return reconstruct_vectors(
            self.payload[pages, offsets], self.scales[pages, offsets]
        )

    def read_attention_bf16(self, page_table, start, count):
        """Read valid K/V rows with the same BF16 semantics as new rows."""
        pages, offsets = self._locations(page_table, start, count)
        if not self.valid[pages, offsets].all():
            raise ValueError("logical read contains an unwritten or discarded token")
        return attention_bf16_vectors(
            self.payload[pages, offsets], self.scales[pages, offsets]
        )

    def copy_page(self, source, destination):
        """Copy payload, scales, and valid rows before a private tail update."""
        source, destination = self._page(source), self._page(destination)
        self.payload[destination] = self.payload[source]
        self.scales[destination] = self.scales[source]
        self.valid[destination] = self.valid[source]

    def discard_page(self, page):
        """Invalidate a private page; stale bytes must not be read as valid."""
        self.valid[self._page(page)] = False
