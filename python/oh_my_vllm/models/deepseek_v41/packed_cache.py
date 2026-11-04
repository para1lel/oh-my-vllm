"""Physical FP4 history planes and views over the Rust block table."""

from dataclasses import dataclass
from math import ceil

import torch

from .quantization import fp4_dequantize


@torch.inference_mode(False)
def _allocate(shape, dtype, device, lazy):
    if lazy and device.type == "cuda":
        from ...operators.runtime.paged import allocate_paged

        return allocate_paged(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


@dataclass
class PackedPage:
    codes: torch.Tensor
    scales: torch.Tensor

    @property
    def shape(self):
        return self.codes.shape

    @property
    def device(self):
        return self.codes.device

    @property
    def dtype(self):
        return torch.uint8

    def __getitem__(self, key):
        return PackedPage(self.codes[key], self.scales[key])

    def copy_(self, source):
        self.codes.copy_(source.codes)
        self.scales.copy_(source.scales)

    def numel(self):
        return self.codes.numel() + self.scales.numel()

    def element_size(self):
        return 1

    def to(self, *, device, dtype=None, copy=True):
        if dtype not in (None, torch.uint8):
            raise ValueError("packed pages must retain their original bytes")
        return PackedPage(
            self.codes.to(device=device, copy=copy),
            self.scales.to(device=device, copy=copy),
        )

    def detach(self):
        return PackedPage(self.codes.detach(), self.scales.detach())

    def pin_memory(self):
        return PackedPage(self.codes.pin_memory(), self.scales.pin_memory())


class IndexWorkspace:
    """One fixed address for complete-T BF16 index operands across sources."""

    def __init__(self, slots, length, dim, device, lazy, dtype=torch.bfloat16):
        self.values = _allocate((1, slots * length, dim), dtype, device, lazy)
        self.dim = dim
        self.capacity = slots * length

    def reserve(self, rows):
        if not 0 <= rows <= self.capacity:
            raise ValueError("packed index workspace capacity exceeded")
        owner = getattr(self.values, "_paged_storage", None)
        if owner is not None:
            owner.reserve(0, rows)

    def view(self, batch, width):
        return self.values.view(-1)[: batch * width * self.dim].view(
            batch, width, self.dim
        )

    def close(self):
        owner = getattr(self.values, "_paged_storage", None)
        if owner is not None:
            owner.release(0)


class PageTable:
    @torch.inference_mode(False)
    def __init__(self, slots, length, block_tokens, capacity, device):
        if capacity < 2:
            raise ValueError("packed cache requires a null page and a writable page")
        self.block_tokens = block_tokens
        self.capacity = capacity
        self.values = torch.zeros(
            (slots, ceil(length / block_tokens)), dtype=torch.int64, device=device
        )
        self.rows = tuple(self.values.unbind(0))
        self.generation = 0
        self.leases = 0
        self.closed = False
        self.active = {}
        self.upload_device = self.uploads = None
        if device.type == "cuda":
            from ...operators.runtime.device import Device

            self.upload_device = Device()
            self.uploads = self.upload_device.uploads()

    def bind(self, slot, blocks):
        if self.closed or len(blocks) > self.values.size(1):
            raise ValueError("packed cache block table is unavailable or too long")
        if any(
            type(block) is not int or not 0 <= block < self.capacity for block in blocks
        ):
            raise ValueError("packed cache block is outside the physical pool")
        if type(slot) is not int or not 0 <= slot < self.values.size(0):
            raise ValueError("packed cache slot is outside the active table")
        source = torch.zeros(self.values.size(1), dtype=torch.int64, device="cpu")
        if blocks:
            source[: len(blocks)] = torch.tensor(
                blocks, dtype=torch.int64, device="cpu"
            )
        destination = self.rows[slot]
        if self.uploads is None:
            destination.copy_(source)
        else:
            self.uploads.copy(destination, source)
        if blocks:
            self.active[slot] = tuple(blocks)
        else:
            self.active.pop(slot, None)

    def acquire(self):
        if self.closed:
            raise RuntimeError("packed cache pool is closed")
        self.leases += 1
        return PageLease(self, self.generation)

    def close(self):
        if self.closed:
            return
        if self.leases:
            raise RuntimeError("close captured graphs before releasing the packed pool")
        if self.uploads is not None:
            self.uploads.close()
        if self.upload_device is not None:
            self.upload_device.close()
        self.closed = True
        self.generation += 1


@dataclass
class PageLease:
    table: PageTable
    generation: int
    released: bool = False

    def validate(self):
        if (
            self.released
            or self.table.closed
            or self.generation != self.table.generation
        ):
            raise RuntimeError("DeepSeek graph packed cache bindings have changed")

    def close(self):
        if not self.released:
            self.table.leases -= 1
            self.released = True


class PackedHistory:
    """A source has one encoded plane and one scale plane, shared by active slots."""

    def __init__(self, args, width, ratio, group_size, scale_dtype):
        self.shape = (args.max_batch_size, args.max_seq_len // ratio, width)
        self.ratio = ratio
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.anchor = torch.empty(0)
        self.lazy = getattr(args, "lazy_cache", False)
        self.table = None
        self.codes = self.scales = None
        self.workspace = None
        self.initialized = set()
        self.automatic = False

    @property
    def dtype(self):
        return self.anchor.dtype

    @property
    def device(self):
        return self.anchor.device

    def size(self, dimension):
        return self.shape[dimension]

    def apply(self, function):
        if self.table is not None:
            raise RuntimeError("bound packed cache does not support tensor conversion")
        self.anchor = function(self.anchor)

    def bind(self, table, workspace=None):
        if self.table is not None:
            raise RuntimeError("packed history is already bound")
        if table.block_tokens % self.ratio:
            raise ValueError("packed page size must be divisible by the source ratio")
        self.table = table
        self.page_rows = table.block_tokens // self.ratio
        rows = table.capacity * self.page_rows
        self.codes = _allocate(
            (1, rows, self.shape[-1] // 2), torch.uint8, self.device, self.lazy
        )
        self.scales = _allocate(
            (1, rows, self.shape[-1] // self.group_size),
            torch.uint8,
            self.device,
            self.lazy,
        )
        self.workspace = workspace
        self.reserve_page(0)

    @torch.inference_mode()
    def _standalone(self, slots, end):
        if self.table is None:
            tokens = max(self.ratio, self.shape[1] * self.ratio)
            blocks = ceil(tokens / 128)
            table = PageTable(
                self.shape[0], tokens, 128, 1 + self.shape[0] * blocks, self.device
            )
            self.bind(table)
            self.automatic = True
        if self.automatic:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return
            blocks = self.table.values.size(1)
            for slot in slots.tolist():
                count = ceil(end * self.ratio / self.table.block_tokens)
                ids = list(range(1 + slot * blocks, 1 + slot * blocks + count))
                for block in ids:
                    self.reserve_page(block)
                self.table.bind(slot, ids)

    def reserve_page(self, block):
        if block in self.initialized:
            return
        if not 0 <= block < self.table.capacity:
            raise ValueError("packed page is outside the physical pool")
        for value in (self.codes, self.scales):
            owner = getattr(value, "_paged_storage", None)
            if owner is not None:
                owner.reserve(0, (block + 1) * self.page_rows)
        page = self.page(block)
        page.codes.zero_()
        page.scales.fill_(1)
        self.initialized.add(block)

    def page(self, block):
        first = block * self.page_rows
        return PackedPage(
            self.codes[0, first : first + self.page_rows],
            self.scales[0, first : first + self.page_rows],
        )

    def page_bytes(self, block_tokens):
        return (
            block_tokens
            // self.ratio
            * (self.shape[-1] // 2 + self.shape[-1] // self.group_size)
        )

    def discard_after(self, block, accepted_tokens):
        page = self.page(block)
        count = accepted_tokens // self.ratio
        page.codes[count:].zero_()
        page.scales[count:].fill_(1)

    def _rows(self, slots, positions):
        table = self.table.values[
            slots.reshape(-1, *((1,) * (positions.ndim - 1))),
            positions // self.page_rows,
        ]
        return table * self.page_rows + positions.remainder(self.page_rows)

    def write(self, slots, first, encoded, scales):
        length = encoded.size(1)
        if isinstance(first, int):
            self._standalone(slots, first + length)
            positions = torch.arange(first, first + length, device=self.device).expand(
                slots.numel(), length
            )
        else:
            if self.table is None or self.automatic:
                capturing = (
                    self.device.type == "cuda"
                    and torch.cuda.is_current_stream_capturing()
                )
                if not capturing:
                    self._standalone(slots, int(first.max()) + length)
                elif self.table is None:
                    raise RuntimeError("prepare the packed pool before Graph capture")
            positions = first.reshape(-1, 1) + torch.arange(length, device=self.device)
            positions = positions.expand(slots.numel(), length)
        rows = self._rows(slots, positions)
        self.codes[0, rows] = encoded.view(torch.uint8)
        self.scales[0, rows] = scales.view(torch.uint8)

    def view(self, slots, width):
        if not 0 <= width <= self.shape[1]:
            raise ValueError("packed history view exceeds the context capacity")
        self._standalone(slots, width)
        return PackedView(self, slots, width)

    def __getitem__(self, key):
        slot, columns = key if isinstance(key, tuple) else (key, slice(None))
        scalar = isinstance(slot, int)
        slots = torch.as_tensor(
            [slot] if scalar else slot, dtype=torch.int64, device=self.device
        ).reshape(-1)
        if not isinstance(columns, slice) or columns.step not in (None, 1):
            raise ValueError("packed inspection requires a contiguous history range")
        first, stop, _ = columns.indices(self.shape[1])
        values = self.view(slots, stop).materialize(shared=False)[:, first:stop]
        return values[0] if scalar else values

    def close(self):
        for value in (self.codes, self.scales):
            owner = getattr(value, "_paged_storage", None)
            if owner is not None:
                owner.release(0)


class TokenHistory(PackedHistory):
    def __init__(self, args):
        super().__init__(args, 1, 1, 1, None)
        self.shape = self.shape[:2]
        self.values = None

    @property
    def dtype(self):
        return torch.int64

    def bind(self, table, workspace=None):
        if self.table is not None:
            raise RuntimeError("token history is already bound")
        self.table = table
        self.page_rows = table.block_tokens
        self.values = _allocate(
            (1, table.capacity * self.page_rows), torch.int64, self.device, self.lazy
        )
        self.reserve_page(0)

    def reserve_page(self, block):
        if block in self.initialized:
            return
        if not 0 <= block < self.table.capacity:
            raise ValueError("token page is outside the physical pool")
        owner = getattr(self.values, "_paged_storage", None)
        if owner is not None:
            owner.reserve(0, (block + 1) * self.page_rows)
        self.page(block).zero_()
        self.initialized.add(block)

    def page(self, block):
        first = block * self.page_rows
        return self.values[0, first : first + self.page_rows]

    def page_bytes(self, block_tokens):
        return block_tokens * 8

    def discard_after(self, block, accepted_tokens):
        self.page(block)[accepted_tokens:].zero_()

    def _selection(self, key):
        slot, columns = key if isinstance(key, tuple) else (key, slice(None))
        scalar = isinstance(slot, int)
        slots = torch.as_tensor(
            [slot] if scalar else slot, dtype=torch.int64, device=self.device
        ).reshape(-1)
        if isinstance(columns, slice):
            first, stop, step = columns.indices(self.shape[1])
            if step != 1:
                raise ValueError("token history range must be contiguous")
            self._standalone(slots, stop)
            positions = torch.arange(first, stop, device=self.device).expand(
                slots.numel(), -1
            )
        else:
            positions = torch.as_tensor(columns, dtype=torch.int64, device=self.device)
            positions = positions.reshape(slots.numel(), -1)
            if self.table is None:
                self._standalone(slots, int(positions.max()) + 1)
        return self._rows(slots, positions), scalar

    def __getitem__(self, key):
        rows, scalar = self._selection(key)
        values = self.values[0, rows]
        return values[0] if scalar else values

    def __setitem__(self, key, values):
        rows, _scalar = self._selection(key)
        self.values[0, rows] = values

    def close(self):
        owner = getattr(self.values, "_paged_storage", None)
        if owner is not None:
            owner.release(0)


@dataclass
class PackedView:
    history: PackedHistory
    slots: torch.Tensor
    width: int

    @property
    def shape(self):
        return (self.slots.numel(), self.width, self.history.shape[-1])

    @property
    def device(self):
        return self.history.device

    def size(self, dimension):
        return self.shape[dimension]

    def materialize(self, *, shared=True):
        history = self.history
        batch, width, dim = self.shape
        workspace = history.workspace if shared else None
        if workspace is None:
            output = torch.empty(self.shape, dtype=history.dtype, device=self.device)
        else:
            output = workspace.view(batch, width)
        if output.is_cuda and output.dtype == torch.bfloat16:
            from ...operators.runtime.native import load
            from ...operators.transform.api import FUNCTIONS

            status = torch.zeros(1, dtype=torch.int32, device=self.device)
            load("transform", FUNCTIONS)[0].gather_fp4(
                history.codes.view(-1, dim // 2),
                history.scales.view(-1, dim // history.group_size),
                history.table.values,
                self.slots,
                output,
                status,
                history.page_rows,
                history.group_size,
                2 if history.scale_dtype == torch.float8_e4m3fn else 1,
            )
            from .kernel import _sparse_status_accumulator

            _sparse_status_accumulator(self.device).logical_or_(status.any())
        else:
            positions = torch.arange(width, device=self.device).expand(batch, width)
            rows = history._rows(self.slots, positions)
            output.copy_(
                fp4_dequantize(
                    history.codes[0, rows],
                    history.scales[0, rows].view(history.scale_dtype),
                    history.group_size,
                    dtype=history.dtype,
                )
            )
        return output

    def native_rows(self, logical_rows):
        return self.history._rows(self.slots, logical_rows)

    def with_width(self, width):
        if not 0 <= width <= self.width:
            raise ValueError("packed source view is shorter than its consumer")
        return PackedView(self.history, self.slots, width)


def write_history(history, slots, first, encoded, scales, group_size):
    if isinstance(history, PackedHistory):
        history.write(slots, first, encoded, scales)
        return
    values = fp4_dequantize(encoded, scales, group_size, dtype=history.dtype)
    if isinstance(first, int):
        history[slots, first : first + values.size(1)] = values
    else:
        positions = first.reshape(-1, 1) + torch.arange(
            values.size(1), device=values.device
        )
        history[slots[:, None], positions] = values


def history_view(history, slots, width):
    if isinstance(history, PackedHistory):
        return history.view(slots, width)
    return history[:, :width][slots]
