from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist


def _cuda_allocation_bytes(size):
    if not size:
        return 0
    rounded = (size + 511) // 512 * 512
    if rounded <= 1 << 20:
        return 2 << 20
    if rounded < 10 << 20:
        return 20 << 20
    segment = 2 << 20
    return (rounded + segment - 1) // segment * segment


@dataclass(frozen=True)
class FixedDispatchCapacity:
    source_rows: int
    ranks: int
    width: int
    topk: int
    num_experts: int

    def __post_init__(self):
        dimensions = (
            self.source_rows,
            self.ranks,
            self.width,
            self.topk,
            self.num_experts,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("fixed dispatch dimensions must be positive integers")
        if self.ranks != 4 or self.num_experts % self.ranks:
            raise ValueError("DEP4 requires four equal expert partitions")
        if self.topk > min(6, self.num_experts) or self.receive_rows > 1024:
            raise ValueError("fixed dispatch exceeds the bounded expert batch")

    @property
    def receive_rows(self):
        return self.source_rows * self.ranks

    @property
    def routed_rows(self):
        return self.receive_rows * self.topk

    def buffers(self):
        rows, ranks, width, topk = self.source_rows, self.ranks, self.width, self.topk
        specifications = {
            "local_rows": ((rows, width), torch.bfloat16),
            "local_indices": ((rows, topk), torch.int32),
            "local_weights": ((rows, topk), torch.float32),
            "destinations": ((rows, topk), torch.int32),
            "matches": ((rows, topk, ranks), torch.bool),
            "present": ((ranks, rows), torch.bool),
            "prefix": ((ranks, rows), torch.int32),
            "positions": ((ranks, rows), torch.int64),
            "send_valid": ((ranks, rows), torch.bool),
            "receive_valid": ((ranks, rows), torch.bool),
            "send_counts": ((ranks,), torch.int32),
            "receive_counts": ((ranks,), torch.int32),
            "partial": ((ranks, rows, width), torch.float32),
            "returned": ((ranks, rows, width), torch.float32),
            "output": ((rows, width), torch.float32),
            "combine_scratch": ((rows, width), torch.float32),
            "expert_counts": ((self.num_experts // ranks + 1,), torch.int32),
            "expert_keys": ((self.routed_rows,), torch.int64),
            "route_order": ((self.routed_rows,), torch.int64),
            "route_tokens": ((self.routed_rows,), torch.int64),
            "route_ones": ((self.routed_rows,), torch.int32),
            "routed_rows": ((self.routed_rows, width), torch.bfloat16),
            "routed_weights": ((self.routed_rows,), torch.float32),
        }
        for direction in ("send", "receive"):
            specifications[f"{direction}_rows"] = ((ranks, rows, width), torch.bfloat16)
            specifications[f"{direction}_indices"] = ((ranks, rows, topk), torch.int32)
            specifications[f"{direction}_weights"] = (
                (ranks, rows, topk),
                torch.float32,
            )
        return specifications

    @property
    def tensor_bytes(self):
        arrays = sum(
            prod(shape) * dtype.itemsize for shape, dtype in self.buffers().values()
        )
        return arrays + 8 * (self.ranks + self.source_rows)

    @property
    def allocation_bytes(self):
        arrays = sum(
            _cuda_allocation_bytes(prod(shape) * dtype.itemsize)
            for shape, dtype in self.buffers().values()
        )
        return arrays + sum(
            _cuda_allocation_bytes(8 * count)
            for count in (self.ranks, self.source_rows)
        )


class DispatchStatus:
    def __init__(self, device):
        self.tensor = torch.zeros(1, dtype=torch.int32, device=device)

    def begin_execute(self):
        self.tensor.zero_()

    def mark(self, invalid):
        self.tensor.bitwise_or_(invalid.any().to(torch.int32).reshape(1))

    def finish_execute(self, *, group=None):
        failed = self.tensor.ne(0).to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        self.tensor.bitwise_or_(failed)
        if failed.item():
            raise RuntimeError("DEP4 exchange rejected Execute; no result is published")


class FixedDispatchBuffers:
    """Fixed pair slots and device counts, independent of the padded expert ABI."""

    def __init__(self, capacity, status, *, device, buffer_budget_bytes, group=None):
        if (
            type(buffer_budget_bytes) is not int
            or buffer_budget_bytes < capacity.tensor_bytes
        ):
            raise ValueError("fixed dispatch exceeds its fixed buffer budget")
        requested = torch.device(device)
        device = status.tensor.device
        if (
            status.tensor.shape != (1,)
            or status.tensor.dtype != torch.int32
            or device.type != requested.type
            or (requested.index is not None and device.index != requested.index)
        ):
            raise ValueError("fixed dispatch needs one device int32 status value")
        self.capacity, self.status, self.group = capacity, status, group
        self.storage = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in capacity.buffers().items()
        }
        self.slots = torch.arange(capacity.source_rows, device=device)
        self.ranks = torch.arange(capacity.ranks, device=device)
        self.received = tuple(
            self.storage[f"receive_{name}"].flatten(0, 1)
            for name in ("rows", "indices", "weights")
        )
        self.partial = self.storage["partial"].flatten(0, 1)
        self.receive_counts = self.storage["receive_counts"]
        self.receive_valid = self.storage["receive_valid"].flatten()
        self.expert_counts = self.storage["expert_counts"][:-1]
        self.sorted_routes = self.storage["route_order"]
        self.storage["route_ones"].fill_(1)
        self._source_rows = None

    def _exchange(self, output, value):
        dist.all_to_all_single(
            output.flatten(0, 1) if output.ndim > 1 else output,
            value.flatten(0, 1) if value.ndim > 1 else value,
            group=self.group,
        )

    def _validate(self, rows, indices, weights):
        capacity = self.capacity
        if (
            rows.ndim != 2
            or rows.dtype != torch.bfloat16
            or rows.size(1) != capacity.width
            or rows.size(0) > capacity.source_rows
        ):
            raise ValueError("dense rows differ from the fixed dispatch capacity")
        if (
            indices.shape != (rows.size(0), capacity.topk)
            or weights.shape != indices.shape
            or indices.dtype not in (torch.int32, torch.int64)
            or not weights.is_floating_point()
            or any(
                value.device != self.slots.device for value in (rows, indices, weights)
            )
        ):
            raise ValueError(
                "routing metadata differs from the fixed dispatch capacity"
            )
        if dist.get_world_size(self.group) != capacity.ranks:
            raise ValueError("fixed dispatch needs the prepared expert group")

    def dispatch(self, rows, indices, weights):
        self._validate(rows, indices, weights)
        if self._source_rows is not None:
            raise RuntimeError("combine the previous fixed dispatch before reuse")
        count, capacity, buffers = rows.size(0), self.capacity, self.storage
        for name, source, fill in (
            ("rows", rows, 0),
            ("indices", indices, -1),
            ("weights", weights, 0),
        ):
            target = buffers[f"local_{name}"]
            target[:count].copy_(source)
            target[count:].fill_(fill)
        self.status.mark((indices < 0) | (indices >= capacity.num_experts))
        torch.div(
            buffers["local_indices"],
            capacity.num_experts // capacity.ranks,
            rounding_mode="floor",
            out=buffers["destinations"],
        )
        torch.eq(buffers["destinations"][..., None], self.ranks, out=buffers["matches"])
        torch.any(buffers["matches"], dim=1, out=buffers["present"].T)
        torch.cumsum(
            buffers["present"], dim=1, dtype=torch.int32, out=buffers["prefix"]
        )
        buffers["send_counts"].copy_(buffers["prefix"][:, -1])
        buffers["positions"].copy_(
            torch.where(
                buffers["present"],
                buffers["prefix"] - 1,
                buffers["send_counts"][:, None] + self.slots - buffers["prefix"],
            )
        )
        torch.lt(self.slots, buffers["send_counts"][:, None], out=buffers["send_valid"])
        for name, fill in (("rows", 0), ("indices", -1), ("weights", 0)):
            target, source = buffers[f"send_{name}"], buffers[f"local_{name}"]
            positions = buffers["positions"][..., None].expand_as(target)
            target.scatter_(1, positions, source[None].expand_as(target))
            target.masked_fill_(~buffers["send_valid"][..., None], fill)
        self._exchange(buffers["receive_counts"], buffers["send_counts"])
        self.status.mark(
            (self.receive_counts < 0) | (self.receive_counts > capacity.source_rows)
        )
        torch.lt(
            self.slots,
            self.receive_counts.clamp(0, capacity.source_rows)[:, None],
            out=buffers["receive_valid"],
        )
        for name in ("rows", "indices", "weights"):
            self._exchange(buffers[f"receive_{name}"], buffers[f"send_{name}"])
        self._source_rows = count
        return self.received

    def local_routes(self, expert_rank):
        if self._source_rows is None:
            raise RuntimeError("fixed local routes require a dispatch")
        capacity, buffers = self.capacity, self.storage
        if type(expert_rank) is not int or not 0 <= expert_rank < capacity.ranks:
            raise ValueError("expert rank is outside the fixed dispatch group")
        groups = capacity.num_experts // capacity.ranks
        first = expert_rank * groups
        indices = self.received[1]
        owned = (
            self.receive_valid[:, None]
            & (indices >= first)
            & (indices < first + groups)
        )
        buffers["expert_keys"].copy_(
            torch.where(owned, indices - first, groups).flatten()
        )
        buffers["expert_counts"].zero_()
        buffers["expert_counts"].scatter_add_(
            0, buffers["expert_keys"], buffers["route_ones"]
        )
        self.status.mark(self.expert_counts > capacity.receive_rows)
        torch.argsort(buffers["expert_keys"], stable=True, out=self.sorted_routes)
        torch.div(
            self.sorted_routes,
            capacity.topk,
            rounding_mode="floor",
            out=buffers["route_tokens"],
        )
        torch.index_select(
            self.received[0], 0, buffers["route_tokens"], out=buffers["routed_rows"]
        )
        torch.index_select(
            self.received[2].flatten(),
            0,
            self.sorted_routes,
            out=buffers["routed_weights"],
        )
        return (
            buffers["routed_rows"],
            buffers["routed_weights"],
            self.expert_counts,
            self.sorted_routes,
        )

    def combine(self, partial):
        if self._source_rows is None:
            raise RuntimeError("fixed combine requires a dispatch")
        capacity, buffers = self.capacity, self.storage
        if (
            partial.shape != (capacity.receive_rows, capacity.width)
            or partial.dtype != torch.float32
            or partial.device != self.slots.device
            or not partial.is_contiguous()
        ):
            raise ValueError("fixed combine needs one FP32 row per prepared pair slot")
        self._exchange(buffers["returned"], partial.view_as(buffers["returned"]))
        buffers["returned"].masked_fill_(~buffers["send_valid"][..., None], 0)
        buffers["output"].zero_()
        for destination in range(capacity.ranks):
            torch.gather(
                buffers["returned"][destination],
                0,
                buffers["positions"][destination, :, None].expand_as(buffers["output"]),
                out=buffers["combine_scratch"],
            )
            buffers["output"].add_(buffers["combine_scratch"])
        result = buffers["output"][: self._source_rows]
        self._source_rows = None
        return result


class PaddedExpertBuffers:
    def __init__(self, capacity, intermediate, device, *, budget_bytes):
        from .kernel_native import _projection_module

        self.capacity = capacity
        self.intermediate = intermediate
        self.module = _projection_module()
        groups = capacity.num_experts // capacity.ranks
        routed = capacity.routed_rows
        self.max_rows = max(capacity.receive_rows, (routed + groups - 1) // groups)
        scratch = max(
            self.module.grouped_workspace_size(
                routed, groups, self.max_rows, intermediate, capacity.width
            ),
            self.module.grouped_workspace_size(
                routed, groups, self.max_rows, capacity.width, intermediate
            ),
        )
        specifications = {
            "gate": ((routed, intermediate), torch.bfloat16),
            "up": ((routed, intermediate), torch.bfloat16),
            "down": ((routed, capacity.width), torch.bfloat16),
            "scratch": ((scratch,), torch.uint8),
            "route_rows": ((capacity.receive_rows, capacity.topk), torch.int32),
        }
        fixed_bytes = sum(
            _cuda_allocation_bytes(prod(shape) * dtype.itemsize)
            for shape, dtype in specifications.values()
        )
        temporaries = (
            (capacity.width, 1),
            (capacity.width // 32, 1),
            (intermediate, 2),
            (intermediate, 1),
            (intermediate // 32, 1),
        )
        self.temporary_bytes = (4 << 20) + sum(
            _cuda_allocation_bytes(routed * width * itemsize)
            for width, itemsize in temporaries
        )
        self.reserved_bytes = (
            capacity.allocation_bytes + fixed_bytes + self.temporary_bytes
        )
        if self.reserved_bytes > budget_bytes:
            raise ValueError("DEP4 padded expert buffers exceed their workspace budget")
        self.storage = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in specifications.items()
        }

    def run(self, moe, fixed, expert_rank):
        from .kernel import (
            act_quant,
            grouped_fp4_gemm_padded,
            ordered_expert_sum_padded,
            swiglu,
        )
        from .kernel_weights import _static_pointer_arrays

        state = moe._routed_state
        if state.weights is None:
            moe.prepare_routed_experts()
        weights = state.weights
        if weights is False:
            raise RuntimeError("DEP4 padded execution needs prepared FP4 experts")
        if state.pointers is None:
            state.pointers = tuple(
                _static_pointer_arrays(projection, fixed.slots.device)
                for projection in (weights.gate, weights.up, weights.down)
            )
        rows, route_weights, counts, routes = fixed.local_routes(expert_rank)
        encoded, scales = act_quant(rows, 32, "ue8m0", torch.float8_e8m0fnu)
        for index, (name, projection) in enumerate(
            (("gate", weights.gate), ("up", weights.up))
        ):
            grouped_fp4_gemm_padded(
                encoded,
                scales,
                counts,
                projection,
                pointers=state.pointers[index][:2],
                status=fixed.status.tensor,
                max_rows=self.max_rows,
                output=self.storage[name],
                scratch=self.storage["scratch"],
            )
        product = swiglu(
            self.storage["gate"],
            self.storage["up"],
            route_weights,
            moe.experts[moe.experts_start_idx].swiglu_limit,
        )
        product_encoded, product_scales = act_quant(
            product, 32, "ue8m0", torch.float8_e8m0fnu
        )
        grouped_fp4_gemm_padded(
            product_encoded,
            product_scales,
            counts,
            weights.down,
            pointers=state.pointers[2][:2],
            status=fixed.status.tensor,
            max_rows=self.max_rows,
            output=self.storage["down"],
            scratch=self.storage["scratch"],
        )
        ordered_expert_sum_padded(
            self.storage["down"],
            routes,
            counts,
            fixed.partial,
            fixed.capacity.topk,
            fixed.status.tensor,
            route_rows=self.storage["route_rows"],
        )
        return fixed.partial
