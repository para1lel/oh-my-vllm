"""Reusable Eager prefill projections with the released collective sequence."""

import os

import torch
import torch.distributed as dist

from . import kernel_native as native

COLLECTIVE_ROWS = 1024


def projection_rows(value=None, *, strategy=None, partitions=1):
    default = "8192" if partitions == 4 and strategy in ("TP4", "TEP4") else "1024"
    value = (
        os.environ.get("OH_MY_VLLM_DEEPSEEK_MOE_PREFILL_ROWS", default)
        if value is None
        else value
    )
    try:
        rows = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("MoE prefill rows must be 1024, 4096, or 8192") from error
    if str(rows) != str(value) or rows not in (1024, 4096, 8192):
        raise ValueError("MoE prefill rows must be 1024, 4096, or 8192")
    return rows


class PrefillWorkspace:
    """One model owns this storage; its target layers use it on the same stream."""

    def __init__(self, rows):
        self.rows = projection_rows(rows)
        self.binding = None
        self.buffers = {}

    def clear(self):
        self.binding = None
        self.buffers = {}

    def reserve(self, moe, device, partitions):
        prepared = moe._routed_state.weights
        inter = prepared.gate[0][0].size(0)
        width = prepared.down[0][0].size(1) * 2
        outputs = prepared.down[0][0].size(0)
        binding = (
            device,
            moe.dim,
            inter,
            width,
            outputs,
            moe.n_activated_experts,
            moe.n_routed_experts,
            moe.tensor_parallel,
            len(prepared.gate),
            partitions,
        )
        if binding == self.binding:
            return
        capacity = self.rows * moe.n_activated_experts
        small = COLLECTIVE_ROWS * moe.n_activated_experts
        max_width = max(moe.dim, width)
        groups = len(prepared.gate)
        projection = native._projection_module()
        scratch_bytes = max(
            projection.grouped_workspace_size(capacity, groups, capacity, n, k)
            for n, k in ((inter, moe.dim), (outputs, width))
        )
        buffers = {}

        def empty(name, shape, dtype):
            buffers[name] = torch.empty(shape, dtype=dtype, device=device)

        for name in ("ids", "order", "tokens", "chunk_order", "inverse", "positions"):
            empty(name, (capacity,), torch.int64)
        buffers["route_ids"] = torch.arange(capacity, dtype=torch.int64, device=device)
        buffers["row_ids"] = torch.arange(capacity, dtype=torch.int32, device=device)
        empty("weights", (self.rows, moe.n_activated_experts), torch.float32)
        empty("indices", (self.rows, moe.n_activated_experts), torch.int64)
        empty("sorted_weights", (capacity,), torch.float32)
        empty("local_weights", (capacity,), torch.float32)
        empty("counts", (moe.n_routed_experts,), torch.int32)
        empty("ones", (capacity,), torch.int32)
        buffers["ones"].fill_(1)
        empty("route_rows", (self.rows, moe.n_activated_experts), torch.int32)
        empty("values", (capacity * max(moe.dim, outputs),), torch.bfloat16)
        for name in ("gate", "up", "product"):
            empty(name, (capacity, inter), torch.bfloat16)
        empty("encoded", (capacity * max_width,), torch.uint8)
        empty("scales", (capacity * (max_width // 32),), torch.uint8)
        empty("scratch", (scratch_bytes,), torch.uint8)
        empty("sum", (self.rows, outputs), torch.float32)
        empty("output", (self.rows, moe.dim), torch.bfloat16)
        if moe.tensor_parallel:
            empty("gather_input", (small, inter), torch.bfloat16)
            empty("gather_shards", (partitions, small, inter), torch.bfloat16)
            empty("full_product", (small, width), torch.bfloat16)
            empty("sum_shards", (partitions, COLLECTIVE_ROWS, outputs), torch.float32)
            empty("full_sum", (COLLECTIVE_ROWS, moe.dim), torch.float32)
        self.buffers = buffers
        self.binding = binding
        self.partitions = partitions

    def routing(self, moe, x, mask, routing):
        rows = x.size(0)
        weights = self.buffers["weights"][:rows]
        indices = self.buffers["indices"][:rows]
        if routing is not None:
            weights.copy_(routing[0])
            indices.copy_(routing[1])
        else:
            for first in range(0, rows, COLLECTIVE_ROWS):
                last = min(first + COLLECTIVE_ROWS, rows)
                chunk = moe.gate(
                    x[first:last], None if mask is None else mask[first:last]
                )
                weights[first:last].copy_(chunk[0])
                indices[first:last].copy_(chunk[1])
        return weights, indices

    def _quantize(self, values):
        rows, width = values.shape
        encoded = self.buffers["encoded"][: rows * width].view(rows, width)
        scales = self.buffers["scales"][: rows * (width // 32)].view(rows, width // 32)
        native._transform_module().quantize(
            values, encoded, scales, 8, 32, 1, True, False
        )
        return encoded, scales

    def _project(self, values, counts, pointers, output, *, quantized=None):
        encoded, scales = self._quantize(values) if quantized is None else quantized
        native._projection_module().grouped_project(
            encoded,
            scales,
            *pointers[:2],
            counts,
            output,
            self.buffers["scratch"],
            max(1, values.size(0)),
        )

    def _sum(self, values, routes, output):
        route_rows = self.buffers["route_rows"][: output.size(0)]
        route_rows.fill_(-1)
        route_rows.view(-1).scatter_(
            0, routes, self.buffers["row_ids"][: values.size(0)]
        )
        native._routing_module().expert_sum(values, route_rows, output)

    def _gather_product(self, product, positions):
        count = positions.numel()
        inputs = self.buffers["gather_input"][:count]
        torch.index_select(product, 0, positions, out=inputs)
        if self.partitions == 1:
            return inputs
        inter = product.size(1)
        shards = (
            self.buffers["gather_shards"]
            .view(-1)[: self.partitions * count * inter]
            .view(self.partitions, count, inter)
        )
        dist.all_gather_into_tensor(shards.flatten(0, 1), inputs)
        output = self.buffers["full_product"][:count]
        for rank in range(self.partitions):
            output[:, rank * inter : (rank + 1) * inter].copy_(shards[rank])
        return output

    def _gather_sum(self, values):
        if self.partitions == 1:
            return values
        rows, width = values.shape
        shards = (
            self.buffers["sum_shards"]
            .view(-1)[: self.partitions * rows * width]
            .view(self.partitions, rows, width)
        )
        dist.all_gather_into_tensor(shards.flatten(0, 1), values)
        output = self.buffers["full_sum"][:rows]
        for rank in range(self.partitions):
            output[:, rank * width : (rank + 1) * width].copy_(shards[rank])
        return output

    def forward_rows(self, moe, x, weights, indices, shared_output):
        buffers = self.buffers
        rows, topk = indices.shape
        capacity = rows * topk
        ids, order = buffers["ids"][:capacity], buffers["order"][:capacity]
        torch.sort(indices.flatten(), out=(ids, order))
        tokens = buffers["tokens"][:capacity]
        torch.div(order, topk, rounding_mode="floor", out=tokens)
        sorted_weights = buffers["sorted_weights"][:capacity]
        torch.index_select(weights.flatten(), 0, order, out=sorted_weights)
        values = buffers["values"][: capacity * x.size(1)].view(capacity, x.size(1))
        torch.index_select(x, 0, tokens, out=values)
        counts = buffers["counts"].zero_()
        counts.scatter_add_(0, ids, buffers["ones"][:capacity])
        host_counts = counts.tolist() if moe.expert_parallel else None
        first = 0 if host_counts is None else sum(host_counts[: moe.experts_start_idx])
        count = (
            capacity
            if host_counts is None
            else sum(host_counts[moe.experts_start_idx : moe.experts_end_idx])
        )
        local_counts = counts[moe.experts_start_idx : moe.experts_end_idx]
        pointers = moe._routed_state.pointers
        gate, up, product = (
            buffers[name][:count] for name in ("gate", "up", "product")
        )
        local_values = values[first : first + count]
        if count:
            self._project_gate_up(
                moe,
                local_values,
                local_counts,
                pointers,
                gate,
                up,
                product,
                sorted_weights,
                first,
                count,
            )
        output = buffers["output"][:rows]
        if not moe.tensor_parallel:
            y = buffers["sum"][:rows]
            if count:
                routed = buffers["values"][: count * moe.dim].view(count, moe.dim)
                self._project(product, local_counts, pointers[2], routed)
                self._sum(routed, order[first : first + count], y)
            else:
                y.zero_()
            for start in range(0, rows, COLLECTIVE_ROWS):
                end = min(start + COLLECTIVE_ROWS, rows)
                if moe.expert_parallel and self.partitions > 1:
                    dist.all_reduce(y[start:end])
                y[start:end].add_(shared_output[start:end])
                output[start:end].copy_(y[start:end])
            return output
        return self._tensor_parallel_sum(
            moe, product, indices, counts, order, pointers, output, shared_output
        )

    def _tensor_parallel_sum(
        self, moe, product, indices, counts, order, pointers, output, shared_output
    ):
        buffers = self.buffers
        rows, topk = indices.shape
        capacity = rows * topk
        inverse = buffers["inverse"][:capacity]
        inverse.scatter_(0, order, buffers["route_ids"][:capacity])
        for start in range(0, rows, COLLECTIVE_ROWS):
            end = min(start + COLLECTIVE_ROWS, rows)
            begin_route, end_route = start * topk, end * topk
            chunk_ids = buffers["ids"][begin_route:end_route]
            chunk_order = buffers["chunk_order"][begin_route:end_route]
            torch.sort(indices[start:end].flatten(), out=(chunk_ids, chunk_order))
            counts.zero_().scatter_add_(
                0, chunk_ids, buffers["ones"][: end_route - begin_route]
            )
            chunk_order.add_(begin_route)
            positions = buffers["positions"][: end_route - begin_route]
            torch.index_select(inverse, 0, chunk_order, out=positions)
            full_product = self._gather_product(product, positions)
            width = buffers["sum"].size(1)
            routed = buffers["values"][: full_product.size(0) * width].view(-1, width)
            self._project(full_product, counts, pointers[2], routed)
            y = buffers["sum"][: end - start]
            chunk_order.sub_(begin_route)
            self._sum(routed, chunk_order, y)
            full_sum = self._gather_sum(y)
            full_sum.add_(shared_output[start:end])
            output[start:end].copy_(full_sum)
        return output

    def _project_gate_up(
        self,
        moe,
        local_values,
        local_counts,
        pointers,
        gate,
        up,
        product,
        sorted_weights,
        first,
        count,
    ):
        buffers = self.buffers
        quantized = self._quantize(local_values)
        self._project(
            local_values, local_counts, pointers[0], gate, quantized=quantized
        )
        self._project(local_values, local_counts, pointers[1], up, quantized=quantized)
        local_weights = buffers["local_weights"][:count]
        local_weights.copy_(sorted_weights[first : first + count])
        native._activation_module().swiglu(
            gate,
            up,
            local_weights,
            product,
            int(moe.experts[moe.experts_start_idx].swiglu_limit),
            True,
        )
