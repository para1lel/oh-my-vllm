"""Eager expert dispatch for request-owned DeepSeek rows."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DispatchLayout:
    tokens: torch.Tensor
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]


def destination_pairs(indices, experts_per_rank, world_size):
    """Return destination-major distinct (token, destination) pairs on the device."""
    destinations = indices // experts_per_rank
    ranks = torch.arange(world_size, device=indices.device)
    present = (destinations.unsqueeze(-1) == ranks).any(dim=1)
    destination, token = present.T.nonzero(as_tuple=True)
    counts = present.sum(dim=0, dtype=torch.int64)
    return token, destination, counts


def _exchange(value, send_counts, receive_counts, group):
    output = torch.empty(
        (sum(receive_counts), *value.shape[1:]),
        device=value.device,
        dtype=value.dtype,
    )
    dist.all_to_all_single(
        output,
        value.contiguous(),
        output_split_sizes=list(receive_counts),
        input_split_sizes=list(send_counts),
        group=group,
    )
    return output


def dispatch(rows, indices, weights, *, num_experts, group=None):
    """Send one BF16 row per destination, with its complete routing metadata."""
    size = dist.get_world_size(group)
    if num_experts % size:
        raise ValueError("expert count must divide the expert group")
    if rows.dtype != torch.bfloat16 or rows.ndim != 2:
        raise ValueError("DEP4 dispatch requires a BF16 row matrix")
    if indices.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("routing IDs and weights must have the same matrix shape")
    if indices.size(0) != rows.size(0):
        raise ValueError("routing rows must match dense rows")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("routing IDs must use an integer tensor")
    indices = indices.to(torch.int32)
    tokens, _destinations, send = destination_pairs(indices, num_experts // size, size)
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    send_counts = tuple(send.tolist())
    receive_counts = tuple(receive.tolist())
    layout = DispatchLayout(tokens, send_counts, receive_counts)
    received = tuple(
        _exchange(value.index_select(0, tokens), send_counts, receive_counts, group)
        for value in (rows, indices, weights.float())
    )
    return received, layout


def combine(local_sum, layout, *, num_rows, group=None):
    """Return one FP32 local expert sum per pair and add in destination order."""
    if local_sum.ndim != 2 or local_sum.dtype != torch.float32:
        raise ValueError("DEP4 combine requires FP32 local sums")
    if local_sum.size(0) != sum(layout.receive_counts):
        raise ValueError("local expert sums differ from dispatch rows")
    returned = _exchange(local_sum, layout.receive_counts, layout.send_counts, group)
    output = torch.zeros(
        (num_rows, local_sum.size(1)), dtype=torch.float32, device=local_sum.device
    )
    first = 0
    for count in layout.send_counts:
        last = first + count
        # Each destination has at most one row for a source token.
        output.index_add_(0, layout.tokens[first:last], returned[first:last])
        first = last
    return output


def local_expert_sum(moe, rows, indices, weights):
    """Bound expert scratch after the single pair dispatch for this layer."""
    output = torch.empty_like(rows, dtype=torch.float32)
    for first in range(0, rows.size(0), 1024):
        last = min(first + 1024, rows.size(0))
        output[first:last] = _local_expert_sum(
            moe, rows[first:last], indices[first:last], weights[first:last]
        )
    return output


def _local_expert_sum(moe, rows, indices, weights):
    """Compute only this rank's experts, rounding each expert before FP32 sum."""
    output = torch.zeros_like(rows, dtype=torch.float32)
    owned = (indices >= moe.experts_start_idx) & (indices < moe.experts_end_idx)
    tokens, routes = owned.nonzero(as_tuple=True)
    expert_ids = indices[tokens, routes]
    order = torch.argsort(expert_ids, stable=True)
    expert_ids = expert_ids[order]
    tokens = tokens[order]
    routed_weights = weights[tokens, routes[order]]
    routed_rows = rows.index_select(0, tokens)
    counts_tensor = torch.bincount(expert_ids, minlength=moe.n_routed_experts)
    counts = counts_tensor.tolist()
    grouped = moe._grouped_routed(
        routed_rows, tokens, routed_weights, counts, counts_tensor
    )
    if grouped is not None:
        from .kernel import ordered_expert_sum

        _grouped_tokens, result = grouped
        sorted_routes = tokens * indices.size(1) + routes[order]
        ordered_expert_sum(result, sorted_routes, output, indices.size(1))
        return output
    first = 0
    for expert_id in range(moe.experts_start_idx, moe.experts_end_idx):
        last = first + counts[expert_id]
        if last > first:
            result = moe.experts[expert_id](
                routed_rows[first:last], routed_weights[first:last].unsqueeze(-1)
            )
            output[tokens[first:last]] += result.float()
        first = last
    return output


class ExpertExchange:
    def __init__(self, *, group=None):
        self.group = group

    def __call__(self, moe, rows, routing):
        indices, weights = routing[1], routing[0]
        (remote_rows, remote_indices, remote_weights), layout = dispatch(
            rows,
            indices,
            weights,
            num_experts=moe.n_routed_experts,
            group=self.group,
        )
        partial = local_expert_sum(moe, remote_rows, remote_indices, remote_weights)
        return combine(partial, layout, num_rows=rows.size(0), group=self.group)
