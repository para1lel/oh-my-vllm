from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from . import model_context as context
from .model_args import ModelArgs
from .model_expert import Expert, Gate, _gather_channels
from .routed_projection import project_routed
from .routed_state import (
    _register_routed_owners,
    _RoutedModuleList,
    _RoutedState,
    _unregister_routed_state,
)


@dataclass(frozen=True)
class _RoutedWeights:
    gate: tuple
    up: tuple
    down: tuple


def _route_counts(flat_indices: torch.Tensor, experts: int) -> torch.Tensor:
    """Count each expert's routed rows without reading a CUDA scalar on the host."""
    counts = torch.zeros(experts, dtype=torch.int32, device=flat_indices.device)
    ones = torch.ones_like(flat_indices, dtype=torch.int32)
    return counts.scatter_add_(0, flat_indices.to(torch.int64), ones)


class MoE(nn.Module):
    """
    Top-k routed experts plus one shared expert every token goes through. Experts are
    split
    across ranks, so `self.experts` is None for those another rank owns.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.data_parallel = args.parallel_strategy == "DEP4"
        self.expert_world_size = (
            context.expert_world_size if self.data_parallel else context.world_size
        )
        assert n_routed_experts % self.expert_world_size == 0, (
            "Number of experts must be divisible by world size "
            f"(world_size={context.world_size})"
        )
        self.n_routed_experts = n_routed_experts
        self.expert_parallel = args.parallel_strategy in ("TEP4", "DEP4")
        self.tensor_parallel = args.parallel_strategy == "TP4"
        self.n_local_experts = (
            n_routed_experts // self.expert_world_size
            if self.expert_parallel
            else n_routed_experts
        )
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = (
            context.rank * self.n_local_experts if self.expert_parallel else 0
        )
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self.experts = _RoutedModuleList(
            [
                Expert(
                    args.dim,
                    args.moe_inter_dim,
                    dtype=expert_dtype,
                    swiglu_limit=args.swiglu_limit,
                    tensor_parallel=self.tensor_parallel,
                )
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim,
            args.moe_inter_dim,
            swiglu_limit=args.swiglu_limit,
        )
        self._routed_state = _RoutedState()
        self._prefill_workspace = None

    def __setattr__(self, name, value):
        if name == "experts":
            state = getattr(self, "_routed_state", None)
            if state is not None:
                state.clear_weights()
        super().__setattr__(name, value)

    def _apply(self, fn, recurse=True):
        self._routed_state.clear_weights()
        if self._prefill_workspace is not None:
            self._prefill_workspace.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._routed_state.clear_weights()
        return super()._load_from_state_dict(*args, **kwargs)

    def prepare_routed_experts(self):
        """
        Prepare immutable expert bindings. Weight or expert rebinds invalidate them.
        """
        if getattr(self, "_routed_state", None) is None:
            self._routed_state = _RoutedState()
        state = self._routed_state
        _unregister_routed_state(state)
        state.clear_weights()
        state.weights = False
        experts = tuple(
            self.experts[index]
            for index in range(self.experts_start_idx, self.experts_end_idx)
        )
        targets = [getattr(self, "experts", None), *experts]
        if not experts or any(
            expert is None
            or not all(hasattr(expert, name) for name in ("w1", "w2", "w3"))
            for expert in experts
        ):
            state.registered = _register_routed_owners(state, targets)
            return
        projections = tuple(
            tuple(getattr(expert, name) for expert in experts)
            for name in ("w1", "w3", "w2")
        )
        projection_layers = []
        projection_ids = set()
        for projection in projections:
            for layer in projection:
                if id(layer) not in projection_ids:
                    projection_layers.append(layer)
                    projection_ids.add(id(layer))
        state.registered = _register_routed_owners(
            state, [*targets, *projection_layers]
        )
        if any(
            layer.weight.dtype != torch.float4_e2m1fn_x2 or layer.scale is None
            for projection in projections
            for layer in projection
        ):
            return
        state.weights = _RoutedWeights(
            *(
                tuple((layer.weight, layer.scale) for layer in projection)
                for projection in projections
            ),
        )

    def _grouped_routed(
        self,
        sorted_x: torch.Tensor,
        sorted_tokens: torch.Tensor,
        sorted_weights: torch.Tensor,
        counts: list[int] | None,
        count_tensor: torch.Tensor,
    ):
        state = getattr(self, "_routed_state", None)
        if state is None or state.weights is None:
            self.prepare_routed_experts()
            state = self._routed_state
        prepared = state.weights
        if prepared is False:
            return None
        if counts is None:
            if not (sorted_x.is_cuda and prepared.gate[0][0].is_cuda):
                return None
            # This rank owns every expert, so the local rows are the whole input.
            local_x, local_tokens = sorted_x, sorted_tokens
            local_weights = sorted_weights.unsqueeze(-1)
            local_counts = None
            local_count_tensor = count_tensor
        else:
            first = sum(counts[: self.experts_start_idx])
            local_counts = counts[self.experts_start_idx : self.experts_end_idx]
            last = first + sum(local_counts)
            if first == last:
                return None
            local_x = sorted_x[first:last]
            local_tokens = sorted_tokens[first:last]
            local_weights = sorted_weights[first:last].unsqueeze(-1)
            local_count_tensor = count_tensor[
                self.experts_start_idx : self.experts_end_idx
            ].to(dtype=torch.int32)
        return project_routed(
            self,
            local_x,
            local_tokens,
            local_weights,
            local_counts,
            local_count_tensor,
            state,
            prepared,
        )

    def _prepare_prefill_workspace(self, rows, enabled):
        workspace = self._prefill_workspace
        if (
            not enabled
            or workspace is None
            or workspace.rows == context.MAX_MOE_ROWS
            or rows.size(0) <= context.MAX_MOE_ROWS
            or self.data_parallel
            or not rows.is_cuda
            or rows.dtype != torch.bfloat16
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        state = self._routed_state
        if state.weights is None:
            self.prepare_routed_experts()
        prepared = state.weights
        if prepared is False:
            return None
        expert = self.experts[self.experts_start_idx]
        if prepared.gate[0][0].size(0) not in (576, 2304) or expert.swiglu_limit != 10:
            return None
        if state.pointers is None:
            from .kernel_weights import _static_pointer_arrays

            state.pointers = tuple(
                _static_pointer_arrays(projection, rows.device)
                for projection in (prepared.gate, prepared.up, prepared.down)
            )
        workspace.reserve(self, rows.device, context.world_size)
        return workspace

    def _forward_rows(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        *,
        routing=None,
        shared_output=None,
        workspace=None,
    ) -> torch.Tensor:
        if routing is None:
            routing = self.gate(x, None if image_mask is None else image_mask.flatten())
        weights, indices = routing
        trace = getattr(self, "_routing_trace", None)
        if trace is not None:
            trace.record(self.layer_id, indices, getattr(self, "_trace_first", 0))
        if workspace is not None:
            return workspace.forward_rows(self, x, weights, indices, shared_output)
        output_width = (
            x.size(-1) // context.world_size if self.tensor_parallel else x.size(-1)
        )
        y = torch.zeros((x.size(0), output_width), dtype=torch.float32, device=x.device)
        flat_indices = indices.flatten()
        order = torch.argsort(flat_indices)
        sorted_tokens = (
            torch.arange(x.size(0), device=x.device)
            .view(-1, 1)
            .expand_as(indices)
            .flatten()[order]
        )
        sorted_weights = weights.flatten()[order]
        sorted_x = x.index_select(0, sorted_tokens)
        count_tensor = _route_counts(flat_indices, self.n_routed_experts)
        # The grouped grid covers groups and column tiles only, so the device
        # counts path has no routed-row bound and never reads a host scalar.
        counts = (
            None if x.is_cuda and not self.expert_parallel else count_tensor.tolist()
        )
        grouped = self._grouped_routed(
            sorted_x,
            sorted_tokens,
            sorted_weights,
            counts,
            count_tensor,
        )
        if grouped is None:
            if counts is None:
                counts = count_tensor.tolist()
            first = sum(counts[: self.experts_start_idx])
            for i in range(self.experts_start_idx, self.experts_end_idx):
                last = first + counts[i]
                if first == last:
                    first = last
                    continue
                expert = self.experts[i]
                idx = sorted_tokens[first:last]
                expert_weights = sorted_weights[first:last].unsqueeze(-1)
                y[idx] += expert(sorted_x[first:last], expert_weights)
                first = last
        else:
            from .kernel import ordered_expert_sum

            _local_tokens, routed = grouped
            first = 0 if counts is None else sum(counts[: self.experts_start_idx])
            ordered_expert_sum(
                routed, order[first : first + routed.size(0)], y, indices.size(1)
            )
        if self.tensor_parallel:
            y = _gather_channels(y)
        elif self.expert_parallel and context.world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x) if shared_output is None else shared_output
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        *,
        logical_rows=0,
        blas_workspace=None,
        plan=None,
        valid_rows=None,
        dispatch_combine=None,
        prefill_eager=False,
    ) -> torch.Tensor:
        shape = x.size()
        self._trace_first = 0
        rows = x.view(-1, self.dim)
        hook = (
            dispatch_combine
            if dispatch_combine is not None
            else getattr(self, "dispatch_combine", None)
        )
        if self.data_parallel and hook is None:
            return self._forward_owned(
                rows, image_mask, logical_rows, blas_workspace
            ).view(shape)
        mask = None if image_mask is None else image_mask.flatten()
        routing = (
            self.gate(
                rows, mask, logical_rows=logical_rows, blas_workspace=blas_workspace
            )
            if logical_rows or hook is not None
            else None
        )
        if hook is not None:
            return self._forward_dispatched(hook, rows, routing, valid_rows, plan).view(
                shape
            )
        if rows.size(0) <= context.MAX_MOE_ROWS:
            if routing is not None:
                return self._forward_rows(rows, mask, routing=routing).view(shape)
            return self._forward_rows(rows, mask).view(shape)
        return self._forward_waves(rows, mask, routing, prefill_eager).view(shape)

    def _forward_owned(self, rows, image_mask, logical_rows, blas_workspace):
        routing = self.gate(
            rows,
            None if image_mask is None else image_mask.flatten(),
            logical_rows=logical_rows,
            blas_workspace=blas_workspace,
        )
        exchange = getattr(self, "dep4_exchange", None)
        if exchange is None:
            raise RuntimeError("DEP4 requires the request-owned executor")
        trace = getattr(self, "_routing_trace", None)
        if trace is not None:
            trace.record(self.layer_id, routing[1])
        routed = exchange(self, rows, routing)
        routed += self.shared_experts(rows)
        return routed.to(rows.dtype)

    def _forward_dispatched(self, hook, rows, routing, valid_rows, plan):
        trace = getattr(self, "_routing_trace", None)
        if trace is not None:
            trace.record(self.layer_id, routing[1])
        self._routing_trace = None
        try:
            value = hook(
                layer=self,
                rows=rows,
                routing=routing,
                valid_rows=valid_rows,
                plan=plan,
            )
        finally:
            self._routing_trace = trace
        if value.shape != rows.shape or value.device != rows.device:
            raise ValueError(
                "expert dispatch must return the input row shape and device"
            )
        return value.to(dtype=rows.dtype)

    def _forward_waves(self, rows, mask, routing, prefill_eager):
        workspace = self._prepare_prefill_workspace(rows, prefill_eager)
        wave_rows = context.MAX_MOE_ROWS if workspace is None else workspace.rows
        shared_output = self.shared_experts(rows)
        output = torch.empty_like(rows)
        for first in range(0, rows.size(0), wave_rows):
            self._trace_first = first
            last = min(first + wave_rows, rows.size(0))
            chunk_mask = None if mask is None else mask[first:last]
            if workspace is not None:
                chunk_routing = (
                    None
                    if routing is None
                    else (routing[0][first:last], routing[1][first:last])
                )
                chunk_routing = workspace.routing(
                    self, rows[first:last], chunk_mask, chunk_routing
                )
                output[first:last] = self._forward_rows(
                    rows[first:last],
                    chunk_mask,
                    routing=chunk_routing,
                    shared_output=shared_output[first:last],
                    workspace=workspace,
                )
            elif routing is None:
                output[first:last] = self._forward_rows(
                    rows[first:last],
                    chunk_mask,
                    shared_output=shared_output[first:last],
                )
            else:
                output[first:last] = self._forward_rows(
                    rows[first:last],
                    chunk_mask,
                    routing=(routing[0][first:last], routing[1][first:last]),
                    shared_output=shared_output[first:last],
                )
        return output
