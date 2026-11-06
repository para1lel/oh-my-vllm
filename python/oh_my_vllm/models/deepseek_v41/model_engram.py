from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from . import model_context as context
from .engram import EngramLayout
from .kernel import engram_mix
from .model_args import ModelArgs
from .model_linear import Linear


class ParallelEngramEmbedding(nn.Module):
    """
    The n-gram hash table, sharded over its rows. Stays fp8: rows are dequantized on
    lookup.
    """

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.part_num_embeddings = (
            num_embeddings + context.world_size - 1
        ) // context.world_size
        self.vocab_start_idx = (
            context.rank if context.world_size > 1 else 0
        ) * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings

        self.block_size = context.fp8_block_size
        # The loader replaces these CPU placeholders with file-backed FP8 views.
        self.weight = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
            )
        )
        self.scale = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                dim // self.block_size,
                dtype=context.scale_dtype,
                device="cpu",
            )
        )

    def _staged_lookup(self, indices):
        from .kernel_native import _transform_module

        if indices.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("Engram indices must be int32 or int64")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("stage Engram lookup before CUDA Graph recording")
        module = _transform_module()
        device = indices.device
        host_indices = (
            indices.to(device="cpu", dtype=torch.int64).contiguous().flatten()
        )
        count = host_indices.numel()
        output = torch.empty((count, self.dim), dtype=torch.bfloat16, device=device)
        if not count:
            return output.reshape(*indices.shape, self.dim)
        capacity = min(65536, count)
        stride = ((self.dim + self.dim // 32 + 16) // 16) * 16
        signature = (device, capacity, stride)
        buffers = getattr(self, "_lookup_buffers", None)
        if buffers is None or buffers[0] != signature:
            if buffers is not None:
                for _, _, event in buffers[1]:
                    event.synchronize()
            torch.cuda.current_stream(device).synchronize()
            slots = [
                (
                    torch.empty(
                        (capacity, stride),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    ),
                    torch.empty((capacity, stride), dtype=torch.uint8, device=device),
                    torch.cuda.Event(),
                )
                for _ in range(2)
            ]
            self._lookup_buffers = (signature, slots)
        slots = self._lookup_buffers[1]
        weight = self.weight.detach().view(torch.uint8)
        scales = self.scale.detach().view(torch.uint8)
        for chunk, first in enumerate(range(0, count, capacity)):
            host, packed, event = slots[chunk % 2]
            event.synchronize()
            length = min(capacity, count - first)
            staging = host[:length]
            device_rows = packed[:length]
            module.engram_lookup_stage(
                weight,
                scales,
                host_indices[first : first + length],
                staging,
                self.vocab_start_idx,
            )
            stream = torch.cuda.current_stream(device)
            packed.record_stream(stream)
            output.record_stream(stream)
            try:
                device_rows.copy_(staging, non_blocking=True)
                module.engram_lookup_decode(device_rows, output[first : first + length])
                event.record(stream)
            except BaseException:
                stream.synchronize()
                raise
        return output.reshape(*indices.shape, self.dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if (
            indices.device.type == "cuda"
            and self.weight.dtype == torch.float8_e4m3fn
            and self.scale.dtype == torch.float8_e8m0fnu
            and self.dim % 32 == 0
        ):
            with torch.cuda.device(indices.device):
                values = self._staged_lookup(indices)
                if context.world_size > 1:
                    dist.all_reduce(values)
                return values
        device = indices.device
        host_indices = indices.to(device="cpu")
        mask = (host_indices < self.vocab_start_idx) | (
            host_indices >= self.vocab_end_idx
        )
        local_indices = host_indices - self.vocab_start_idx
        local_indices = local_indices.masked_fill(mask, 0)

        values = F.embedding(local_indices, self.weight)
        scales = F.embedding(local_indices, self.scale)
        values = values.float().unflatten(
            -1, (-1, self.block_size)
        ) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(mask.unsqueeze(-1), 0)

        if device.type != "cpu":
            values = values.pin_memory().to(device, non_blocking=True)
        if context.world_size > 1:
            dist.all_reduce(values)
        return values


class Engram(nn.Module):
    """
    Writes an n-gram lookup into the residual stream, gated by how well it matches that
    stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy
    plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, args: ModelArgs, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6

        self.embed = ParallelEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index], layout.head_dim
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.n_hash_cols = n_hash_cols
        self.head_dim = layout.head_dim
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        logical_rows=0,
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L],
        False shuts
        the gate so those positions pass through untouched. `embedding` supplies the
        staged lookup
        output as [B, L, n_hash_cols * head_dim], which keeps the host table lookup out
        of a
        captured step.
        """
        if embedding is None:
            embedding = self.embed(hash_ids).flatten(-2)
        kv = self.wkv(embedding)
        return engram_mix(
            x,
            kv,
            self.q_weight,
            self.k_weight,
            token_mask,
            epsilon=self.eps,
            clamp_value=self.clamp_value,
            logical_rows=logical_rows,
        )
