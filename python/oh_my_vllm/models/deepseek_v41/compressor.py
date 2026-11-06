from __future__ import annotations

import torch
from torch import nn

from .kernel import bf16_linear, fp32_project
from .model_args import ModelArgs
from .model_linear import Linear, RMSNorm


class Compressor(nn.Module):
    """
    Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax
    gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so
    during decode it
    only yields every `compress_ratio` steps, holding the partial group in
    `kv_state`/`score_state`.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so Attention rotates
    afterwards.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]
        head_dim = args.head_dim
        self.compress_ratio = compress_ratio
        self.dim = args.dim
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the
        # softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        self.wkv = Linear(
            args.dim,
            head_dim,
            dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16,
        )
        if compress_ratio > 1:
            self.wgate = Linear(args.dim, head_dim, dtype=torch.float32)
            # tail of an incomplete group, carried across decode steps
            self.kv_state: torch.Tensor
            self.score_state: torch.Tensor
            state_shape = (args.max_batch_size, compress_ratio, head_dim)
            self.register_buffer(
                "kv_state",
                torch.zeros(state_shape, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "score_state",
                torch.full(state_shape, -torch.inf, dtype=torch.float32),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cache_slots: torch.Tensor | None = None,
        *,
        logical_rows=0,
    ) -> torch.Tensor | None:
        bsz, _, _ = x.size()
        if cache_slots is None:
            cache_slots = torch.arange(bsz, device=x.device)
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            if x.is_cuda and self.wkv.weight.dtype == torch.bfloat16:
                return self.norm(
                    bf16_linear(x, self.wkv.weight, logical_rows=logical_rows),
                    logical_rows=logical_rows,
                )
            return self.norm(self.wkv(x), logical_rows=logical_rows)

        x = x.float()
        if (
            x.is_cuda
            and self.dim == 5120
            and x.size(-1) == self.dim
            and self.wkv.weight.dtype == torch.float32
        ):
            rows = x.reshape(-1, self.dim).contiguous()
            shape = (x.size(0), x.size(1), self.head_dim)
            kv = fp32_project(rows, self.wkv.weight, logical_rows=logical_rows).view(
                shape
            )
            score = fp32_project(
                rows, self.wgate.weight, logical_rows=logical_rows
            ).view(shape)
        else:
            kv, score = self.wkv(x), self.wgate(x)
        prefix = start_pos % ratio
        if prefix:
            kv = torch.cat((self.kv_state[cache_slots, :prefix], kv), dim=1)
            score = torch.cat((self.score_state[cache_slots, :prefix], score), dim=1)
        export = getattr(self, "_state_export", None)
        if export is not None:
            export.tail(self.kv_state, self.score_state, kv, score, start_pos)
        complete = kv.size(1) // ratio * ratio
        remainder = kv.size(1) - complete
        if remainder:
            self.kv_state[cache_slots, :remainder] = kv[:, complete:]
            self.score_state[cache_slots, :remainder] = score[:, complete:]
        if complete == 0:
            return None
        kv = kv[:, :complete].unflatten(1, (-1, ratio))
        score = score[:, :complete].unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(kv.to(dtype), logical_rows=logical_rows)
