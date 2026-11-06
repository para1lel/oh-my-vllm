from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from . import model_context as context
from .kernel import markov_linear


class ParallelHead(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = vocab_size // context.world_size
        self.local_partitions = (
            4 if context.world_size == 1 and context.expert_world_size == 4 else 1
        )
        # bf16 in the checkpoint, kept as fp32 here so the logits come out in fp32
        # directly
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32)
        )

    def _project(self, inputs, operation):
        if self.local_partitions == 1:
            return operation(inputs, self.weight)
        return torch.cat(
            [
                operation(inputs, weight)
                for weight in self.weight.chunk(self.local_partitions)
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        full_logits=False,
        logit_start=0,
        *,
        logical_rows=0,
        separate_requests=False,
    ):
        """
        x: [b, s, d]. Generation only needs the last position, so that is the default.
        """
        if not full_logits:
            if logit_start:
                raise ValueError("logit_start requires full logits")
            x = x[:, -1]
        elif type(logit_start) is not int or not 0 <= logit_start < x.size(1):
            raise ValueError("logit_start is outside the hidden sequence")
        else:
            x = x[:, logit_start:]
        if (
            self.dim == 256
            and x.is_cuda
            and x.size(-1) == self.dim
            and self.weight.dtype == torch.float32
        ):
            logits = self._project(x.float(), markov_linear)
        elif separate_requests and logical_rows == 1 and x.numel() > self.dim:
            shape = x.shape[:-1]
            rows = x.reshape(-1, self.dim)
            logits = torch.cat(
                [
                    self._project(rows[index : index + 1].float(), F.linear)
                    for index in range(rows.size(0))
                ],
                dim=0,
            ).reshape(*shape, self.part_vocab_size)
        elif logical_rows and x.ndim == 3 and x.size(1) > 1:
            # The released verification projects each source step on its own.
            logits = torch.cat(
                [
                    self._project(x[:, step : step + 1].float(), F.linear)
                    for step in range(x.size(1))
                ],
                dim=1,
            )
        else:
            logits = self._project(x.float(), F.linear)
        if context.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(context.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
