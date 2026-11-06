from __future__ import annotations

import torch

# Set once by Transformer.__init__; one model per process, so layers just read them.
world_size = 1
expert_world_size = 1
rank = 0
default_dtype = (
    torch.float8_e4m3fn
)  # storage dtype for Linear weights, from ModelArgs.dtype
fp8_block_size = 32  # one fp8 scale per 32x32 weight block / 32 activations
fp4_block_size = 32  # one fp4 scale per 32 elements along K
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu
MAX_MOE_ROWS = 1024


class SharedAttentionRuntime:
    """
    What attention layers hand down the stack instead of recomputing. Layers run in
    order and
    every source writes before its consumers read, so one slot each is enough within one
    forward.
    Sources: compress_kv and index_k from kv_source_layers, topk_idxs from
    index_source_layers,
    candidates from candidate_source_layer.
    """

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None

    def reset_query_state(self):
        self.topk_idxs = None
        self.candidates = None


# One model runs in each process.
shared_attn = SharedAttentionRuntime()
