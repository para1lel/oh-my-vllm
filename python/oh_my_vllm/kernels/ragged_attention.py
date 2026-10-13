"""Torch Flash SDPA over independently compacted causal sequences."""

from itertools import pairwise

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


def causal_attention(query, keys, values, query_starts, kv_starts):
    outputs = []
    for (qleft, qright), (kleft, kright) in zip(
        pairwise(query_starts), pairwise(kv_starts), strict=True
    ):
        q = query[qleft:qright].transpose(0, 1).unsqueeze(0)
        k = keys[kleft:kright].transpose(0, 1).unsqueeze(0)
        v = values[kleft:kright].transpose(0, 1).unsqueeze(0)
        query_length, kv_length = qright - qleft, kright - kleft
        if query_length <= 0 or kv_length < query_length:
            raise ValueError("causal attention requires 0 < query length <= KV length")
        mask = (
            causal_lower_right(query_length, kv_length)
            if 1 < query_length < kv_length
            else None
        )
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=query_length == kv_length,
            enable_gqa=True,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)
