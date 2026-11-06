from __future__ import annotations

import torch
import torch.nn.functional as F

from . import model_context as context
from .indexing import INDEX_SCORE_WORK_BYTES, score_chunks, topk_indices
from .kernel import fp4_act_quant
from .model_args import ModelArgs
from .model_linear import (
    ColumnParallelLinear,
    Linear,
    RMSNorm,
    request_linear,
    separate_request_math,
)
from .model_positions import apply_rotary_emb, compressed_context_width, position_rows
from .packed_cache import PackedHistory, PackedView, history_view, write_history


class Indexer(torch.nn.Module):
    """
    Keeps the `index_topk` best compressed positions per query.

    A small side attention: fp4 query heads against one shared key per compressed
    position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the
    second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        # the index keys are derived from the compressor's latent, so only a layer that
        # compresses
        # its own KV can produce them; every other indexer reads them from that layer's
        # cache
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // context.world_size
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.index_head_dim
        )
        self.weights_proj = ColumnParallelLinear(
            self.dim, self.n_heads, dtype=torch.bfloat16
        )
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache = PackedHistory(
                args, args.index_head_dim, self.compress_ratio, 32, torch.float8_e8m0fnu
            )

    def _apply(self, function, recurse=True):
        if isinstance(getattr(self, "k_cache", None), PackedHistory):
            self.k_cache.apply(function)
        return super()._apply(function, recurse=recurse)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor,
        start_pos: int,
        offset: int,
        cache_slots: torch.Tensor | None = None,
        *,
        logical_rows=0,
        position: torch.Tensor | None = None,
    ):
        """
        `latent` is this layer's RoPE-free compressed latent, None when this layer does
        not
        compress or when its current group is still incomplete. An index-key owner turns
        it into
        index keys here, which has to happen before Attention overwrites that same
        storage with
        the RoPE'd, quantized values.
        """
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        if cache_slots is None:
            cache_slots = torch.arange(bsz, device=x.device)
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        if position is None:
            width = end_pos // ratio
        elif self.owns_k:
            width = compressed_context_width(end_pos, ratio, self.k_cache.size(1))
        else:
            width = compressed_context_width(
                end_pos, ratio, context.shared_attn.index_k.size(1)
            )

        self._publish_keys(
            latent, start_pos, position, cache_slots, width, x, logical_rows
        )

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        if position is None:
            apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        else:
            apply_rotary_emb(
                q[..., -rd:], position_rows(self.freqs_cis, position, seqlen)
            )
        fp4_act_quant(q, context.fp4_block_size, True)

        source_keys = context.shared_attn.index_k
        index_k = (
            source_keys.with_width(width).materialize()
            if isinstance(source_keys, PackedView)
            else source_keys[:, :width]
        )
        weights = request_linear(
            self.weights_proj,
            x,
            separate_requests=separate_request_math(x, logical_rows),
        ) * (self.softmax_scale * self.n_heads**-0.5)
        return self._select_requests(
            q, index_k, weights, x, start_pos, offset, position, seqlen, logical_rows
        )

    def _publish_keys(
        self, latent, start_pos, position, cache_slots, width, x, logical_rows
    ):
        ratio, rd = self.compress_ratio, self.rope_head_dim
        # latent is None while a group is still filling up, so there is nothing to
        # publish yet
        if self.owns_k:
            if latent is not None:
                # a latent stands for the first token of its group, so group j takes
                # position j * ratio
                if position is None:
                    group_start = start_pos - start_pos % ratio
                    freqs = self.freqs_cis[
                        group_start : group_start + latent.size(1) * ratio : ratio
                    ]
                else:
                    group_start = position.div(ratio, rounding_mode="floor") * ratio
                    freqs = position_rows(
                        self.freqs_cis, group_start, latent.size(1), stride=ratio
                    )
                k = self.k_norm(
                    request_linear(
                        self.wk,
                        latent,
                        separate_requests=separate_request_math(x, logical_rows),
                    ),
                    logical_rows=logical_rows,
                )
                apply_rotary_emb(k[..., -rd:], freqs)
                encoded, scales = fp4_act_quant(k, context.fp4_block_size)
                first = (
                    start_pos // ratio
                    if position is None
                    else position.div(ratio, rounding_mode="floor")
                )
                write_history(
                    self.k_cache,
                    cache_slots,
                    first,
                    encoded,
                    scales,
                    context.fp4_block_size,
                )
            context.shared_attn.index_k = history_view(self.k_cache, cache_slots, width)

    def _select_requests(
        self, q, index_k, weights, x, start_pos, offset, position, seqlen, logical_rows
    ):
        bsz = x.size(0)
        ratio = self.compress_ratio
        request_widths = None
        if position is not None and position.device.type == "cpu":
            # CPU top-k tie selection depends on the physical row width.
            starts = position.reshape(-1).tolist()
            if len(starts) == 1:
                starts *= bsz
            request_widths = tuple((start + seqlen) // ratio for start in starts)
        groups = range(bsz) if separate_request_math(x, logical_rows) else (None,)
        indices, candidates = [], []
        for request in groups:
            selected = slice(None) if request is None else slice(request, request + 1)
            selected_position = (
                position[selected]
                if position is not None and position.ndim != 0
                else position
            )
            selected_widths = (
                request_widths[selected] if request_widths is not None else None
            )
            candidate_blocks = (
                context.shared_attn.candidates[selected]
                if self.uses_candidates
                else None
            )
            result, blocks = self._select_indices(
                q[selected],
                index_k[selected],
                weights[selected],
                start_pos,
                offset,
                position=selected_position,
                request_widths=selected_widths,
                candidate_blocks=candidate_blocks,
            )
            indices.append(result)
            if blocks is not None:
                candidates.append(blocks)
        if self.is_candidate_source:
            context.shared_attn.candidates = torch.cat(candidates, dim=0)
        return torch.cat(indices, dim=0)

    def _select_indices(
        self,
        query,
        keys,
        weights,
        start_pos,
        offset,
        *,
        position=None,
        request_widths=None,
        candidate_blocks=None,
    ):
        width = keys.size(1)
        ratio = self.compress_ratio
        positions = torch.arange(width, device=query.device)
        indices, candidates = [], []
        for first, last, index_score in score_chunks(
            query,
            keys,
            weights,
            work_bytes=INDEX_SCORE_WORK_BYTES,
            rank_sum=context.world_size > 1,
            head_partitions=context.expert_world_size if context.world_size == 1 else 1,
        ):
            base = start_pos if position is None else position
            compress_lens = (
                (base + torch.arange(first + 1, last + 1, device=query.device)) // ratio
            ).unsqueeze(-1)
            index_score.masked_fill_(positions >= compress_lens, -torch.inf)
            if self.is_candidate_source:
                candidates.append(
                    select_candidate_blocks(
                        index_score,
                        compress_lens,
                        self.candidate_topk_blocks,
                        self.candidate_block_size,
                        request_widths=request_widths,
                    )
                )
            elif self.uses_candidates:
                keep = candidate_blocks[:, first:last].repeat_interleave(
                    self.candidate_block_size, dim=-1
                )[..., :width]
                index_score.masked_fill_(~keep, -torch.inf)
            topk = min(self.index_topk, width)
            # Valid model execution requires finite QDQ reconstruction, weighted_heads
            # products/sums, and FP32 rank reduction with finite final BF16 scores.
            # Causal and candidate masks add only -inf after that score arithmetic.
            idxs = topk_indices(
                index_score, topk, request_widths=request_widths, finite_scores=True
            )
            indices.append(torch.where(idxs < compress_lens, idxs + offset, -1).int())
            del index_score
        blocks = torch.cat(candidates, dim=1) if self.is_candidate_source else None
        return torch.cat(indices, dim=1), blocks


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
    *,
    request_widths=None,
) -> torch.Tensor:
    """
    Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per
    query.

    `logits` is [..., n_positions] with positions the query cannot reach already at
    -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a
    plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns one bool
    per block.
    Consumers expand only their current query chunk to position granularity.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in:
    # it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    if request_widths is None:
        top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        # fewer reachable blocks than topk_blocks means leftover picks came back -inf:
        # drop them
        return torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, top.indices, top.values > -torch.inf
        )
    block_widths = tuple(
        (width + block_size - 1) // block_size for width in request_widths
    )
    top = topk_indices(
        scores, min(topk_blocks, num_blocks), request_widths=block_widths
    )
    padded = F.pad(scores, (0, 1), value=-torch.inf)
    keep = torch.zeros_like(padded, dtype=torch.bool).scatter_(
        -1, top, padded.gather(-1, top) > -torch.inf
    )
    return keep[..., :num_blocks]
