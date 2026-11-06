from __future__ import annotations

import torch
from torch import nn

from . import model_context as context
from .compressor import Compressor
from .kernel import act_quant, fp4_act_quant, sparse_attn
from .model_args import ModelArgs
from .model_indexer import Indexer
from .model_linear import (
    ColumnParallelLinear,
    Linear,
    RMSNorm,
    RowParallelLinear,
    request_einsum,
    separate_request_math,
)
from .model_positions import (
    apply_rotary_emb,
    compressed_context_width,
    get_prefill_window_topk_idxs,
    get_window_topk_idxs,
    position_rows,
    position_window_topk_idxs,
    precompute_freqs_cis,
)
from .packed_cache import PackedHistory, history_view, write_history


class Attention(nn.Module):
    """
    Latent attention over two KV sources at once, concatenated into one `sparse_attn`
    call: a
    sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed
    positions
    reaching further back. Q and the output projection are both low-rank, the latter
    grouped.

    compress_ratio > 0 does not mean the layer compresses its own KV: only
    kv_source_layers do,
    the rest read that same cache.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // context.world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // context.world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        self.window_kv_cache: torch.Tensor
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim),
            persistent=False,
        )
        if self.is_kv_source:
            self.compress_kv_cache = PackedHistory(
                args, self.head_dim, self.compress_ratio, 16, torch.float8_e4m3fn
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        ).to(torch.get_default_device())
        self.freqs_cis: torch.Tensor
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _apply(self, function, recurse=True):
        if isinstance(getattr(self, "compress_kv_cache", None), PackedHistory):
            self.compress_kv_cache.apply(function)
        return super()._apply(function, recurse=recurse)

    def _window_kv(
        self, x, freqs_cis, start_pos, cache_slots, *, logical_rows=0, position=None
    ):
        """
        This layer's sliding-window K and the window positions every query may attend
        to. The K
        stays fp8, quantized over the whole post-RoPE vector, RoPE tail included.
        """
        bsz, seqlen, _ = x.size()
        win = self.window_size
        projected = self.wkv(x)
        kv = (
            self.kv_norm(projected, logical_rows=logical_rows)
            if logical_rows
            else self.kv_norm(projected)
        )
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        act_quant(
            kv, context.fp8_block_size, context.scale_fmt, context.scale_dtype, True
        )
        export = getattr(self, "_state_export", None)
        if export is not None:
            export.window(self.window_kv_cache, kv, start_pos)
        if (
            start_pos == 0
        ):  # prefill: attend over this chunk, seeding the ring buffer for decode
            if seqlen <= win:
                self.window_kv_cache[cache_slots, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (
                    self.window_kv_cache[cache_slots, cutoff:win],
                    self.window_kv_cache[cache_slots, :cutoff],
                ) = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            window_kv = kv
            topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        elif (
            seqlen == 1
        ):  # decode: one token into the ring buffer, attend over the whole window
            if position is None:
                self.window_kv_cache[cache_slots, start_pos % win] = kv.squeeze(1)
                topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
            else:
                ring = position.reshape(-1).remainder(win).expand_as(cache_slots)
                self.window_kv_cache[cache_slots, ring] = kv.squeeze(1)
                topk_idxs = position_window_topk_idxs(win, bsz, position)
            window_kv = self.window_kv_cache[cache_slots]
        else:  # chunked prefill: attend over the previous ring and this causal chunk
            previous_kv = self._previous_window(x, start_pos, cache_slots, position)
            window_kv = torch.cat((previous_kv, kv), dim=1)
            write_kv = kv[:, -win:]
            write_start = start_pos + seqlen - write_kv.size(1)
            if position is None:
                positions = torch.arange(
                    write_start,
                    start_pos + seqlen,
                    device=x.device,
                ).remainder(win)[None, :]
            else:
                positions = (
                    position.reshape(-1, 1)
                    + torch.arange(seqlen - write_kv.size(1), seqlen, device=x.device)
                ).remainder(win)
            self.window_kv_cache[cache_slots[:, None], positions] = write_kv
            topk_idxs = get_prefill_window_topk_idxs(win, bsz, seqlen, start_pos)
        return window_kv, topk_idxs.to(x.device)

    def _previous_window(self, x, start_pos, cache_slots, position):
        bsz = x.size(0)
        win = self.window_size
        previous = min(start_pos, win)
        if position is not None:
            previous_positions = position.reshape(-1, 1) + torch.arange(
                -previous, 0, device=x.device
            )
            offsets = previous_positions.remainder(win)
            previous_kv = self.window_kv_cache[cache_slots].gather(
                1, offsets[..., None].expand(bsz, previous, self.head_dim)
            )
        elif start_pos >= win:
            ring_start = start_pos % win
            previous_kv = torch.cat(
                (
                    self.window_kv_cache[cache_slots, ring_start:],
                    self.window_kv_cache[cache_slots, :ring_start],
                ),
                dim=1,
            )
        else:
            previous_kv = self.window_kv_cache[cache_slots, :previous]

        return previous_kv

    def _compress_topk_idxs(
        self,
        x,
        qr,
        latent,
        start_pos,
        offset,
        compress_len,
        cache_slots,
        *,
        logical_rows=0,
        position=None,
    ):
        """
        Which compressed positions each query attends to. Index sources run their own
        indexer;
        the layers in between reuse the result their source published.
        """
        if not self.is_index_source:
            return context.shared_attn.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            idxs = self.indexer(
                x,
                qr,
                latent,
                start_pos,
                offset,
                cache_slots,
                logical_rows=logical_rows,
                position=position,
            )
        context.shared_attn.topk_idxs = idxs
        return idxs

    def _compress_kv(
        self, x, qr, start_pos, offset, cache_slots, *, logical_rows=0, position=None
    ):
        """
        The shared compressed KV and the compressed positions every query may attend to.
        This
        layer compresses its own KV only when it is a source; otherwise it just reads
        the cache.
        """
        _, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(
                x, start_pos, cache_slots, logical_rows=logical_rows
            )
        # the indexer needs the latent before RoPE, so it runs before the cache is
        # written
        idxs = self._compress_topk_idxs(
            x,
            qr,
            latent,
            start_pos,
            offset,
            compress_len,
            cache_slots,
            logical_rows=logical_rows,
            position=position,
        )
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
            apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with
            # E8M0.
            encoded, scales = fp4_act_quant(latent, 16, scale_dtype=torch.float8_e4m3fn)
            first = (
                start_pos // ratio
                if position is None
                else position.div(ratio, rounding_mode="floor")
            )
            write_history(
                self.compress_kv_cache, cache_slots, first, encoded, scales, 16
            )
        if self.is_kv_source:
            # Slot indexing makes a copy, so publish only after the current write.
            width = (
                compress_len
                if position is None
                else compressed_context_width(
                    start_pos + seqlen, ratio, self.compress_kv_cache.size(1)
                )
            )
            context.shared_attn.compress_kv = history_view(
                self.compress_kv_cache, cache_slots, width
            )
        return context.shared_attn.compress_kv, idxs

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cache_slots: torch.Tensor | None = None,
        *,
        logical_rows=0,
        position: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()
        if cache_slots is None:
            cache_slots = torch.arange(bsz, device=x.device)
        if position is None:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = position_rows(self.freqs_cis, position, seqlen)
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x), logical_rows=logical_rows)
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv, topk_idxs = self._window_kv(
            x,
            freqs_cis,
            start_pos,
            cache_slots,
            logical_rows=logical_rows,
            position=position,
        )
        native_window = kv
        native_compressed = kv.new_empty((bsz, 0, self.head_dim))
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x,
                qr,
                start_pos,
                kv.size(1),
                cache_slots,
                logical_rows=logical_rows,
                position=position,
            )
            native_compressed = compress_kv
            kv = None
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
            native_window=native_window,
            native_compressed=native_compressed,
        )
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence
        # einsum not
        # Linear. convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the
        # memory.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = request_einsum(
            "bsgd,grd->bsgr",
            o,
            wo_a,
            separate_requests=separate_request_math(x, logical_rows),
        )
        x = self.wo_b(o.flatten(2))
        return x
