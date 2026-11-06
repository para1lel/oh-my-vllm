from __future__ import annotations

import torch
from torch import nn

from . import model_context as context
from .kernel import act_quant, dot_linear, sparse_attn
from .model_args import ModelArgs
from .model_attention import Attention
from .model_block import Block
from .model_head import ParallelHead
from .model_linear import Linear, ParallelEmbedding, RMSNorm
from .model_positions import apply_rotary_emb, get_dspark_topk_idxs, position_rows
from .model_sampling import sample


class DSparkAttention(Attention):
    def _store_main_kv(self, main_x, start_pos, cache_slots, position=None):
        _bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim
        freqs_cis = (
            self.freqs_cis[start_pos : start_pos + seqlen]
            if position is None
            else position_rows(self.freqs_cis, position, seqlen)
        )
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], freqs_cis)
        act_quant(
            main_kv,
            context.fp8_block_size,
            context.scale_fmt,
            context.scale_dtype,
            True,
        )
        export = getattr(self, "_state_export", None)
        if export is not None:
            export.window(self.window_kv_cache, main_kv, start_pos)
        if position is not None:
            offsets = torch.arange(max(0, seqlen - win), seqlen, device=main_x.device)
            positions = position + offsets
            self.window_kv_cache[cache_slots[:, None], positions.remainder(win)] = (
                main_kv[:, -win:]
            )
        elif start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[cache_slots, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                (
                    self.window_kv_cache[cache_slots, cutoff:win],
                    self.window_kv_cache[cache_slots, :cutoff],
                ) = main_kv[:, -win:].split([win - cutoff, cutoff], dim=1)
        else:
            if seqlen > win:
                main_kv = main_kv[:, -win:]
                positions = torch.arange(
                    start_pos + seqlen - win,
                    start_pos + seqlen,
                    device=main_x.device,
                )
            else:
                positions = torch.arange(
                    start_pos,
                    start_pos + seqlen,
                    device=main_x.device,
                )
            self.window_kv_cache[cache_slots[:, None], positions.remainder(win)] = (
                main_kv
            )
        return main_kv

    def seed(self, main_x, start_pos, cache_slots):
        """Store accepted target hidden states without running draft queries."""
        self._store_main_kv(main_x, start_pos, cache_slots)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        main_x: torch.Tensor,
        cache_slots: torch.Tensor | None = None,
        *,
        position: torch.Tensor | None = None,
    ):
        assert self.compress_ratio == 0
        bsz, seqlen, _ = main_x.size()
        if cache_slots is None:
            cache_slots = torch.arange(bsz, device=x.device)
        win = self.window_size
        rd = self.rope_head_dim

        self._store_main_kv(main_x, start_pos, cache_slots, position)

        if start_pos == 0:
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = (
            self.freqs_cis[start_pos + seqlen : start_pos + seqlen + block_size]
            if position is None
            else position_rows(self.freqs_cis, position + seqlen, block_size)
        )

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(
            kv, context.fp8_block_size, context.scale_fmt, context.scale_dtype, True
        )

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos).to(x.device)
        kv = torch.cat([self.window_kv_cache[cache_slots], kv], dim=1)
        o = sparse_attn(
            q,
            kv,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
            native_window=kv,
            native_compressed=kv.new_empty((bsz, 0, self.head_dim)),
        )
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.embed = ParallelEmbedding(vocab_size, dspark_markov_rank)
        self.head = ParallelHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        logits = self.head(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # proj in the checkpoint is stored in bf16, while the parameter here is stored
        # in fp32 for fp32 confidence score.
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        if (
            hidden.is_cuda
            and self.proj.weight.dtype == torch.float32
            and hidden.size(-1) == self.proj.weight.size(1)
        ):
            return dot_linear(hidden.float(), self.proj.weight).squeeze(-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark stage stored under the mtp.* checkpoint namespace."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"
            self.main_proj = Linear(
                args.dim * len(args.dspark_target_layer_ids), args.dim
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(
                args.vocab_size, args.dspark_markov_rank
            )
            self.confidence_head = DSparkConfidenceHead(
                args.dim + args.dspark_markov_rank
            )
        self.embed: ParallelEmbedding | None = None
        self.head: ParallelHead | None = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        main_x: torch.Tensor,
        cache_slots: torch.Tensor | None = None,
        *,
        position: torch.Tensor | None = None,
    ):
        if start_pos == 0:
            self.attn(
                x, start_pos, main_x, cache_slots
            )  # prefill only seeds the window KV cache
            return x, pre_mix
        options = {} if position is None else {"position": position}
        return super().forward(
            x, start_pos, pre_mix, None, main_x, cache_slots, **options
        )  # drafts are text: no VL bias

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        assert self.embed is not None
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full(
            [input_ids.size(0), self.block_size], self.noise_token_id
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.head is not None
        x = self.hc_pre(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence
