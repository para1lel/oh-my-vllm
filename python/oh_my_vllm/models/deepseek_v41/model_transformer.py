from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from . import model_context as context
from .engram import EngramLayout, NgramHashState
from .image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from .kernel import require_sparse_status, reset_sparse_status
from .model_args import ModelArgs
from .model_block import Block, make_identity_pre_mix
from .model_dspark import DSparkBlock
from .model_head import ParallelHead
from .model_linear import ParallelEmbedding, RMSNorm
from .model_positions import context_token_bucket
from .model_sampling import sample
from .moe_prefill import PrefillWorkspace, projection_rows
from .vision import Aligner, ViT


@dataclass
class _TargetInputs:
    hidden: torch.Tensor | None
    cache_slots: torch.Tensor
    image_mask: torch.Tensor | None
    engram_mask: torch.Tensor | None
    engram_hashes: torch.Tensor | None


class Transformer(nn.Module):
    """
    DeepSeek-V4.1: embed -> expand to hc_mult copies -> blocks -> collapse -> logits.
    Building
    this sets the globals at the top of the file. The tokenizer only feeds the engram
    token map.
    """

    def __init__(self, args: ModelArgs, tokenizer=None):

        context.expert_world_size = (
            dist.get_world_size() if dist.is_initialized() else 1
        )
        context.world_size = (
            1 if args.parallel_strategy == "DEP4" else context.expert_world_size
        )
        context.rank = dist.get_rank() if dist.is_initialized() else 0
        context.default_dtype = (
            torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        )
        super().__init__()
        self._moe_prefill_workspace = PrefillWorkspace(
            projection_rows(
                strategy=args.parallel_strategy, partitions=context.world_size
            )
        )
        self.max_seq_len = args.max_seq_len
        self.temperature = args.temperature
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash = (
            NgramHashState(args, self.engram_layout, tokenizer)
            if self.engram_layout is not None
            else None
        )
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, self.engram_layout))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        self.target_layer_ids = args.dspark_target_layer_ids
        if args.dspark_block_size:
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))
                self.mtp[-1].embed = self.embed
                self.mtp[-1].head = self.head
        self.hc_mult = args.hc_mult
        self.vision = None
        if args.vision_enabled:
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            # learned embeddings for the image span delimiters
            self.image_start = nn.Parameter(torch.empty(args.dim))
            self.image_end = nn.Parameter(torch.empty(args.dim))
            self.image_newline = nn.Parameter(torch.empty(args.dim))
        for layer in self.layers:
            layer.ffn._prefill_workspace = self._moe_prefill_workspace

    def _apply(self, function, recurse=True):
        if getattr(self, "_physical_cache", None) is not None:
            raise RuntimeError("bound model storage does not support tensor conversion")
        return super()._apply(function, recurse=recurse)

    def reserve_cache(self, slots, end_pos):
        cache = getattr(self, "_physical_cache", None)
        if cache is not None:
            cache.require_active_slots(slots)
            cache.reserve_workspace(len(slots), end_pos)

    def _bind_state_exports(
        self, cache_slots, cache_slots_cpu=None, positions_cpu=None
    ):
        export = getattr(self, "_state_export", None)
        if export is not None and export.steps:
            slots = cache_slots.tolist() if cache_slots_cpu is None else cache_slots_cpu
            if positions_cpu is None:
                export.bind(slots)
            else:
                export.bind(slots, positions=positions_cpu)

    @torch.inference_mode()
    def encode_image(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: torch.Tensor):
        """
        Overwrite each image's token span in h with its ViT/aligner features. The IMAGE
        slots take
        the aligner rows in row-major order; the span delimiters take learned
        embeddings.
        """
        for i, sample_images in enumerate(images):
            for img in sample_images or ():
                types = img.types.to(h.device)
                span = h[i, img.start : img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = self.encode_image(
                    img.patches.to(h.device), img.n_vit_h, img.n_vit_w
                )
                span[types == IMAGE] = embeds.to(h.dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images=None,
        token_types: torch.Tensor | None = None,
        cache_slots: torch.Tensor | None = None,
        full_logits: bool = False,
        compute_logits: bool = True,
        collect_main_hidden: bool = True,
        logit_start: int = 0,
        sample_output: bool = True,
        cache_slots_cpu=None,
        position: torch.Tensor | None = None,
        check_status: bool = True,
        engram_values: dict[int, torch.Tensor] | None = None,
        reserve_cache: bool = True,
        logical_rows: int | None = None,
        cache_positions_cpu=None,
        routing_trace=None,
        execution_plan=None,
        valid_rows=None,
        dispatch_combine=None,
    ):
        """Run target tokens with the bound cache and execution settings."""
        inputs = self._target_inputs(
            input_ids,
            start_pos,
            images,
            token_types,
            cache_slots,
            cache_slots_cpu,
            cache_positions_cpu,
            position,
            reserve_cache,
            engram_values,
            valid_rows,
            full_logits,
            compute_logits,
        )
        separate_requests = logical_rows is not None
        if logical_rows is None:
            logical_rows = (
                input_ids.size(0) if start_pos > 0 and input_ids.size(1) > 1 else 0
            )
        options, block_options = self._block_options(
            input_ids,
            logical_rows,
            separate_requests,
            position,
            execution_plan,
            valid_rows,
            dispatch_combine,
        )
        h, main_hiddens = self._forward_layers(
            inputs,
            start_pos,
            position,
            options,
            block_options,
            engram_values,
            collect_main_hidden,
            routing_trace,
        )
        return self._target_output(
            input_ids,
            h,
            compute_logits,
            full_logits,
            logit_start,
            sample_output,
            logical_rows,
            separate_requests,
            options,
            main_hiddens,
            check_status,
        )

    def _target_inputs(
        self,
        input_ids,
        start_pos,
        images,
        token_types,
        cache_slots,
        cache_slots_cpu,
        cache_positions_cpu,
        position,
        reserve_cache,
        engram_values,
        valid_rows,
        full_logits,
        compute_logits,
    ):
        """
        input_ids: [b, s], every entry a real token id -- generate.py only ever passes
        positions it
        has already filled, so the padding it uses internally never reaches here.
        token_types /
        images carry the VL inputs built by image_processor.prepare_vl_inputs; image
        spans must lie
        inside the first (start_pos 0) chunk.
        """
        if full_logits and not compute_logits:
            raise ValueError("full logits require vocab projection")
        if position is not None and (
            position.device != input_ids.device
            or position.dtype != torch.int64
            or (position.ndim != 0 and position.shape != (input_ids.size(0), 1))
        ):
            raise ValueError(
                "device positions must be scalar or one int64 start per request"
            )
        if valid_rows is not None and (
            valid_rows.shape != (input_ids.numel(),)
            or valid_rows.dtype != torch.bool
            or valid_rows.device != input_ids.device
        ):
            raise ValueError("valid rows must be one device boolean per target row")
        context.shared_attn.reset_query_state()
        reset_sparse_status()
        if cache_slots is None:
            cache_slots = torch.arange(input_ids.size(0), device=input_ids.device)
        self._bind_state_exports(cache_slots, cache_slots_cpu, cache_positions_cpu)
        if reserve_cache and hasattr(self, "_physical_cache"):
            slots = cache_slots.tolist() if cache_slots_cpu is None else cache_slots_cpu
            end = start_pos + input_ids.size(1)
            self.reserve_cache(
                slots, context_token_bucket(end) if position is not None else end
            )
        image_mask = None if token_types is None else token_types >= 0  # TEXT is -1
        # image tokens take no part in an n-gram and get no engram contribution;
        # text-only needs no mask
        engram_mask = None if image_mask is None else ~image_mask
        engram_hashes = (
            self.engram_hash(
                input_ids, start_pos, engram_mask, cache_slots, position=position
            )
            if self.engram_hash is not None and engram_values is None
            else None
        )
        h = self.embed(input_ids)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"
            self.merge_image_embeddings(images, h)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        return _TargetInputs(h, cache_slots, image_mask, engram_mask, engram_hashes)

    def _block_options(
        self,
        input_ids,
        logical_rows,
        separate_requests,
        position,
        execution_plan,
        valid_rows,
        dispatch_combine,
    ):
        options = {"logical_rows": logical_rows} if logical_rows else {}
        block_options = dict(options)
        if (
            self._moe_prefill_workspace.rows > context.MAX_MOE_ROWS
            and input_ids.size(1) > 1
            and not separate_requests
            and position is None
            and execution_plan is None
            and valid_rows is None
            and dispatch_combine is None
        ):
            block_options["prefill_eager"] = True
        if (
            execution_plan is not None
            or valid_rows is not None
            or dispatch_combine is not None
        ):
            block_options.update(
                plan=execution_plan,
                valid_rows=valid_rows,
                dispatch_combine=dispatch_combine,
            )

        return options, block_options

    def _forward_layers(
        self,
        inputs,
        start_pos,
        position,
        options,
        block_options,
        engram_values,
        collect_main_hidden,
        routing_trace,
    ):
        h = inputs.hidden
        inputs.hidden = None
        cache_slots = inputs.cache_slots
        image_mask = inputs.image_mask
        engram_mask = inputs.engram_mask
        engram_hashes = inputs.engram_hashes
        main_hiddens = []
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for i, layer in enumerate(self.layers):
            layer.ffn._routing_trace = routing_trace
            if layer.engram is not None:
                if engram_values is None:
                    hash_ids = engram_hashes[:, :, layer.engram.layer_hash_index, :]
                    embedding = None
                else:
                    hash_ids = None
                    embedding = engram_values[i]
                h = layer.engram(
                    h,
                    hash_ids,
                    engram_mask,
                    embedding=embedding,
                    **options,
                )
            # the MTP head reads the attention input of its target layers, not their
            # output
            if collect_main_hidden and i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(
                h,
                start_pos,
                pre_mix,
                image_mask,
                cache_slots,
                position=position,
                **block_options,
            )
        h = layer.hc_pre(h, pre_mix)

        return h, main_hiddens

    def _target_output(
        self,
        input_ids,
        h,
        compute_logits,
        full_logits,
        logit_start,
        sample_output,
        logical_rows,
        separate_requests,
        options,
        main_hiddens,
        check_status,
    ):
        if compute_logits:
            logits = self.head(
                self.norm(h, **options),
                full_logits=full_logits,
                logit_start=logit_start,
                logical_rows=logical_rows,
                separate_requests=separate_requests,
            )
            output_ids = (
                sample(logits, self.temperature)
                if sample_output
                else torch.empty(
                    input_ids.size(0),
                    dtype=torch.long,
                    device=input_ids.device,
                )
            )
        else:
            logits = None
            output_ids = torch.empty(
                input_ids.size(0),
                dtype=torch.long,
                device=input_ids.device,
            )

        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        # A captured step cannot read the status on the host; check it after replay
        # instead.
        if check_status:
            require_sparse_status()
        return output_ids, logits, main_hidden

    @torch.inference_mode()
    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
        cache_slots: torch.Tensor | None = None,
        *,
        check_status: bool = True,
        cache_slots_cpu=None,
        position: torch.Tensor | None = None,
    ):
        if position is not None and (
            position.device != input_ids.device
            or position.dtype != torch.int64
            or position.ndim != 0
            or start_pos <= 0
        ):
            raise ValueError(
                "MTP device position requires a positive scalar int64 start"
            )
        context.shared_attn.reset_query_state()
        reset_sparse_status()
        if cache_slots is None:
            cache_slots = torch.arange(input_ids.size(0), device=input_ids.device)
        self._bind_state_exports(cache_slots, cache_slots_cpu)
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        options = {} if position is None else {"position": position}
        for layer in self.mtp:
            h, pre_mix = layer(h, start_pos, pre_mix, main_x, cache_slots, **options)
        if check_status:
            require_sparse_status()
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, pre_mix, input_ids)

    @torch.inference_mode()
    def forward_spec_seed(
        self,
        main_hidden: torch.Tensor,
        start_pos: int = 0,
        cache_slots: torch.Tensor | None = None,
        *,
        cache_slots_cpu=None,
    ):
        """Store target hidden states that the next DSpark proposal can attend to."""
        if not self.mtp:
            return
        if cache_slots is None:
            cache_slots = torch.arange(main_hidden.size(0), device=main_hidden.device)
        self._bind_state_exports(cache_slots, cache_slots_cpu)
        first = self.mtp[0]
        main_x = first.main_norm(first.main_proj(main_hidden))
        for layer in self.mtp:
            layer.attn.seed(main_x, start_pos, cache_slots)
