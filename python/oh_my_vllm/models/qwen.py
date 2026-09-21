"""Single-device, text-only Qwen3.5 FP8 model and its MTP head.

Architecture equations follow the Qwen checkpoint and its published model
implementation. No vLLM model classes, weight loaders or operators are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from oh_my_vllm.kernels import attention, fp8, gdn
from oh_my_vllm.kernels.convolution import causal_conv
from oh_my_vllm.kernels.elementwise import delta_gates, silu_mul
from oh_my_vllm.kernels.mtp_attention import MTPAttention
from oh_my_vllm.kernels.normalization import rms_norm, rotary

LayerCache = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Checkpoint:
    def __init__(self, path: str | Path, device: str | torch.device = "cuda") -> None:
        self.path = Path(path)
        self.device = device
        self.files = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.config = json.loads((self.path / "config.json").read_text())

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(
            self.path / self.files[name], framework="pt", device="cpu"
        ) as shard:
            return shard.get_tensor(name).to(self.device)

    def norm(self, name: str, offset: bool = True) -> torch.Tensor:
        weight = self.tensor(name + ".weight").float()
        return weight + 1 if offset else weight

    def linear(self, *names: str) -> Linear:
        weights = [self.tensor(name + ".weight") for name in names]
        scales = [
            self.tensor(name + ".weight_scale_inv").float()
            for name in names
            if name + ".weight_scale_inv" in self.files
        ]
        if scales and len(scales) != len(weights):
            raise ValueError("cannot pack mixed quantized/unquantized projections")
        return Linear(
            torch.cat(weights, 0) if len(weights) > 1 else weights[0],
            torch.cat(scales, 0) if len(scales) > 1 else scales[0] if scales else None,
        )


@dataclass
class Linear:
    weight: torch.Tensor
    scale: torch.Tensor | None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            return F.linear(x, self.weight)
        return fp8.linear(x, self.weight, self.scale)


@dataclass
class AttentionBatch:
    positions: torch.Tensor
    fa_slots: torch.Tensor
    attention: attention.PagedAttention | MTPAttention


@dataclass
class Batch(AttentionBatch):
    starts: torch.Tensor
    sequence_ids: torch.Tensor
    state_reads: torch.Tensor
    state_writes: torch.Tensor
    final_state_writes: torch.Tensor
    prefill_sequences: int
    prefill_tokens: int

    def delta(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        pool: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill requests precede decode/verification requests in batch order."""
        results = []
        n, t = self.prefill_sequences, self.prefill_tokens
        if n:
            initial = pool.index_select(0, self.state_reads[:n]).float()
            out, states = gdn.prefill(
                q[:t], k[:t], v[:t], decay[:t], beta[:t], initial, self.starts[: n + 1]
            )
            pool.index_copy_(0, self.final_state_writes[:n], states.to(pool.dtype))
            results.append(out)
        if n < self.state_reads.numel():
            out = gdn.recurrent(
                q[t:],
                k[t:],
                v[t:],
                decay[t:],
                beta[t:],
                pool,
                self.starts[n:] - t,
                self.state_reads[n:],
                self.state_writes[t:],
            )
            results.append(out)
        return results[0] if len(results) == 1 else torch.cat(results, 0)


class Layer:
    def __init__(self, checkpoint: Checkpoint, prefix: str, kind: str) -> None:
        self.kind = kind
        self.input_norm = checkpoint.norm(prefix + ".input_layernorm")
        self.post_norm = checkpoint.norm(prefix + ".post_attention_layernorm")
        self.gate_up = checkpoint.linear(
            prefix + ".mlp.gate_proj", prefix + ".mlp.up_proj"
        )
        self.down = checkpoint.linear(prefix + ".mlp.down_proj")
        if kind == "full_attention":
            p = prefix + ".self_attn"
            self.qkv = checkpoint.linear(p + ".q_proj", p + ".k_proj", p + ".v_proj")
            self.q_norm = checkpoint.norm(p + ".q_norm")
            self.k_norm = checkpoint.norm(p + ".k_norm")
            self.out = checkpoint.linear(p + ".o_proj")
        else:
            p = prefix + ".linear_attn"
            self.qkvz = checkpoint.linear(p + ".in_proj_qkv", p + ".in_proj_z")
            self.ba = checkpoint.linear(p + ".in_proj_b", p + ".in_proj_a")
            self.conv = checkpoint.tensor(p + ".conv1d.weight").squeeze(1).contiguous()
            self.a_log = checkpoint.tensor(p + ".A_log").float()
            self.dt_bias = checkpoint.tensor(p + ".dt_bias").float()
            self.gate_norm = checkpoint.norm(p + ".norm", offset=False)
            self.out = checkpoint.linear(p + ".out_proj")

    def full_attention(
        self, x: torch.Tensor, batch: AttentionBatch, cache: torch.Tensor
    ) -> torch.Tensor:
        qg, k, v = self.qkv(x).split([12288, 1024, 1024], -1)
        q, gate = qg.reshape(-1, 24, 512).chunk(2, -1)
        q = rotary(rms_norm(q, self.q_norm), batch.positions)
        k = rotary(rms_norm(k.reshape(-1, 4, 256), self.k_norm), batch.positions)
        v = v.reshape(-1, 4, 256).contiguous()
        attention.append(cache, k, v, batch.fa_slots)
        out = batch.attention(q, cache)
        return self.out((out * gate.sigmoid()).flatten(1))

    def delta_attention(
        self, x: torch.Tensor, batch: Batch, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        conv_pool, state_pool = cache
        mixed, z = self.qkvz(x).split([10240, 6144], -1)
        mixed = causal_conv(
            mixed.contiguous(),
            self.conv,
            conv_pool,
            batch.sequence_ids,
            batch.starts,
            batch.state_reads,
            batch.state_writes,
        )
        q, k, v = mixed.split([2048, 2048, 6144], -1)
        q, k, v = (
            q.reshape(-1, 16, 128).contiguous(),
            k.reshape(-1, 16, 128).contiguous(),
            v.reshape(-1, 48, 128).contiguous(),
        )
        decay, beta = delta_gates(self.ba(x), self.a_log, self.dt_bias)
        out = batch.delta(q, k, v, decay, beta, state_pool)
        out = rms_norm(out, self.gate_norm, gate=z.reshape(-1, 48, 128))
        return self.out(out.flatten(1))

    def __call__(
        self, x: torch.Tensor, batch: Batch, cache: LayerCache
    ) -> torch.Tensor:
        normalized = rms_norm(x, self.input_norm)
        if self.kind == "full_attention":
            result = self.full_attention(normalized, batch, cache)
        else:
            result = self.delta_attention(normalized, batch, cache)
        x = x + result
        packed = self.gate_up(rms_norm(x, self.post_norm))
        return x + self.down(silu_mul(packed))


class Qwen:
    def __init__(
        self,
        path: str | Path,
        *,
        mtp: bool = False,
        device: str | torch.device = "cuda",
    ) -> None:
        checkpoint = Checkpoint(path, device)
        text = checkpoint.config["text_config"]
        expected = {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4,
            "hidden_act": "silu",
            "attention_bias": False,
            "attn_output_gate": True,
        }
        if any(text.get(k) != v for k, v in expected.items()):
            raise ValueError("model is not the supported Qwen3.5-27B architecture")
        rope = text.get("rope_parameters", {})
        if any(
            rope.get(k) != v
            for k, v in {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            }.items()
        ):
            raise ValueError("unsupported Qwen rotary configuration")
        if checkpoint.config["quantization_config"].get("weight_block_size") != [
            128,
            128,
        ]:
            raise ValueError("model requires block-scaled FP8 weights")
        self.kinds = text["layer_types"]
        if self.kinds != (["linear_attention"] * 3 + ["full_attention"]) * 16:
            raise ValueError("unexpected hybrid layer order")
        self.embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.head = checkpoint.tensor("lm_head.weight")
        self.norm = checkpoint.norm("model.language_model.norm")
        self.layers = [
            Layer(checkpoint, f"model.language_model.layers.{i}", kind)
            for i, kind in enumerate(self.kinds)
        ]
        self.mtp = None
        if mtp:
            self.mtp = Layer(checkpoint, "mtp.layers.0", "full_attention")
            self.mtp_fc = checkpoint.linear("mtp.fc")
            self.mtp_embedding_norm = checkpoint.norm("mtp.pre_fc_norm_embedding")
            self.mtp_hidden_norm = checkpoint.norm("mtp.pre_fc_norm_hidden")
            self.mtp_norm = checkpoint.norm("mtp.norm")

    def forward(
        self, tokens: torch.Tensor, batch: Batch, caches: list[LayerCache]
    ) -> torch.Tensor:
        hidden = F.embedding(tokens, self.embedding)
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden = layer(hidden, batch, cache)
        return rms_norm(hidden, self.norm)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.head).float()

    def draft(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
        batch: AttentionBatch,
        cache: torch.Tensor,
    ) -> torch.Tensor:
        embedded = rms_norm(
            F.embedding(tokens, self.embedding), self.mtp_embedding_norm
        )
        hidden = rms_norm(hidden, self.mtp_hidden_norm)
        hidden = self.mtp_fc(torch.cat((embedded, hidden), -1))
        return rms_norm(self.mtp(hidden, batch, cache), self.mtp_norm)
