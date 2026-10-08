"""Project-owned DSpark checkpoint, dual-source backbone, and learned heads.

Target features are true layer outputs at the same token position. Persistent
KV contains their projection only; the seven bidirectional noise rows remain
local to one proposal. No checkpoint Python code is imported or executed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 248320
    markov_rank: int = 256
    block_size: int = 7
    mask_token_id: int = 248070
    target_layer_ids: tuple[int, ...] = (5, 19, 33, 47, 61)
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    rope_factor: float = 32.0
    original_max_position_embeddings: int = 8192
    max_position_embeddings: int = 262144
    beta_fast: float = 32.0
    beta_slow: float = 1.0

    @property
    def feature_size(self) -> int:
        return self.hidden_size * len(self.target_layer_ids)

    @classmethod
    def read(cls, path: str | Path) -> DSparkConfig:
        """Accept the fixed, fully implemented local checkpoint architecture."""
        raw = json.loads((Path(path) / "config.json").read_text())
        config = cls()
        expected = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "draft_vocab_size": config.vocab_size,
            "markov_rank": config.markov_rank,
            "block_size": config.block_size,
            "mask_token_id": config.mask_token_id,
            "target_layer_ids": list(config.target_layer_ids),
            "num_target_layers": 64,
            "rms_norm_eps": config.rms_norm_eps,
            "max_position_embeddings": config.max_position_embeddings,
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_sliding_window": False,
            "sliding_window": None,
            "projector_type": "dspark",
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "attention_mode": "gqa",
            "layer_types": ["full_attention"] * config.num_hidden_layers,
        }
        for key, value in expected.items():
            if key not in raw or raw[key] != value:
                raise ValueError(f"unsupported DSpark config {key}: {raw.get(key)!r}")
        for name in ("dflash_config", "dspark_config"):
            nested = raw.get(name, {})
            for key in (
                "target_layer_ids",
                "mask_token_id",
                "projector_type",
                "markov_head_type",
                "markov_rank",
                "enable_confidence_head",
                "confidence_head_with_markov",
                "attention_mode",
            ):
                if nested.get(key) != expected[key]:
                    raise ValueError(f"inconsistent DSpark config {name}.{key}")
        rope = raw.get("rope_parameters", {})
        for key, value in {
            "rope_type": "yarn",
            "rope_theta": config.rope_theta,
            "factor": config.rope_factor,
            "original_max_position_embeddings": config.original_max_position_embeddings,
            "beta_fast": config.beta_fast,
            "beta_slow": config.beta_slow,
        }.items():
            if rope.get(key) != value:
                raise ValueError(f"unsupported DSpark YaRN parameter {key}")
        return config


def expected_shapes(config: DSparkConfig) -> dict[str, tuple[int, ...]]:
    h, m = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    shapes = {
        "fc.weight": (h, config.feature_size),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
        "markov_head.markov_w1.weight": (config.vocab_size, config.markov_rank),
        "markov_head.markov_w2.weight": (config.vocab_size, config.markov_rank),
        "confidence_head.proj.weight": (1, h + config.markov_rank),
        "confidence_head.proj.bias": (1,),
    }
    for i in range(config.num_hidden_layers):
        prefix = f"layers.{i}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (h,),
                prefix + "post_attention_layernorm.weight": (h,),
                prefix + "self_attn.q_proj.weight": (q, h),
                prefix + "self_attn.k_proj.weight": (kv, h),
                prefix + "self_attn.v_proj.weight": (kv, h),
                prefix + "self_attn.o_proj.weight": (h, q),
                prefix + "self_attn.q_norm.weight": (config.head_dim,),
                prefix + "self_attn.k_norm.weight": (config.head_dim,),
                prefix + "mlp.gate_proj.weight": (m, h),
                prefix + "mlp.up_proj.weight": (m, h),
                prefix + "mlp.down_proj.weight": (h, m),
            }
        )
    return shapes


class DSparkCheckpoint:
    """Validate all headers before transferring any tensor to the target device."""

    def __init__(self, path: str | Path, device: str | torch.device) -> None:
        self.path = Path(path)
        self.device = torch.device(device)
        self.file = self.path / "model.safetensors"
        self.identities = {
            name: self._identity(self.path / name)
            for name in ("config.json", "model.safetensors")
        }
        self.config = DSparkConfig.read(self.path)
        shapes = expected_shapes(self.config)
        with safe_open(self.file, framework="pt", device="cpu") as tensors:
            actual = set(tensors.keys())
            if actual != set(shapes):
                raise ValueError(
                    "DSpark tensor keys differ: "
                    f"missing={sorted(set(shapes) - actual)}, "
                    f"extra={sorted(actual - set(shapes))}"
                )
            for key, shape in shapes.items():
                tensor = tensors.get_slice(key)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "BF16":
                    raise ValueError(f"DSpark tensor shape/dtype mismatch: {key}")
        self.validate_files()

    @staticmethod
    def _identity(path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def validate_files(self) -> None:
        for name, identity in self.identities.items():
            if self._identity(self.path / name) != identity:
                raise RuntimeError(f"DSpark checkpoint changed during loading: {name}")

    def provenance(self) -> dict:
        """Bind loaded tensors to stable source bytes, outside measured execution."""
        self.validate_files()
        hashes = {}
        for name in self.identities:
            digest = hashlib.sha256()
            with (self.path / name).open("rb") as stream:
                while block := stream.read(1 << 20):
                    digest.update(block)
            hashes[name] = digest.hexdigest()
            self.validate_files()
        return {
            "path": str(self.path.resolve()),
            "config_sha256": hashes["config.json"],
            "weights_sha256": {"model.safetensors": hashes["model.safetensors"]},
        }

    def tensor(self, name: str) -> torch.Tensor:
        self.validate_files()
        with safe_open(self.file, framework="pt", device="cpu") as tensors:
            result = tensors.get_tensor(name).to(self.device)
        self.validate_files()
        return result


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3 norm: normalized BF16 is rounded before the learned multiplier."""
    if x.device.type == "cuda":
        from oh_my_vllm.kernels.dspark_attention import rms_norm as dspark_rms_norm

        return dspark_rms_norm(x, weight, eps)
    values = x.float()
    normalized = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    ).to(x.dtype)
    return normalized * weight.to(x.dtype)


def yarn_frequencies(
    config: DSparkConfig, device: str | torch.device
) -> tuple[torch.Tensor, float]:
    """HF YaRN's interpolation/extrapolation blend, stored as rounded FP32."""
    dim = config.head_dim
    indices = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    position_frequencies = config.rope_theta ** (indices / dim)
    extrapolation = 1 / position_frequencies
    interpolation = 1 / (config.rope_factor * position_frequencies)

    def correction(rotations):
        return (
            dim
            * math.log(
                config.original_max_position_embeddings / (rotations * 2 * math.pi)
            )
            / (2 * math.log(config.rope_theta))
        )

    low = max(0, math.floor(correction(config.beta_fast)))
    high = min(dim - 1, math.ceil(correction(config.beta_slow)))
    if low == high:
        high += 0.001
    ramp = (
        (torch.arange(dim // 2, device=device, dtype=torch.float32) - low)
        / (high - low)
    ).clamp(0, 1)
    extrapolation_factor = 1 - ramp
    frequencies = (
        interpolation * (1 - extrapolation_factor)
        + extrapolation * extrapolation_factor
    )
    return frequencies, 1 + 0.1 * math.log(config.rope_factor)


class DSparkLayer:
    def __init__(self, checkpoint: DSparkCheckpoint, index: int) -> None:
        prefix = f"layers.{index}."
        tensor = checkpoint.tensor
        self.input_norm = tensor(prefix + "input_layernorm.weight")
        self.post_norm = tensor(prefix + "post_attention_layernorm.weight")
        self.q_norm = tensor(prefix + "self_attn.q_norm.weight").float()
        self.k_norm = tensor(prefix + "self_attn.k_norm.weight").float()
        self.q = tensor(prefix + "self_attn.q_proj.weight")
        self.k = tensor(prefix + "self_attn.k_proj.weight")
        self.v = tensor(prefix + "self_attn.v_proj.weight")
        self.o = tensor(prefix + "self_attn.o_proj.weight")
        self.gate = tensor(prefix + "mlp.gate_proj.weight")
        self.up = tensor(prefix + "mlp.up_proj.weight")
        self.down = tensor(prefix + "mlp.down_proj.weight")


class DSparkModel:
    def __init__(self, target_model, path: str | Path) -> None:
        self.target = target_model
        self.device = target_model.embedding.device
        checkpoint = DSparkCheckpoint(path, self.device)
        self.config = checkpoint.config
        config = self.config
        if target_model.embedding.shape != (config.vocab_size, config.hidden_size):
            raise ValueError("DSpark target embedding shape disagrees with checkpoint")
        if target_model.embedding.dtype != torch.bfloat16:
            raise ValueError("DSpark requires the target BF16 embedding")
        self.fc = checkpoint.tensor("fc.weight")
        self.hidden_norm = checkpoint.tensor("hidden_norm.weight")
        self.norm = checkpoint.tensor("norm.weight")
        self.markov_w1 = checkpoint.tensor("markov_head.markov_w1.weight")
        self.markov_w2 = checkpoint.tensor("markov_head.markov_w2.weight")
        self.confidence_weight = checkpoint.tensor("confidence_head.proj.weight")
        self.confidence_bias = checkpoint.tensor("confidence_head.proj.bias")
        self.layers = [
            DSparkLayer(checkpoint, i) for i in range(config.num_hidden_layers)
        ]
        self.inv_freq, self.attention_factor = yarn_frequencies(config, self.device)
        self.provenance = checkpoint.provenance()
        logger.info("DSPARK_CHECKPOINT_PROVENANCE %s", json.dumps(self.provenance))

    def inject(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        caches: list[torch.Tensor],
    ) -> None:
        from oh_my_vllm.kernels.dspark_attention import append, prepare_qk

        config = self.config
        context = rms_norm(
            F.linear(features, self.fc), self.hidden_norm, config.rms_norm_eps
        )
        shape = (-1, config.num_key_value_heads, config.head_dim)
        for layer, cache in zip(self.layers, caches, strict=True):
            k = F.linear(context, layer.k).view(shape)
            v = F.linear(context, layer.v).view(shape)
            _, k = prepare_qk(
                None,
                k,
                None,
                layer.k_norm,
                positions,
                self.inv_freq,
                self.attention_factor,
                config.rms_norm_eps,
            )
            append(cache, k, v, slots)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        caches: list[torch.Tensor],
    ) -> torch.Tensor:
        from oh_my_vllm.kernels.dspark_attention import attention, prepare_qk

        config = self.config
        hidden = F.embedding(tokens, self.target.embedding)
        q_shape = (*tokens.shape, config.num_attention_heads, config.head_dim)
        kv_shape = (*tokens.shape, config.num_key_value_heads, config.head_dim)
        for layer, cache in zip(self.layers, caches, strict=True):
            normalized = rms_norm(hidden, layer.input_norm, config.rms_norm_eps)
            q = F.linear(normalized, layer.q).view(q_shape)
            k = F.linear(normalized, layer.k).view(kv_shape)
            v = F.linear(normalized, layer.v).view(kv_shape)
            q, k = prepare_qk(
                q,
                k,
                layer.q_norm,
                layer.k_norm,
                positions,
                self.inv_freq,
                self.attention_factor,
                config.rms_norm_eps,
            )
            attended = attention(q, cache, tables, lengths, k, v)
            hidden = hidden + F.linear(attended.flatten(-2), layer.o)
            normalized = rms_norm(hidden, layer.post_norm, config.rms_norm_eps)
            activated = F.silu(F.linear(normalized, layer.gate))
            activated = activated * F.linear(normalized, layer.up)
            hidden = hidden + F.linear(activated, layer.down)
        return rms_norm(hidden, self.norm, config.rms_norm_eps)

    def backbone(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward(tokens, positions, tables, lengths, caches)
        # Shared target head may expose FP32 logits. The published draft adds
        # its BF16 Markov projection to BF16 head logits before softmax.
        logits = self.target.logits(hidden.flatten(0, 1)).to(hidden.dtype)
        return hidden, logits.view(*tokens.shape, self.config.vocab_size)

    def step(
        self, hidden: torch.Tensor, base_logits: torch.Tensor, previous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = F.embedding(previous, self.markov_w1)
        logits = base_logits + F.linear(latent, self.markov_w2)
        confidence_input = torch.cat((hidden, latent), dim=-1)
        confidence = (
            F.linear(confidence_input, self.confidence_weight, self.confidence_bias)
            .squeeze(-1)
            .float()
            .sigmoid()
        )
        return logits, confidence

    def greedy(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep the Markov chain on device; callers stop at the first low score.

        Later candidates never enter earlier confidence decisions. Computing the
        fixed seven rows in a graph reduces host synchronization for greedy runs.
        """
        hidden, base = self.backbone(tokens, positions, tables, lengths, caches)
        previous = tokens[:, 0]
        candidates, confidence, valid = [], [], []
        for position in range(self.config.block_size):
            logits, score = self.step(hidden[:, position], base[:, position], previous)
            maximum, previous = logits.max(-1)
            candidates.append(previous)
            confidence.append(score)
            valid.append(torch.isfinite(maximum))
        return (
            torch.stack(candidates, -1),
            torch.stack(confidence, -1),
            torch.stack(valid, -1),
        )
