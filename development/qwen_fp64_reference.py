"""Small-sequence independent FP64 text-model oracle for the fixed checkpoint.

No inference model, IR operation, or repository CUDA module is imported. FP8
checkpoint weights are decoded into FP64. Activations, KV, recurrent state,
normalization, nonlinearities, projections, and logits stay FP64. There is no
activation quantization or intermediate BF16/FP32 rounding in this ideal oracle.
Large numerical experiments belong to individual operators, not this model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from development import fp64_math as math64

PRECISION_POLICY = {
    "reference": "independent_fp64_checkpoint_mathematics",
    "weight_input": "original checkpoint storage values",
    "weight_decode": "float64 multiplication of FP8 values and stored block scales",
    "activation_quantization": "none",
    "intermediate_rounding": "none",
    "arithmetic": "float64",
    "cache": "float64",
    "logits_and_metrics": "float64",
    "scope": "short text sequences, one sequence per independent cache",
    "maximum_call_tokens": 512,
    "maximum_context_tokens": 1024,
}


class Weights:
    """Independent safetensors reader with projection-row streaming."""

    def __init__(self, checkpoint, device="cpu", *, projection_rows=1024):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("use an explicit CUDA device index for the reference")
        self.rows = projection_rows
        self.config = json.loads((self.path / "config.json").read_text())["text_config"]
        self.files = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        if projection_rows <= 0 or projection_rows % 128:
            raise ValueError(
                "projection rows must contain complete 128-row scale blocks"
            )
        self.read_names = set()

    def stored(self, name, rows=None):
        path = (self.path / self.files[name]).resolve()
        if path.parent != self.path:
            raise ValueError("checkpoint shard must be in the checkpoint directory")
        with safe_open(path, framework="pt", device="cpu") as shard:
            value = (
                shard.get_tensor(name) if rows is None else shard.get_slice(name)[rows]
            )
        self.read_names.add(name)
        return value

    def tensor(self, name):
        return self.stored(name).to(device=self.device, dtype=torch.float64)

    def norm(self, prefix, *, offset=True):
        values = self.tensor(prefix + ".weight")
        return values + 1 if offset else values

    def embedding(self, ids):
        values = self.stored("model.language_model.embed_tokens.weight")
        return values.index_select(0, ids.cpu()).to(
            device=self.device, dtype=torch.float64
        )

    def linear_blocks(self, x, prefix):
        if x.dtype != torch.float64 or x.device != self.device:
            raise ValueError("reference projection requires FP64 input on its device")
        name = prefix + ".weight"
        with safe_open(
            self.path / self.files[name], framework="pt", device="cpu"
        ) as shard:
            shape = shard.get_slice(name).get_shape()
        scale_name = prefix + ".weight_scale_inv"
        for begin in range(0, shape[0], self.rows):
            end = min(begin + self.rows, shape[0])
            raw = self.stored(name, slice(begin, end)).to(self.device)
            scale = None
            if scale_name in self.files:
                scale = self.stored(
                    scale_name, slice(begin // 128, (end + 127) // 128)
                ).to(self.device)
            weight = math64.decode_weight(raw, scale)
            yield begin, x @ weight.T

    def linear(self, x, prefix):
        return torch.cat([result for _, result in self.linear_blocks(x, prefix)], -1)


@dataclass
class Cache:
    length: int = 0
    attention: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)


class Model:
    """Pure mathematical text decoder, limited to small numerical comparisons."""

    def __init__(self, weights):
        self.weights = weights
        self.config = weights.config
        c = self.config
        self.eps = c["rms_norm_eps"]
        if c["hidden_act"] != "silu" or c.get("attention_bias", False):
            raise ValueError("reference supports bias-free SiLU text checkpoints")
        rope = c["rope_parameters"]
        if rope["rope_type"] != "default":
            raise ValueError("reference requires default rotary frequencies")
        self.rotary_dim = int(c["head_dim"] * rope["partial_rotary_factor"])
        self.theta = rope["rope_theta"]

    def _full_attention(self, x, prefix, layer, cache):
        w, c = self.weights, self.config
        p = prefix + ".self_attn"
        width, heads = c["head_dim"], c["num_attention_heads"]
        query, gate = (
            w.linear(x, p + ".q_proj").reshape(len(x), heads, 2 * width).chunk(2, -1)
        )
        key = w.linear(x, p + ".k_proj").reshape(
            len(x), c["num_key_value_heads"], width
        )
        value = w.linear(x, p + ".v_proj").reshape_as(key)
        positions = torch.arange(cache.length, cache.length + len(x), device=x.device)
        query = math64.rotary(
            math64.rms(query, w.norm(p + ".q_norm"), self.eps),
            positions,
            theta=self.theta,
            rotary_dim=self.rotary_dim,
        )
        key = math64.rotary(
            math64.rms(key, w.norm(p + ".k_norm"), self.eps),
            positions,
            theta=self.theta,
            rotary_dim=self.rotary_dim,
        )
        if layer in cache.attention:
            prior_key, prior_value = cache.attention[layer]
            key, value = torch.cat((prior_key, key)), torch.cat((prior_value, value))
        cache.attention[layer] = key, value
        result = math64.attention(query, key, value, first_position=cache.length)
        return w.linear((result * gate.sigmoid()).flatten(1), p + ".o_proj")

    def _delta_attention(self, x, prefix, layer, cache):
        w, c = self.weights, self.config
        p = prefix + ".linear_attn"
        key_heads, value_heads = c["linear_num_key_heads"], c["linear_num_value_heads"]
        key_width, value_width = c["linear_key_head_dim"], c["linear_value_head_dim"]
        mixed = w.linear(x, p + ".in_proj_qkv")
        kernel = w.tensor(p + ".conv1d.weight").squeeze(1)
        if layer in cache.delta:
            history, state = cache.delta[layer]
        else:
            history = x.new_zeros((mixed.shape[1], kernel.shape[1] - 1))
            state = x.new_zeros((value_heads, value_width, key_width))
        convolved, history = math64.causal_conv(mixed, kernel, history)
        query, key, value = convolved.split(
            (key_heads * key_width, key_heads * key_width, value_heads * value_width),
            -1,
        )
        decay, beta = math64.delta_gates(
            w.linear(x, p + ".in_proj_b"),
            w.linear(x, p + ".in_proj_a"),
            w.tensor(p + ".A_log"),
            w.tensor(p + ".dt_bias"),
        )
        result, state = math64.delta_chunk(
            query.reshape(len(x), key_heads, key_width),
            key.reshape(len(x), key_heads, key_width),
            value.reshape(len(x), value_heads, value_width),
            decay,
            beta,
            state,
        )
        cache.delta[layer] = history, state
        gate = w.linear(x, p + ".in_proj_z").reshape_as(result)
        result = math64.rms(
            result, w.norm(p + ".norm", offset=False), self.eps
        ) * math64.silu(gate)
        return w.linear(result.flatten(1), p + ".out_proj")

    @torch.inference_mode()
    def forward(self, token_ids, cache=None, *, observer=None):
        if token_ids.ndim != 1 or token_ids.dtype != torch.int64 or len(token_ids) == 0:
            raise ValueError("reference requires a nonempty vector of int64 token IDs")
        cache = Cache() if cache is None else cache
        if (
            len(token_ids) > PRECISION_POLICY["maximum_call_tokens"]
            or cache.length + len(token_ids)
            > PRECISION_POLICY["maximum_context_tokens"]
        ):
            raise ValueError("full-model reference is restricted to small comparisons")
        w = self.weights
        x = w.embedding(token_ids)
        for layer, kind in enumerate(self.config["layer_types"]):
            prefix = f"model.language_model.layers.{layer}"
            normalized = math64.rms(x, w.norm(prefix + ".input_layernorm"), self.eps)
            if kind == "full_attention":
                mixed = self._full_attention(normalized, prefix, layer, cache)
            elif kind == "linear_attention":
                mixed = self._delta_attention(normalized, prefix, layer, cache)
            else:
                raise ValueError("unknown text layer type")
            residual = x + mixed
            normalized = math64.rms(
                residual, w.norm(prefix + ".post_attention_layernorm"), self.eps
            )
            gate = w.linear(normalized, prefix + ".mlp.gate_proj")
            up = w.linear(normalized, prefix + ".mlp.up_proj")
            x = residual + w.linear(math64.silu(gate) * up, prefix + ".mlp.down_proj")
            if observer is not None:
                observer(layer, x, cache)
        cache.length += len(token_ids)
        return math64.rms(x, w.norm("model.language_model.norm"), self.eps), cache

    def logits(self, hidden):
        return self.weights.linear(hidden, "lm_head")
