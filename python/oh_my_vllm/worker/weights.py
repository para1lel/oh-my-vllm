import json
from dataclasses import dataclass, replace

import numpy as np

from ..kernels.fa import Weights as FAWeights
from ..kernels.ffn import Weights as FFNWeights
from ..kernels.gdn import Weights as GDNWeights
from ..kernels.mtp import InputWeights
from ..kernels.sample import Weights as SampleWeights
from ..kernels.vision import Weights as VisionWeights
from .checkpoint import Checkpoint


@dataclass(frozen=True)
class Layer:
    kind: str
    attention: object
    feed_forward: object


class Weights:
    def __init__(self, device, path, mtp):
        self.device = device
        self.checkpoint = Checkpoint(path)
        self.config = json.loads((self.checkpoint.path / "config.json").read_text())
        self.embedding = self.upload("model.language_model.embed_tokens.weight")
        self.sample = SampleWeights(
            self.floats("model.language_model.norm.weight"),
            self.upload("lm_head.weight"),
        )
        self.layers = tuple(
            self.layer(f"model.language_model.layers.{index}.", kind)
            for index, kind in enumerate(self.config["text_config"]["layer_types"])
        )
        first = self.layers[0]
        self.layers = (
            replace(
                first, attention=replace(first.attention, embedding=self.embedding)
            ),
            *self.layers[1:],
        )
        self.mtp_input = self.mtp_layer = self.mtp_sample = None
        if mtp:
            self.mtp_input = InputWeights(
                self.embedding,
                self.floats("mtp.pre_fc_norm_hidden.weight"),
                self.floats("mtp.pre_fc_norm_embedding.weight"),
                self.upload("mtp.fc.weight"),
            )
            self.mtp_layer = self.layer("mtp.layers.0.", "full_attention")
            self.mtp_sample = SampleWeights(
                self.floats("mtp.norm.weight"), self.sample.projection
            )
        self.visual = None

    def upload(self, name):
        if name.endswith(".weight_scale_inv"):
            return self.floats(name)
        value, kind = self.checkpoint.raw_tensor(name)
        dtype = {"BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F32": "float32"}[kind]
        return self.device.upload(value, dtype)

    def floats(self, name):
        return self.device.upload(self.checkpoint.tensor(name).astype(np.float32))

    def concatenate(self, names):
        if all(name.endswith(".weight_scale_inv") for name in names):
            values = [self.checkpoint.tensor(name).astype(np.float32) for name in names]
            return self.device.upload(np.concatenate(values))
        arrays, kinds = zip(
            *(self.checkpoint.raw_tensor(name) for name in names), strict=True
        )
        if len(set(kinds)) != 1:
            raise ValueError("projection parts have different storage types")
        dtype = {"BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F32": "float32"}[
            kinds[0]
        ]
        return self.device.upload(np.concatenate(arrays), dtype)

    def feed_forward(self, prefix):
        return FFNWeights(
            self.floats(prefix + "post_attention_layernorm.weight"),
            self.concatenate(
                [prefix + f"mlp.{name}_proj.weight" for name in ("gate", "up")]
            ),
            self.concatenate(
                [
                    prefix + f"mlp.{name}_proj.weight_scale_inv"
                    for name in ("gate", "up")
                ]
            ),
            self.upload(prefix + "mlp.down_proj.weight"),
            self.upload(prefix + "mlp.down_proj.weight_scale_inv"),
        )

    def attention(self, prefix):
        attention = prefix + "self_attn."
        return FAWeights(
            self.floats(prefix + "input_layernorm.weight"),
            self.concatenate(
                [attention + f"{name}_proj.weight" for name in ("q", "k", "v")]
            ),
            self.concatenate(
                [
                    attention + f"{name}_proj.weight_scale_inv"
                    for name in ("q", "k", "v")
                ]
            ),
            self.floats(attention + "q_norm.weight"),
            self.floats(attention + "k_norm.weight"),
            self.upload(attention + "o_proj.weight"),
            self.upload(attention + "o_proj.weight_scale_inv"),
        )

    def recurrent(self, prefix):
        attention = prefix + "linear_attn."
        convolution = self.upload(attention + "conv1d.weight")
        return GDNWeights(
            self.floats(prefix + "input_layernorm.weight"),
            self.concatenate(
                [attention + f"in_proj_{name}.weight" for name in ("qkv", "z")]
            ),
            self.concatenate(
                [
                    attention + f"in_proj_{name}.weight_scale_inv"
                    for name in ("qkv", "z")
                ]
            ),
            self.concatenate(
                [attention + f"in_proj_{name}.weight" for name in ("b", "a")]
            ),
            self.device.view(convolution, (10240, 4)),
            self.floats(attention + "A_log"),
            self.floats(attention + "dt_bias"),
            self.floats(attention + "norm.weight"),
            self.upload(attention + "out_proj.weight"),
            self.upload(attention + "out_proj.weight_scale_inv"),
        )

    def layer(self, prefix, kind):
        attention = (
            self.attention(prefix)
            if kind == "full_attention"
            else self.recurrent(prefix)
        )
        return Layer(kind, attention, self.feed_forward(prefix))

    def attention_variant(self, weights, variant):
        first, rows = {
            "full": (0, 14336),
            "kv_only": (12288, 2048),
            "query_only": (0, 12288),
        }[variant]
        return replace(
            weights,
            projection=self.device.view(weights.projection, (rows, 5120), first * 5120),
            projection_scale=self.device.view(
                weights.projection_scale, (rows // 128, 40), first // 128 * 40
            ),
        )

    def vision(self):
        if self.visual is not None:
            return self.visual
        block_names = (
            "norm1.weight",
            "norm1.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias",
        )
        merger_names = (
            "norm.weight",
            "norm.bias",
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
        )
        blocks = tuple(
            tuple(
                self.upload(f"model.visual.blocks.{index}.{name}")
                for name in block_names
            )
            for index in range(27)
        )
        patch_weight = self.upload("model.visual.patch_embed.proj.weight")
        patch = (
            self.device.view(patch_weight, (1152, 1536)),
            self.upload("model.visual.patch_embed.proj.bias"),
            self.upload("model.visual.pos_embed.weight"),
        )
        merger = tuple(
            self.upload("model.visual.merger." + name) for name in merger_names
        )
        self.visual = VisionWeights(blocks, patch, merger)
        return self.visual
