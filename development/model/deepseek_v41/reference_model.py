"""Layer-streamed FP64 reference components, not inference runtime code."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from .engram import build_layout, hash_rows
from .reference_math import hc_collapse, hc_expand, hc_mixes, require_fp64, rms, route
from .weights import select_rows


@dataclass(frozen=True)
class ModelState:
    length: int
    layers: tuple
    engram_history: tuple[int, ...]


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    state: ModelState


class Model:
    def __init__(self, weights, config, token_map):
        if weights.device.type != "cpu":
            raise ValueError(
                "the target reference currently supports CPU execution only"
            )
        self.weights = weights
        self.config = copy.deepcopy(config["text_config"])
        self.image_token_id = config.get("image_token_id")
        self.token_map = np.array(token_map, copy=True)
        text = self.config
        if (
            self.token_map.shape != (text["vocab_size"],)
            or self.token_map.dtype.kind not in "iu"
        ):
            raise ValueError(
                "reference token map must contain one integer per vocabulary row"
            )
        self.hash_layout = None
        if text["engram_layer_ids"]:
            size = text["engram_compressed_vocab_size"]
            if not np.array_equal(np.unique(self.token_map), np.arange(size)):
                raise ValueError(
                    "compressed token map differs from the configured vocabulary"
                )
            self.hash_layout = build_layout(
                layer_ids=tuple(text["engram_layer_ids"]),
                max_ngram_size=text["engram_max_ngram_size"],
                n_heads=text["engram_n_heads"],
                bucket_size=text["engram_vocab_size"],
                compressed_vocab_size=size,
                table_sizes=tuple(text["engram_num_embeddings"]),
            )

    def _mixes(self, value, prefix, branch):
        return hc_mixes(
            value,
            self.weights.tensor(f"{prefix}.hc_{branch}_fn"),
            self.weights.tensor(f"{prefix}.hc_{branch}_scale"),
            self.weights.tensor(f"{prefix}.hc_{branch}_base"),
            iterations=self.config["hc_sinkhorn_iters"],
            eps=self.config["hc_eps"],
            norm_eps=self.config["rms_norm_eps"],
        )

    def _block(self, value, pre_mix, layer, previous, start, shared, image_mask=None):
        from .reference_attention import forward

        prefix = f"layers.{layer}"
        weights, config = self.weights, self.config
        attn_pre, post, comb = self._mixes(value, prefix, "attn")
        normalized = rms(
            hc_collapse(value, pre_mix),
            weights.tensor(prefix + ".attn_norm.weight"),
            eps=config["rms_norm_eps"],
        )
        output, state = forward(
            weights,
            prefix + ".attn",
            normalized,
            layer=layer,
            config=config,
            previous=previous,
            start=start,
            shared=shared,
        )
        value = hc_expand(output, value, post, comb)
        ffn_pre, post, comb = self._mixes(value, prefix, "ffn")
        normalized = rms(
            hc_collapse(value, attn_pre),
            weights.tensor(prefix + ".ffn_norm.weight"),
            eps=config["rms_norm_eps"],
        )
        output = moe(
            weights,
            prefix + ".ffn",
            normalized,
            config=config,
            layer=layer,
            image_mask=image_mask,
        )
        return hc_expand(output, value, post, comb), ffn_pre, state

    def _image_mask(self, tokens, token_types, embeddings, start):
        if token_types is None:
            if embeddings is not None:
                raise ValueError("image embeddings need explicit token types")
            return None
        types = np.asarray(token_types)
        if (
            types.shape != tokens.shape
            or types.dtype.kind not in "iu"
            or not np.isin(types, [-1, 0, 1, 2, 3]).all()
        ):
            raise ValueError("image token types must identify each input row")
        mask = types >= 0
        if not mask.any():
            if embeddings is not None:
                raise ValueError("image embeddings have no image token positions")
            return None
        if start != 0:
            raise ValueError("image embeddings belong to the first prefill chunk")
        if self.image_token_id is None or (tokens[mask] != self.image_token_id).any():
            raise ValueError("image positions must carry the configured image token ID")
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.float64
            or embeddings.shape != (int(mask.sum()), self.config["hidden_size"])
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("image embeddings must contain full finite CPU FP64 rows")
        return torch.from_numpy(mask)

    @torch.inference_mode()
    def forward(
        self,
        tokens,
        *,
        previous=None,
        endpoint_rows=None,
        token_types=None,
        image_embeddings=None,
    ):
        config = self.config
        tokens = select_rows(tokens, config["vocab_size"])
        if tokens.size == 0:
            raise ValueError("reference forward needs at least one input token")
        start = 0 if previous is None else previous.length
        if start + len(tokens) > config["max_position_embeddings"]:
            raise ValueError("reference input exceeds total context capacity")
        if previous is not None and len(previous.layers) != config["num_hidden_layers"]:
            raise ValueError("reference state layer count differs from the model")
        endpoints = select_rows(
            [len(tokens) - 1] if endpoint_rows is None else endpoint_rows, len(tokens)
        )
        if endpoints.size == 0:
            raise ValueError("reference forward needs at least one endpoint row")
        image_mask = self._image_mask(tokens, token_types, image_embeddings, start)
        history = () if previous is None else previous.engram_history
        hashes = None
        if self.hash_layout is not None:
            if len(history) != min(start, config["engram_max_ngram_size"] - 1):
                raise ValueError("reference continuation has incomplete Engram history")
            compressed = self.token_map[tokens].astype(np.int64, copy=False)
            if image_mask is not None:
                compressed[image_mask.numpy()] = -1
            hashes, history = hash_rows(
                compressed,
                self.hash_layout,
                pad_id=self.token_map[config["engram_pad_token_id"]],
                history=history,
            )
        value = self.weights.tensor("embed.weight", rows=tokens)
        if image_mask is not None:
            value[image_mask] = image_embeddings
        value = value[:, None, :].expand(-1, config["hc_mult"], -1).clone()
        pre_mix = value.new_zeros((len(tokens), config["hc_mult"]))
        pre_mix[:, 0] = 1.0
        states, features, shared = [], {}, {}
        for layer in range(config["num_hidden_layers"]):
            if layer in config["engram_layer_ids"]:
                slot = config["engram_layer_ids"].index(layer)
                value = engram(
                    self.weights,
                    f"layers.{layer}.engram",
                    value,
                    hashes[:, slot],
                    norm_eps=config["rms_norm_eps"],
                    token_mask=None if image_mask is None else ~image_mask,
                )
            if layer in config["dspark_target_layer_ids"]:
                features[layer] = value.mean(-2)
            prior = None if previous is None else previous.layers[layer]
            value, pre_mix, state = self._block(
                value, pre_mix, layer, prior, start, shared, image_mask
            )
            states.append(state)
        collapsed = hc_collapse(value, pre_mix)[
            torch.as_tensor(endpoints, device=value.device)
        ]
        normalized = rms(
            collapsed, self.weights.tensor("norm.weight"), eps=config["rms_norm_eps"]
        )
        logits = self.weights.linear(normalized, "head")
        selected_features = (
            torch.cat(
                [features[layer] for layer in config["dspark_target_layer_ids"]], dim=-1
            )
            if features
            else value.new_empty((len(tokens), 0))
        )
        return ModelOutput(
            logits,
            selected_features,
            ModelState(start + len(tokens), tuple(states), history),
        )


class TorchWeights:
    def __init__(self, source, *, device, row_limit=2048):
        if type(row_limit) is not int or row_limit < 1:
            raise ValueError("reference projection row limit must be positive")
        self.source = source
        self.device = torch.empty(0, device=device).device
        self.row_limit = row_limit

    def tensor(self, name, *, rows=None):
        array = self.source.tensor(name, rows=rows)
        if array.dtype != np.float64:
            raise ValueError("reference weight reader must return FP64")
        return torch.from_numpy(array).to(self.device)

    def linear(self, value, prefix):
        require_fp64(value)
        if value.device != self.device:
            raise ValueError("reference value and weights must use the same device")
        name = prefix + ".weight"
        rows = self.source.layout[name]["shape"][0]
        output = torch.empty(
            (*value.shape[:-1], rows), dtype=torch.float64, device=self.device
        )
        for begin in range(0, rows, self.row_limit):
            end = min(begin + self.row_limit, rows)
            matrix = self.tensor(name, rows=np.arange(begin, end, dtype=np.int64))
            output[..., begin:end] = value @ matrix.T
            del matrix
        return output


def expert(weights, prefix, value, *, limit, routing_weight=None):
    gate = weights.linear(value, prefix + ".w1")
    up = weights.linear(value, prefix + ".w3")
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    hidden = functional.silu(gate) * up
    if routing_weight is not None:
        require_fp64(routing_weight)
        hidden = hidden * routing_weight.unsqueeze(-1)
    return weights.linear(hidden, prefix + ".w2")


def moe(weights, prefix, value, *, config, layer, image_mask=None):
    logits = weights.linear(value, prefix + ".gate")
    bias = weights.tensor(prefix + ".gate.bias")
    if image_mask is not None:
        if image_mask.shape != value.shape[:-1] or image_mask.dtype != torch.bool:
            raise ValueError("routing image mask must identify each input row")
        bias = torch.where(
            image_mask[..., None], weights.tensor(prefix + ".gate.bias_vl"), bias
        )
    topk = config[
        "num_experts_per_tok"
        if layer < config["num_hidden_layers"]
        else "dspark_num_experts_per_tok"
    ]
    indices, probabilities = route(
        logits, bias, topk=topk, scale=config["routed_scaling_factor"]
    )
    output = torch.zeros_like(value)
    for index in indices.unique(sorted=True).tolist():
        rows, slots = torch.where(indices == index)
        computed = expert(
            weights,
            f"{prefix}.experts.{index}",
            value[rows],
            limit=config["swiglu_limit"],
            routing_weight=probabilities[rows, slots],
        )
        output.index_add_(0, rows, computed)
    return output + expert(
        weights, prefix + ".shared_experts", value, limit=config["swiglu_limit"]
    )


def engram(weights, prefix, value, hash_ids, *, norm_eps, token_mask=None):
    require_fp64(value)
    rows, copies, width = value.shape
    embedding = weights.tensor(prefix + ".embed.weight", rows=hash_ids.reshape(-1))
    projected = weights.linear(embedding.reshape(rows, -1), prefix + ".wkv")
    key, added = projected.split((copies * width, width), dim=-1)
    key = key.reshape(rows, copies, width)
    norm_weight = weights.tensor(prefix + ".q_weight") * weights.tensor(
        prefix + ".k_weight"
    )
    rstd = torch.rsqrt(value.square().mean(-1) + norm_eps) * torch.rsqrt(
        key.square().mean(-1) + norm_eps
    )
    dot = (value * norm_weight * key).sum(-1) * rstd * width**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        if token_mask.shape != (rows,) or token_mask.dtype != torch.bool:
            raise ValueError("Engram token mask must identify each input row")
        gate = torch.where(token_mask[:, None], gate, 0.0)
    return value + gate[..., None] * added[:, None, :]
