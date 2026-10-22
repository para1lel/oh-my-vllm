"""CPU FP64 DSpark mathematics from LOCAL-V41 inference/model.py:1021-1182.

Use project_features(target_features) for mtp.0's projection and normalization.
seed(projected_main) starts at position 0. Further seed chunks require previous
and its contiguous start_pos. Seeding fills only the three main-window caches.
append_main(features, previous=state, start_pos=p) projects raw target features
in window-sized chunks and appends only main KV. previous=None requires p=0.
forward(projected_main, start_pos=p, input_token=t, previous=state) adds one
main row at p>0 and computes five bidirectional draft rows at p+1 through p+5.
The caller must treat state tensors as read-only. Calls never change them.

proposal_tokens can fix five selected IDs for endpoint checks. Otherwise each
row selects the lowest-ID argmax after its preceding-token Markov bias. Returned
positions identify source computation rows only. This module does not assign
accepted or pending token positions, apply probability or confidence transforms,
or implement service sampling, acceptance, RNG, or capacity policy. It omits
activation quantization and all intermediate BF16/FP32 rounding.
"""

import copy
from dataclasses import dataclass

import torch

from .reference_math import (
    hc_collapse,
    hc_expand,
    hc_mixes,
    require_fp64,
    rms,
    rotary,
    sparse_attention,
)
from .reference_model import moe
from .weights import select_rows


@dataclass(frozen=True)
class DSparkState:
    """Main rows [0, length), with KV in absolute-position modulo window slots."""

    length: int
    windows: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class DSparkOutput:
    main_feature_position: int
    input_token: int
    draft_row_positions: tuple[int, ...]
    draft_input_tokens: tuple[int, ...]
    proposal_tokens: tuple[int, ...]
    previous_tokens: tuple[int, ...]
    base_logits: torch.Tensor
    logits: torch.Tensor
    markov_embeddings: torch.Tensor
    raw_confidence: torch.Tensor
    collapsed_hidden: torch.Tensor
    state: DSparkState


class DSpark:
    def __init__(self, weights, config):
        if weights.device.type != "cpu":
            raise ValueError("DSpark reference weights must use the CPU")
        self.weights = weights
        self.config = copy.deepcopy(config["text_config"])
        text = self.config
        if text["num_nextn_predict_layers"] != 3 or text["dspark_block_size"] != 5:
            raise ValueError(
                "DSpark reference requires three stages and five draft rows"
            )
        dimensions = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "hc_mult",
            "head_dim",
            "num_attention_heads",
            "o_groups",
            "o_lora_rank",
            "sliding_window",
            "max_position_embeddings",
            "dspark_markov_rank",
        )
        if any(type(text[name]) is not int or text[name] < 1 for name in dimensions):
            raise ValueError("DSpark dimensions must be positive integers")
        first = text["num_hidden_layers"]
        if (
            not text["dspark_target_layer_ids"]
            or len(text["compress_ratios"]) < first + 3
            or any(text["compress_ratios"][first : first + 3])
            or text["num_attention_heads"] % text["o_groups"]
            or type(text["qk_rope_head_dim"]) is not int
            or not 0 <= text["qk_rope_head_dim"] <= text["head_dim"]
            or text["qk_rope_head_dim"] % 2
        ):
            raise ValueError(
                "invalid DSpark features, attention groups, or rotary dimensions"
            )
        self._tokens([text["dspark_noise_token_id"]], count=1)

    def _tokens(self, tokens, *, count):
        if isinstance(tokens, torch.Tensor) and tokens.device.type != "cpu":
            raise ValueError("DSpark token IDs must use the CPU")
        selected = select_rows(tokens, self.config["vocab_size"])
        if selected.shape != (count,):
            raise ValueError(f"DSpark needs exactly {count} token IDs")
        return tuple(int(token) for token in selected)

    def _rows(self, value, width):
        require_fp64(value)
        if (
            value.device.type != "cpu"
            or value.ndim != 2
            or value.shape[0] == 0
            or value.shape[1] != width
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                "DSpark needs finite nonempty CPU FP64 rows of the configured width"
            )

    def _previous(self, previous, start_pos):
        if type(start_pos) is not int or start_pos < 0:
            raise ValueError("main-feature start_pos must be a nonnegative integer")
        if previous is None:
            if start_pos != 0:
                raise ValueError("seed history must start at position 0")
            return
        text = self.config
        if (
            not isinstance(previous, DSparkState)
            or type(previous.length) is not int
            or previous.length != start_pos
            or len(previous.windows) != 3
        ):
            raise ValueError(
                "previous main-window state must end at the contiguous start_pos"
            )
        for window in previous.windows:
            self._rows(window, text["head_dim"])
            if window.shape[0] != text["sliding_window"]:
                raise ValueError("previous main-window state has the wrong window size")

    def _rotate(self, value, positions, *, inverse=False):
        width = self.config["qk_rope_head_dim"]
        if not width:
            return value
        rotated = rotary(
            value[..., -width:],
            positions,
            base=self.config["rope_theta"],
            inverse=inverse,
        )
        return torch.cat((value[..., :-width], rotated), dim=-1)

    def _kv(self, value, positions, stage):
        prefix = f"mtp.{stage}.attn"
        return self._rotate(
            rms(
                self.weights.linear(value, prefix + ".wkv"),
                self.weights.tensor(prefix + ".kv_norm.weight"),
                eps=self.config["rms_norm_eps"],
            ),
            positions,
        )

    def _windows(self, main, start_pos, previous):
        text = self.config
        window_size = text["sliding_window"]
        length = start_pos + main.shape[0]
        retained = min(main.shape[0], window_size)
        positions = torch.arange(length - retained, length)
        windows = []
        for stage in range(3):
            window = (
                main.new_zeros((window_size, text["head_dim"]))
                if previous is None
                else previous.windows[stage].clone()
            )
            window.index_copy_(
                0, positions % window_size, self._kv(main[-retained:], positions, stage)
            )
            windows.append(window)
        return DSparkState(length, tuple(windows))

    @torch.inference_mode()
    def project_features(self, target_features):
        """Map concatenated target block-input means to normalized main rows."""
        self._rows(
            target_features,
            self.config["hidden_size"] * len(self.config["dspark_target_layer_ids"]),
        )
        return rms(
            self.weights.linear(target_features, "mtp.0.main_proj"),
            self.weights.tensor("mtp.0.main_norm.weight"),
            eps=self.config["rms_norm_eps"],
        )

    @torch.inference_mode()
    def seed(self, projected_main, *, start_pos=0, previous=None):
        """Fill main KV only; contiguous chunks have the same ring-slot result."""
        self._rows(projected_main, self.config["hidden_size"])
        self._previous(previous, start_pos)
        if start_pos + projected_main.shape[0] > self.config["max_position_embeddings"]:
            raise ValueError("main-feature seed exceeds the source position limit")
        return self._windows(projected_main, start_pos, previous)

    @torch.inference_mode()
    def append_main(self, features, *, previous=None, start_pos=0):
        """Project raw target features and append contiguous main-window KV only."""
        text = self.config
        self._rows(features, text["hidden_size"] * len(text["dspark_target_layer_ids"]))
        self._previous(previous, start_pos)
        if start_pos + features.shape[0] > text["max_position_embeddings"]:
            raise ValueError("main-feature append exceeds the source position limit")
        state = previous
        chunk_size = text["sliding_window"]
        for begin in range(0, features.shape[0], chunk_size):
            projected = self.project_features(features[begin : begin + chunk_size])
            state = self.seed(projected, start_pos=start_pos + begin, previous=state)
        return state

    def _attention(self, value, window, *, start_pos, stage):
        text, weights = self.config, self.weights
        prefix = f"mtp.{stage}.attn"
        positions = torch.arange(start_pos + 1, start_pos + 6)
        query_latent = rms(
            weights.linear(value, prefix + ".wq_a"),
            weights.tensor(prefix + ".q_norm.weight"),
            eps=text["rms_norm_eps"],
        )
        query = weights.linear(query_latent, prefix + ".wq_b").reshape(
            5, text["num_attention_heads"], text["head_dim"]
        )
        query = self._rotate(query, positions)
        draft_kv = self._kv(value, positions, stage)
        cache = torch.cat((window, draft_kv))
        selected = torch.cat(
            (
                torch.arange(min(text["sliding_window"], start_pos + 1)),
                text["sliding_window"] + torch.arange(5),
            )
        )
        output = sparse_attention(
            query,
            cache,
            weights.tensor(prefix + ".attn_sink"),
            [selected] * 5,
            scale=text["head_dim"] ** -0.5,
        )
        output = self._rotate(output, positions, inverse=True)
        grouped = output.reshape(5, text["o_groups"], -1)
        projection = weights.tensor(prefix + ".wo_a.weight").reshape(
            text["o_groups"], text["o_lora_rank"], -1
        )
        output = torch.einsum("sgd,grd->sgr", grouped, projection).flatten(1)
        del projection
        return weights.linear(output, prefix + ".wo_b")

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

    def _block(self, value, pre_mix, window, start_pos, stage):
        prefix = f"mtp.{stage}"
        eps = self.config["rms_norm_eps"]
        attn_pre, post, comb = self._mixes(value, prefix, "attn")
        normalized = rms(
            hc_collapse(value, pre_mix),
            self.weights.tensor(prefix + ".attn_norm.weight"),
            eps=eps,
        )
        output = self._attention(normalized, window, start_pos=start_pos, stage=stage)
        value = hc_expand(output, value, post, comb)
        ffn_pre, post, comb = self._mixes(value, prefix, "ffn")
        normalized = rms(
            hc_collapse(value, attn_pre),
            self.weights.tensor(prefix + ".ffn_norm.weight"),
            eps=eps,
        )
        output = moe(
            self.weights,
            prefix + ".ffn",
            normalized,
            config=self.config,
            layer=self.config["num_hidden_layers"] + stage,
        )
        return hc_expand(output, value, post, comb), ffn_pre

    def _head(self, collapsed, input_token, forced):
        weights = self.weights
        base = weights.linear(
            rms(
                collapsed,
                weights.tensor("mtp.2.norm.weight"),
                eps=self.config["rms_norm_eps"],
            ),
            "head",
        )
        logits, embeddings, previous_tokens, proposals = [], [], [], []
        previous_token = input_token
        for row in range(5):
            previous_tokens.append(previous_token)
            embedding = weights.tensor(
                "mtp.2.markov_head.embed.weight", rows=[previous_token]
            )
            logits.append(
                base[row] + weights.linear(embedding, "mtp.2.markov_head.head")[0]
            )
            embeddings.append(embedding[0])
            previous_token = (
                int(torch.argmax(logits[-1])) if forced is None else forced[row]
            )
            proposals.append(previous_token)
        embeddings = torch.stack(embeddings)
        confidence = weights.linear(
            torch.cat((collapsed, embeddings), dim=-1), "mtp.2.confidence_head.proj"
        ).squeeze(-1)
        return (
            base,
            torch.stack(logits),
            embeddings,
            confidence,
            tuple(previous_tokens),
            tuple(proposals),
        )

    @torch.inference_mode()
    def forward(
        self, projected_main, *, start_pos, input_token, previous, proposal_tokens=None
    ):
        """Append one main row and return five source-row endpoints and new state."""
        self._rows(projected_main, self.config["hidden_size"])
        self._previous(previous, start_pos)
        if previous is None or start_pos == 0 or projected_main.shape[0] != 1:
            raise ValueError(
                "a proposal needs seeded state and one main row at start_pos>0"
            )
        if start_pos + 6 > self.config["max_position_embeddings"]:
            raise ValueError("draft rows exceed the source rotary position limit")
        input_token = self._tokens([input_token], count=1)[0]
        forced = (
            None if proposal_tokens is None else self._tokens(proposal_tokens, count=5)
        )
        state = self._windows(projected_main, start_pos, previous)
        draft_tokens = (input_token,) + (self.config["dspark_noise_token_id"],) * 4
        value = self.weights.tensor("embed.weight", rows=list(draft_tokens))
        value = value[:, None, :].expand(-1, self.config["hc_mult"], -1).clone()
        pre_mix = value.new_zeros((5, self.config["hc_mult"]))
        pre_mix[:, 0] = 1.0
        for stage in range(3):
            value, pre_mix = self._block(
                value, pre_mix, state.windows[stage], start_pos, stage
            )
        collapsed = hc_collapse(value, pre_mix)
        base, logits, embeddings, confidence, previous_tokens, proposals = self._head(
            collapsed, input_token, forced
        )
        return DSparkOutput(
            main_feature_position=start_pos,
            input_token=input_token,
            draft_row_positions=tuple(range(start_pos + 1, start_pos + 6)),
            draft_input_tokens=draft_tokens,
            proposal_tokens=proposals,
            previous_tokens=previous_tokens,
            base_logits=base,
            logits=logits,
            markov_embeddings=embeddings,
            raw_confidence=confidence,
            collapsed_hidden=collapsed,
            state=state,
        )
