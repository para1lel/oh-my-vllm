import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from oh_my_vllm.worker.checkpoint import Checkpoint

from .capture import PROMPT


class Weights:
    def __init__(self, path):
        self.checkpoint = Checkpoint(path)
        self.config = json.loads((Path(path) / "config.json").read_text())[
            "text_config"
        ]

    def tensor(self, name, rows=None):
        array, kind = self.checkpoint.raw_tensor(name)
        if rows is not None:
            array = array[rows]
        raw = torch.from_numpy(np.array(array, copy=True)).to("cuda")
        if kind == "BF16":
            return raw.view(torch.bfloat16).to(torch.float64)
        if kind == "F32":
            return raw.to(torch.float64)
        if kind != "F8_E4M3":
            raise ValueError("unsupported checkpoint weight type")
        exponent = ((raw >> 3) & 15).to(torch.float64)
        mantissa = (raw & 7).to(torch.float64)
        value = torch.where(
            exponent == 0,
            mantissa * 2.0**-9,
            (1 + mantissa / 8) * torch.exp2(exponent - 7),
        )
        value = torch.where((raw & 128) != 0, -value, value)
        if bool(((raw & 127) == 127).any()):
            raise ValueError("checkpoint contains nonfinite FP8 weights")
        scale = self.tensor(name.removesuffix(".weight") + ".weight_scale_inv")
        outer, inner = value.shape
        return (
            value.reshape(outer // 128, 128, inner // 128, 128)
            * scale[:, None, :, None]
        ).reshape(outer, inner)

    def linear(self, value, prefix):
        weight = self.tensor(prefix + ".weight")
        return value @ weight.T


def rms(value, weight, *, offset=True):
    return (
        value
        * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
        * (weight + int(offset))
    )


def rotary(value, start):
    positions = torch.arange(
        start, start + value.shape[0], device="cuda", dtype=torch.float64
    )
    inverse = 10000000.0 ** (-torch.arange(32, device="cuda", dtype=torch.float64) / 32)
    angle = positions[:, None, None] * inverse[None, None, :]
    first, second = value[..., :32], value[..., 32:64]
    return torch.cat(
        (
            first * angle.cos() - second * angle.sin(),
            second * angle.cos() + first * angle.sin(),
            value[..., 64:],
        ),
        dim=-1,
    )


def attention(weights, prefix, value, previous, start):
    query_gate = weights.linear(value, prefix + "q_proj").reshape(-1, 24, 512)
    query, gate = query_gate.split(256, dim=-1)
    key = weights.linear(value, prefix + "k_proj").reshape(-1, 4, 256)
    values = weights.linear(value, prefix + "v_proj").reshape(-1, 4, 256)
    query = rotary(rms(query, weights.tensor(prefix + "q_norm.weight")), start)
    key = rotary(rms(key, weights.tensor(prefix + "k_norm.weight")), start)
    if previous is not None:
        key, values = (
            torch.cat((old, new))
            for old, new in zip(previous, (key, values), strict=True)
        )
    state = (key, values)
    key, values = (
        tensor.repeat_interleave(6, dim=1).transpose(0, 1) for tensor in state
    )
    scores = query.transpose(0, 1) @ key.transpose(1, 2) / 16
    row = torch.arange(start, start + value.shape[0], device="cuda")
    column = torch.arange(start + value.shape[0], device="cuda")
    scores.masked_fill_(column[None, :] > row[:, None], -torch.inf)
    output = (torch.softmax(scores, dim=-1) @ values).transpose(0, 1)
    output = (output * torch.sigmoid(gate)).reshape(-1, 6144)
    return weights.linear(output, prefix + "o_proj"), state


def recurrent(weights, prefix, value, previous):
    projected = weights.linear(value, prefix + "in_proj_qkv")
    gate = weights.linear(value, prefix + "in_proj_z").reshape(-1, 48, 128)
    beta = torch.sigmoid(weights.linear(value, prefix + "in_proj_b"))
    rate = weights.linear(value, prefix + "in_proj_a") + weights.tensor(
        prefix + "dt_bias"
    )
    decay = -weights.tensor(prefix + "A_log").exp() * torch.logaddexp(
        rate, torch.zeros_like(rate)
    )
    if previous is None:
        history = torch.zeros((3, 10240), device="cuda", dtype=torch.float64)
        state = torch.zeros((48, 128, 128), device="cuda", dtype=torch.float64)
    else:
        history, state = previous
    window = torch.cat((history, projected))
    convolution = weights.tensor(prefix + "conv1d.weight").reshape(10240, 4)
    convolved = sum(
        window[tap : tap + value.shape[0]] * convolution[:, tap] for tap in range(4)
    )
    query, key, values = functional.silu(convolved).split((2048, 2048, 6144), dim=-1)
    query, key = (tensor.reshape(-1, 16, 128) for tensor in (query, key))
    query, key = (
        (
            tensor * torch.rsqrt(tensor.square().sum(dim=-1, keepdim=True) + 1e-6)
        ).repeat_interleave(3, dim=1)
        for tensor in (query, key)
    )
    values = values.reshape(-1, 48, 128)
    outputs = []
    for index in range(value.shape[0]):
        state = state * decay[index].exp()[:, None, None]
        recalled = (state @ key[index, :, :, None]).squeeze(-1)
        delta = beta[index, :, None] * (values[index] - recalled)
        state = state + delta[:, :, None] * key[index, :, None, :]
        outputs.append((state @ query[index, :, :, None]).squeeze(-1) / 128**0.5)
    output = rms(
        torch.stack(outputs), weights.tensor(prefix + "norm.weight"), offset=False
    )
    output = (output * functional.silu(gate)).reshape(-1, 6144)
    return weights.linear(output, prefix + "out_proj"), (window[-3:].clone(), state)


def forward(weights, tokens, states=None, start=0):
    value = weights.tensor("model.language_model.embed_tokens.weight", rows=tokens)
    next_states = []
    for index, kind in enumerate(weights.config["layer_types"]):
        prefix = f"model.language_model.layers.{index}."
        normalized = rms(value, weights.tensor(prefix + "input_layernorm.weight"))
        previous = None if states is None else states[index]
        if kind == "linear_attention":
            output, state = recurrent(
                weights, prefix + "linear_attn.", normalized, previous
            )
        elif kind == "full_attention":
            output, state = attention(
                weights, prefix + "self_attn.", normalized, previous, start
            )
        else:
            raise ValueError("unsupported language layer")
        next_states.append(state)
        value = value + output
        normalized = rms(
            value, weights.tensor(prefix + "post_attention_layernorm.weight")
        )
        gate = functional.silu(weights.linear(normalized, prefix + "mlp.gate_proj"))
        upper = weights.linear(normalized, prefix + "mlp.up_proj")
        value = value + weights.linear(gate * upper, prefix + "mlp.down_proj")
        if (index + 1) % 16 == 0:
            print(f"FP64 position {start}: {index + 1}/64 layers", flush=True)
    normalized = rms(value[-1:], weights.tensor("model.language_model.norm.weight"))
    return weights.linear(normalized, "lm_head").squeeze(0), next_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("reference output already exists")
    torch.set_num_threads(8)
    with torch.inference_mode():
        weights = Weights(args.model)
        prefill, states = forward(weights, list(PROMPT))
        token = int(prefill.argmax())
        decode, _ = forward(weights, [token], states, len(PROMPT))
        outputs = {
            name: value.cpu().numpy()
            for name, value in (("prefill", prefill), ("decode", decode))
        }
    if any(
        value.shape != (248320,) or not np.isfinite(value).all()
        for value in outputs.values()
    ):
        raise ValueError("invalid FP64 reference logits")
    metadata = dict(
        reference=dict(
            precision="torch.float64",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        checkpoint=weights.checkpoint.verify_identity(),
        case=dict(tokens=list(PROMPT), decode_token=token, iterations=1),
    )
    with args.output.open("xb") as destination:
        np.savez(destination, metadata=json.dumps(metadata, sort_keys=True), **outputs)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
