import torch.nn.functional as functional

from .common import rms


def compute(value, residual, weights, prefix):
    summed = value + residual
    normalized = rms(summed, weights.tensor(prefix + "post_attention_layernorm.weight"))
    gate = functional.silu(weights.linear(normalized, prefix + "mlp.gate_proj"))
    upper = weights.linear(normalized, prefix + "mlp.up_proj")
    return weights.linear(gate * upper, prefix + "mlp.down_proj"), summed


def reference(case, inputs, state, weights):
    prefix = f"model.language_model.layers.{case.weights[0]}."
    value, residual = compute(inputs["value"], inputs["residual"], weights, prefix)
    return {"value": value, "residual": residual}
