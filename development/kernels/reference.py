import math

import numpy as np
from oh_my_vllm.kernels.specification import NUMERICAL_TOLERANCES


def bf16_bits(value):
    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    result = (rounded >> 16).astype(np.uint16)
    return np.where(np.isnan(source), (bits >> 16) | 0x40, result).astype(np.uint16)


def bf16(value):
    bits = bf16_bits(value).astype(np.uint32) << 16
    return bits.view(np.float32).astype(np.float64)


def fp8_decode(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> 3) & 15).astype(np.int32)
    mantissa = (bits & 7).astype(np.float64)
    magnitude = np.where(
        exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7)
    )
    result = np.where((bits & 128) != 0, -magnitude, magnitude)
    return np.where((bits & 127) == 127, np.nan, result)


def fp8_encode(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("FP8 reference requires finite values")
    positive = fp8_decode(np.arange(127, dtype=np.uint8))
    magnitude = np.minimum(np.abs(value), 448)
    high = np.minimum(np.searchsorted(positive, magnitude), 126)
    low = np.maximum(high - 1, 0)
    distances = magnitude - positive[low], positive[high] - magnitude
    pick_high = (distances[1] < distances[0]) | (
        (distances[0] == distances[1]) & ((low & 1) != 0)
    )
    return np.where(pick_high, high, low).astype(np.uint8) | (
        np.signbit(value).astype(np.uint8) << 7
    )


def quantize(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] % 128:
        raise ValueError("FP8 group width must be a multiple of 128")
    grouped = value.reshape(*value.shape[:-1], -1, 128)
    scales = (np.maximum(np.max(np.abs(grouped), axis=-1), 1e-10) / 448).astype(
        np.float32
    )
    data = fp8_encode(grouped.astype(np.float32) / scales[..., None])
    return data.reshape(value.shape), scales


def rms(value, weight, *, offset=True, epsilon=1e-6):
    value = np.asarray(value, dtype=np.float64)
    return bf16(
        value
        / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + epsilon)
        * (weight + int(offset))
    )


def residual_rms(value, residual, weight):
    summed = np.asarray(value, dtype=np.float32) + np.asarray(
        residual, dtype=np.float32
    )
    return rms(summed, weight)


def layer_norm(value, weight, bias, epsilon=1e-6):
    value = np.asarray(value, dtype=np.float64)
    centered = value - np.mean(value, axis=-1, keepdims=True)
    return bf16(
        centered
        / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + epsilon)
        * weight
        + bias
    )


def sigmoid(value):
    exponent = np.exp(-np.abs(value))
    return np.where(value >= 0, 1, exponent) / (1 + exponent)


def silu(value):
    return value * sigmoid(value)


def gelu(value, *, approximate=False):
    value = np.asarray(value, dtype=np.float64)
    if approximate:
        return (
            value
            * 0.5
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (value + 0.044715 * value**3)))
        )
    erf = np.fromiter(
        (math.erf(float(v)) for v in (value / np.sqrt(2)).flat), np.float64, value.size
    ).reshape(value.shape)
    return value * 0.5 * (1 + erf)


def linear(value, weight, bias=0):
    return bf16(
        np.asarray(value, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
        + bias
    )


def fp8_linear(data, scales, weight, weight_scales):
    activation = fp8_decode(data) * np.repeat(scales, 128, axis=-1)
    expanded = np.repeat(np.repeat(weight_scales, 128, axis=0), 128, axis=1)
    return linear(activation, fp8_decode(weight) * expanded)


def attention(query, key, value, *, causal=True):
    query, key, value = (
        np.asarray(item, dtype=np.float64) for item in (query, key, value)
    )
    groups = query.shape[1] // key.shape[1]
    key, value = (np.repeat(item, groups, axis=1) for item in (key, value))
    scores = np.einsum("qhd,khd->hqk", query, key) / np.sqrt(query.shape[-1])
    if causal:
        offset = key.shape[0] - query.shape[0]
        mask = (
            np.arange(key.shape[0])[None, :]
            > offset + np.arange(query.shape[0])[:, None]
        )
        scores = np.where(mask[None], -np.inf, scores)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= np.sum(probability, axis=-1, keepdims=True)
    return bf16(np.einsum("hqk,khd->qhd", probability, value))


def gdn(query, key, value, decay, beta, state):
    query, key, value, state = (
        np.asarray(item, dtype=np.float64) for item in (query, key, value, state)
    )
    query = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    key = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    groups = value.shape[1] // query.shape[1]
    query, key = (np.repeat(item, groups, axis=1) for item in (query, key))
    state = state.copy()
    outputs, snapshots = [], []
    for index in range(len(query)):
        state *= np.exp(decay[index])[:, None, None]
        recall = np.einsum("hvk,hk->hv", state, key[index])
        delta = beta[index, :, None] * (value[index] - recall)
        state += delta[:, :, None] * key[index, :, None, :]
        outputs.append(
            bf16(
                np.einsum("hvk,hk->hv", state, query[index]) / np.sqrt(query.shape[-1])
            )
        )
        snapshots.append(bf16(state))
    return np.asarray(outputs), np.asarray(snapshots)


def numerical_error(
    actual, expected, *, state=False, tolerance="bf16", state_tolerance="state"
):
    actual, expected = (
        np.asarray(item, dtype=np.float64) for item in (actual, expected)
    )
    if (
        actual.shape != expected.shape
        or actual.size == 0
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError("numerical check requires equal nonempty finite tensors")
    error = actual - expected
    peak = float(np.max(np.abs(expected)))
    rms_error = float(np.sqrt(np.mean(error * error)))
    rms_reference = float(np.sqrt(np.mean(expected * expected)))
    nrmse = (
        rms_error / rms_reference
        if rms_reference
        else (0 if not rms_error else math.inf)
    )
    maximum = float(np.max(np.abs(error)))
    tolerance_name = state_tolerance if state else tolerance
    limits = NUMERICAL_TOLERANCES[tolerance_name]
    if state:
        maximum_ratio = (
            maximum / (limits["peak"] * peak)
            if peak
            else (0 if maximum == 0 else math.inf)
        )
        relative_ratio = max(nrmse / limits["nrmse"], maximum_ratio)
        tolerance_ratio = min(maximum / limits["ftz_atol"], relative_ratio)
        required_atol = None
    else:
        absolute = np.abs(error)
        tolerance_ratio = float(
            np.max(absolute / (limits["atol"] + limits["rtol"] * np.abs(expected)))
        )
        required_atol = max(
            0.0, float(np.max(absolute - limits["rtol"] * np.abs(expected)))
        )
    return {
        "passed": tolerance_ratio <= 1,
        "nrmse": "infinity" if state and math.isinf(nrmse) else nrmse,
        "max_abs": maximum,
        "reference_peak": peak,
        "max_tolerance_ratio": tolerance_ratio,
        "tolerance_name": tolerance_name,
        "required_atol": required_atol,
        **({"state_ftz_atol": limits["ftz_atol"]} if state else {}),
    }
