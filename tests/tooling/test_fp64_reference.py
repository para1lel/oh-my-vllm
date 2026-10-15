"""Independent equations, precision-sensitive checks, and cached model composition."""

import builtins
import json
import math

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from development import fp64_math as ref
from development.qwen_fp64_reference import Cache, Model, Weights


def test_fp64_normalization_and_checkpoint_decode_preserve_low_bits():
    x = torch.tensor([[1 + 2**-30, 2 - 2**-29]], dtype=torch.float64)
    weight = torch.tensor([1 + 2**-30, 1.0], dtype=torch.float64)
    expected = x * weight / ((x.square().mean(-1, keepdim=True) + 1e-6).sqrt())
    torch.testing.assert_close(ref.rms(x, weight), expected, atol=4e-16, rtol=0)
    assert not torch.equal(ref.rms(x.float(), weight.float()), expected)
    raw = torch.ones(129, 130).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1 + 2**-30, 2], [3, 4]], dtype=torch.float64)
    decoded = ref.decode_weight(raw, scales)
    assert decoded.dtype == torch.float64
    assert decoded[0, 0].item() == 1 + 2**-30
    assert decoded[128, 129].item() == 4


@pytest.mark.parametrize("first", [0, 5])
def test_block_attention_matches_dense_fp64_with_causal_gqa(first):
    generator = torch.Generator().manual_seed(407)
    q = torch.randn(7, 4, 6, generator=generator, dtype=torch.float64) * 20
    k = torch.randn(first + 7, 2, 6, generator=generator, dtype=torch.float64)
    v = torch.randn(first + 7, 2, 6, generator=generator, dtype=torch.float64)
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, 1)) / math.sqrt(6)
    visible = torch.arange(first + 7)[None] <= (torch.arange(7) + first)[:, None]
    scores.masked_fill_(~visible, -torch.inf)
    expected = torch.einsum(
        "hts,shd->thd", scores.softmax(-1), v.repeat_interleave(2, 1)
    )
    actual = ref.attention(q, k, v, first_position=first, query_block=3, key_block=2)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("tokens", [0, 1, 7, 65])
@pytest.mark.parametrize("decay", [0.0, -0.2, -1000.0])
def test_chunk_delta_matches_direct_fp64_recurrence_with_carry(tokens, decay):
    generator = torch.Generator().manual_seed(408)
    q, k = (
        torch.randn(tokens, 2, 4, generator=generator, dtype=torch.float64)
        for _ in range(2)
    )
    v = torch.randn(tokens, 6, 3, generator=generator, dtype=torch.float64)
    initial = torch.randn(6, 3, 4, generator=generator, dtype=torch.float64)
    log_decay = torch.full((tokens, 6), decay, dtype=torch.float64)
    beta = torch.rand(tokens, 6, generator=generator, dtype=torch.float64)
    actual = ref.delta_chunk(q, k, v, log_decay, beta, initial, chunk=5)
    expected = ref.delta_recurrence(q, k, v, log_decay, beta, initial)
    for result, target in zip(actual, expected, strict=True):
        assert result.dtype == torch.float64
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, target, atol=3e-14, rtol=3e-14)


def test_convolution_matches_fp64_depthwise_definition_and_continuation():
    generator = torch.Generator().manual_seed(409)
    x = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    kernel = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    history = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    expected = F.conv1d(torch.cat((history, x.T), -1)[None], kernel[:, None], groups=3)[
        0
    ].T
    actual, final = ref.causal_conv(x, kernel, history)
    torch.testing.assert_close(actual, F.silu(expected), rtol=2e-15, atol=2e-15)
    first, carry = ref.causal_conv(x[:2], kernel, history)
    second, end = ref.causal_conv(x[2:], kernel, carry)
    assert torch.equal(torch.cat((first, second)), actual)
    assert torch.equal(end, final)


def test_distribution_metrics_normalize_before_any_narrowing():
    logits = torch.tensor([[1 + 2**-30, 1.0]], dtype=torch.float64)
    candidate = torch.zeros_like(logits)
    metrics = ref.distribution_metrics(logits, candidate, torch.tensor([0]))
    expected_nll = torch.logsumexp(logits, -1) - logits[:, 0]
    torch.testing.assert_close(
        metrics["reference_nll"], expected_nll, atol=4e-16, rtol=0
    )
    assert metrics["reference_nll"].item() != math.log(2)
    assert all(value.dtype == torch.float64 for value in metrics.values())


def test_delta_gates_use_fp64_softplus_without_fp32_threshold_approximation():
    a = torch.tensor([-1000.0, -21.0, 21.0, 1000.0], dtype=torch.float64)
    decay, beta = ref.delta_gates(
        torch.zeros_like(a), a, torch.zeros_like(a), torch.zeros_like(a)
    )
    assert decay[2].item() == -math.log1p(math.exp(21))
    assert decay[2].item() != -21
    assert torch.isfinite(decay).all()
    assert torch.equal(beta, torch.full_like(a, 0.5))


def test_fp8_encoder_uses_fp64_distances_and_even_ties_without_float32_cast():
    midpoint = 1.0625
    values = torch.tensor(
        [midpoint - 2**-40, midpoint, midpoint + 2**-40, -midpoint, -0.0],
        dtype=torch.float64,
    )
    encoded = ref.encode_fp8(values)
    assert encoded.double().tolist() == [1.0, 1.0, 1.125, -1.0, -0.0]
    assert encoded.view(torch.uint8)[-1].item() == 128


def test_bf16_storage_rounds_directly_from_fp64_at_midpoint_and_subnormal():
    midpoint = 1 + 2**-8
    values = torch.tensor(
        [
            midpoint - 2**-40,
            midpoint,
            midpoint + 2**-40,
            -midpoint - 2**-40,
            2**-134,
            3 * 2**-134,
            -0.0,
            float("inf"),
            -float("inf"),
        ],
        dtype=torch.float64,
    )
    stored = ref.store(values, torch.bfloat16)
    assert stored.double().tolist() == [
        1.0,
        1.0,
        1 + 2**-7,
        -1 - 2**-7,
        0.0,
        2 * 2**-133,
        -0.0,
        float("inf"),
        -float("inf"),
    ]
    assert stored.view(torch.int16)[6].item() == -32768


def _checkpoint(tmp_path):
    generator = torch.Generator().manual_seed(410)
    tensors = {}

    def raw(name, shape, *, norm=False):
        value = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.05
        tensors[name] = value + 1 if norm else value

    def linear(name, rows, columns, *, quantized=True):
        raw(name + ".weight", (rows, columns))
        if quantized:
            tensors[name + ".weight"] = (tensors[name + ".weight"] * 32).to(
                torch.float8_e4m3fn
            )
            tensors[name + ".weight_scale_inv"] = torch.full(
                ((rows + 127) // 128, (columns + 127) // 128),
                1 / 32,
                dtype=torch.float64,
            )

    config = {
        "hidden_act": "silu",
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.5,
        },
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
    }
    raw("model.language_model.embed_tokens.weight", (23, 8))
    raw("model.language_model.norm.weight", (8,))
    linear("lm_head", 23, 8, quantized=False)
    for layer, kind in enumerate(config["layer_types"]):
        prefix = f"model.language_model.layers.{layer}"
        raw(prefix + ".input_layernorm.weight", (8,))
        raw(prefix + ".post_attention_layernorm.weight", (8,))
        for name, rows, columns in (
            ("gate_proj", 12, 8),
            ("up_proj", 12, 8),
            ("down_proj", 8, 12),
        ):
            linear(prefix + ".mlp." + name, rows, columns)
        if kind == "full_attention":
            p = prefix + ".self_attn"
            for name, rows, columns in (
                ("q_proj", 16, 8),
                ("k_proj", 4, 8),
                ("v_proj", 4, 8),
                ("o_proj", 8, 8),
            ):
                linear(p + "." + name, rows, columns)
            raw(p + ".q_norm.weight", (4,))
            raw(p + ".k_norm.weight", (4,))
        else:
            p = prefix + ".linear_attn"
            for name, rows, columns in (
                ("in_proj_qkv", 16, 8),
                ("in_proj_z", 8, 8),
                ("in_proj_b", 2, 8),
                ("in_proj_a", 2, 8),
                ("out_proj", 8, 8),
            ):
                linear(
                    p + "." + name,
                    rows,
                    columns,
                    quantized=name not in ("in_proj_b", "in_proj_a"),
                )
            raw(p + ".conv1d.weight", (16, 1, 4))
            raw(p + ".A_log", (2,))
            raw(p + ".dt_bias", (2,))
            raw(p + ".norm.weight", (4,), norm=True)
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    return Weights(tmp_path, projection_rows=128)


def test_independent_model_prefill_and_continuation_compose_in_fp64(
    tmp_path, monkeypatch
):
    original_import = builtins.__import__

    def independent_import(name, *args, **kwargs):
        assert not name.startswith("oh_my_vllm"), (
            "reference called an inference provider"
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", independent_import)
    model = Model(_checkpoint(tmp_path))
    ids = torch.tensor([0, 3, 4, 7, 2, 8, 9])
    whole, full_cache = model.forward(ids)
    first, cache = model.forward(ids[:3])
    second, cache = model.forward(ids[3:], cache)
    torch.testing.assert_close(
        torch.cat((first, second)), whole, atol=1e-14, rtol=1e-14
    )
    torch.testing.assert_close(
        model.logits(second), model.logits(whole[3:]), atol=1e-14, rtol=1e-14
    )
    assert cache.length == len(ids)
    for index, (history, state) in cache.delta.items():
        assert history.dtype == state.dtype == torch.float64
        for actual, expected in zip(
            (history, state), full_cache.delta[index], strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    for key, value in cache.attention.values():
        assert key.dtype == value.dtype == torch.float64
    with pytest.raises(ValueError, match="small comparisons"):
        model.forward(torch.zeros(513, dtype=torch.int64), Cache())


def test_reference_rejects_cuda_alias_before_reading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="explicit CUDA device index"):
        Weights(tmp_path / "missing-checkpoint", device="cuda")
