"""CPU contracts for DSpark loading, published YaRN, and learned heads."""

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.models.dspark import (
    DSparkCheckpoint,
    DSparkConfig,
    DSparkModel,
    expected_shapes,
    rms_norm,
    yarn_frequencies,
)
from safetensors.torch import save_file
from transformers import Qwen3Config
from transformers.modeling_rope_utils import _compute_yarn_parameters


def tiny_config():
    return DSparkConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=11,
        markov_rank=2,
        mask_token_id=10,
        target_layer_ids=(0, 1),
    )


def config_dict():
    config = DSparkConfig()
    raw = {
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
        "layer_types": ["full_attention"] * 5,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 10000000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    }
    nested = {
        name: raw[name]
        for name in (
            "target_layer_ids",
            "mask_token_id",
            "projector_type",
            "markov_head_type",
            "markov_rank",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "attention_mode",
        )
    }
    raw["dspark_config"] = dict(nested)
    raw["dflash_config"] = dict(nested)
    return raw


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": 8},
        {"markov_rank": 0},
        {"dtype": "float16"},
        {"attention_bias": True},
        {"confidence_head_with_markov": False},
        {"target_layer_ids": [5, 19, 33, 47, 62]},
    ],
)
def test_unsupported_fixed_checkpoint_config_is_rejected(tmp_path, change):
    raw = config_dict()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    assert DSparkConfig.read(tmp_path) == DSparkConfig()
    raw.update(change)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported DSpark config"):
        DSparkConfig.read(tmp_path)


def test_nested_config_and_rope_disagreement_is_rejected(tmp_path):
    raw = config_dict()
    raw["dspark_config"]["markov_head_type"] = "other"
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="inconsistent"):
        DSparkConfig.read(tmp_path)
    raw = config_dict()
    raw["rope_parameters"]["factor"] = 4
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="YaRN"):
        DSparkConfig.read(tmp_path)


def test_production_header_shape_contract_and_persistent_memory():
    shapes = expected_shapes(DSparkConfig())
    assert len(shapes) == 62
    assert (
        sum(__import__("math").prod(shape) for shape in shapes.values()) == 1857358337
    )
    config = DSparkConfig()
    bytes_per_token = (
        config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 2
    )
    assert bytes_per_token == 20480
    assert bytes_per_token * 262144 == 5 * 1024**3


@pytest.mark.parametrize("defect", ["missing", "extra", "shape", "dtype"])
def test_header_defects_fail_before_device_transfer(tmp_path, defect):
    config = tiny_config()
    (tmp_path / "config.json").write_text("{}")
    tensors = {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in expected_shapes(config).items()
    }
    if defect == "missing":
        tensors.pop("fc.weight")
    elif defect == "extra":
        tensors["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif defect == "shape":
        tensors["fc.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    else:
        tensors["fc.weight"] = tensors["fc.weight"].float()
    save_file(tensors, tmp_path / "model.safetensors")
    with (
        patch.object(DSparkConfig, "read", return_value=config),
        pytest.raises(ValueError),
    ):
        DSparkCheckpoint(tmp_path, "cuda")


def test_yarn_matches_official_frequency_parameters_bit_for_bit():
    config = DSparkConfig()
    official = Qwen3Config(
        hidden_size=5120,
        num_attention_heads=32,
        head_dim=128,
        max_position_embeddings=262144,
        rope_parameters=config_dict()["rope_parameters"],
    )
    expected, factor = _compute_yarn_parameters(official, device=torch.device("cpu"))
    actual, actual_factor = yarn_frequencies(config, "cpu")
    assert torch.equal(actual, expected)
    assert actual_factor == factor


def test_rms_norm_preserves_bf16_rounding_before_multiplier():
    generator = torch.Generator().manual_seed(981)
    x = torch.randn(17, 8, generator=generator).bfloat16()
    weight = torch.randn(8, generator=generator).bfloat16()
    normalized = (
        x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    ).bfloat16()
    actual = rms_norm(x, weight.float(), 1e-6)
    assert torch.equal(actual, normalized * weight)
    fused_rounding = (
        x.float()
        * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * weight.float()
    ).bfloat16()
    assert not torch.equal(actual, fused_rounding)


def load_tiny_model():
    config = tiny_config()
    generator = torch.Generator().manual_seed(12)
    tensors = {
        name: (torch.randn(shape, generator=generator) * 0.1).bfloat16()
        for name, shape in expected_shapes(config).items()
    }
    target_embedding = torch.randn(
        config.vocab_size, config.hidden_size, generator=generator
    ).bfloat16()
    head = torch.randn(
        config.vocab_size, config.hidden_size, generator=generator
    ).bfloat16()
    target = SimpleNamespace(
        embedding=target_embedding, logits=lambda hidden: F.linear(hidden, head)
    )
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text("{}")
        save_file(tensors, Path(directory) / "model.safetensors")
        with patch.object(DSparkConfig, "read", return_value=config):
            model = DSparkModel(target, directory)
    return model, tensors


def test_loaded_checkpoint_provenance_hashes_exact_source_files(caplog):
    with caplog.at_level("INFO", logger="oh_my_vllm.models.dspark"):
        model, _ = load_tiny_model()
    assert model.provenance["config_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert len(model.provenance["weights_sha256"]["model.safetensors"]) == 64
    assert "DSPARK_CHECKPOINT_PROVENANCE" in caplog.text


@pytest.mark.parametrize("change", ["config", "weights", "replace"])
def test_checkpoint_changes_are_rejected_before_reporting_provenance(tmp_path, change):
    config = tiny_config()
    (tmp_path / "config.json").write_text("{}")
    tensors = {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in expected_shapes(config).items()
    }
    save_file(tensors, tmp_path / "model.safetensors")
    with patch.object(DSparkConfig, "read", return_value=config):
        checkpoint = DSparkCheckpoint(tmp_path, "cpu")
    identity = checkpoint.provenance()
    expected = hashlib.sha256((tmp_path / "model.safetensors").read_bytes()).hexdigest()
    assert identity["weights_sha256"] == {"model.safetensors": expected}
    if change == "config":
        (tmp_path / "config.json").write_text('{"changed":true}')
    elif change == "weights":
        with (tmp_path / "model.safetensors").open("ab") as stream:
            stream.write(b"changed")
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes((tmp_path / "model.safetensors").read_bytes())
        replacement.replace(tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="changed during loading"):
        checkpoint.provenance()
    with pytest.raises(RuntimeError, match="changed during loading"):
        checkpoint.tensor("fc.weight")


def test_loaded_markov_and_confidence_heads_use_previous_candidate():
    model, tensors = load_tiny_model()
    hidden = torch.arange(16).view(2, 8).bfloat16() * 0.1
    base = torch.arange(22).view(2, 11).bfloat16() * 0.1
    previous = torch.tensor([2, 7])
    actual_logits, actual_confidence = model.step(hidden, base, previous)
    latent = tensors["markov_head.markov_w1.weight"][previous]
    expected_logits = base + latent @ tensors["markov_head.markov_w2.weight"].T
    expected_confidence = (
        F.linear(
            torch.cat((hidden, latent), -1),
            tensors["confidence_head.proj.weight"],
            tensors["confidence_head.proj.bias"],
        )
        .float()
        .sigmoid()
        .squeeze(-1)
    )
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_confidence, expected_confidence)


def test_injection_projects_each_feature_at_its_own_position():
    import sys
    from types import ModuleType

    model, tensors = load_tiny_model()
    calls = []
    kernels = ModuleType("oh_my_vllm.kernels.dspark_attention")
    kernels.prepare_qk = lambda q, k, *_: (q, k)
    kernels.append = lambda cache, k, v, slots: calls.append((cache, k, v, slots))
    features = torch.arange(48).view(3, 16).bfloat16() * 0.01
    positions, slots = torch.tensor([782, 783, 784]), torch.tensor([782, 783, 1568])
    caches = [object(), object()]
    with patch.dict(sys.modules, {kernels.__name__: kernels}):
        model.inject(features, positions, slots, caches)
    context = rms_norm(
        F.linear(features, tensors["fc.weight"]), tensors["hidden_norm.weight"], 1e-6
    )
    for index, (cache, key, value, actual_slots) in enumerate(calls):
        assert cache is caches[index]
        assert torch.equal(actual_slots, slots)
        prefix = f"layers.{index}.self_attn."
        assert torch.equal(
            key, F.linear(context, tensors[prefix + "k_proj.weight"]).view(3, 1, 4)
        )
        assert torch.equal(
            value, F.linear(context, tensors[prefix + "v_proj.weight"]).view(3, 1, 4)
        )


def test_eager_backbone_matches_independent_dual_source_attention_golden():
    import sys
    from types import ModuleType

    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    model, weights = load_tiny_model()
    config = model.config
    generator = torch.Generator().manual_seed(112)
    features = torch.randn(3, config.feature_size, generator=generator).bfloat16()
    tokens = torch.tensor([[2, 10, 10, 10, 10, 10, 10]])
    positions = torch.arange(3, 10)[None]
    tables = torch.tensor([[1]], dtype=torch.int32)
    lengths = torch.tensor([3], dtype=torch.int32)
    caches = [torch.zeros(3, 2, 784, 1, 4, dtype=torch.bfloat16) for _ in range(2)]

    def official_norm(values, weight):
        norm = Qwen3RMSNorm(values.shape[-1], eps=1e-6).bfloat16()
        norm.weight.data.copy_(weight.bfloat16())
        return norm(values)

    def rotate(values, at_positions):
        angles = at_positions.double()[..., None] * model.inv_freq.double()
        cosine = (angles.cos() * model.attention_factor).bfloat16()[..., None, :]
        sine = (angles.sin() * model.attention_factor).bfloat16()[..., None, :]
        first, second = values.chunk(2, -1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), -1
        )

    def prepare(q, k, q_weight, k_weight, at_positions, *_):
        q = None if q is None else rotate(official_norm(q, q_weight), at_positions)
        k = None if k is None else rotate(official_norm(k, k_weight), at_positions)
        return q, k

    def append(cache, keys, values, slots):
        cache[slots // 784, 0, slots % 784] = keys
        cache[slots // 784, 1, slots % 784] = values

    def attention(q, cache, page_table, context_lengths, block_k, block_v):
        rows = []
        for request, count in enumerate(context_lengths.tolist()):
            at = torch.arange(count)
            pages = page_table[request, at // 784].long()
            context_k = cache[pages, 0, at % 784]
            context_v = cache[pages, 1, at % 784]
            keys = torch.cat((context_k, block_k[request])).repeat_interleave(2, 1)
            values = torch.cat((context_v, block_v[request])).repeat_interleave(2, 1)
            scores = (
                torch.einsum("qhd,khd->hqk", q[request].double(), keys.double()) / 2
            )
            output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values.double())
            rows.append(output.bfloat16())
        return torch.stack(rows)

    kernels = ModuleType("oh_my_vllm.kernels.dspark_attention")
    kernels.prepare_qk, kernels.append, kernels.attention = prepare, append, attention
    with patch.dict(sys.modules, {kernels.__name__: kernels}):
        model.inject(features, torch.arange(3), torch.arange(784, 787), caches)
        before = [cache.clone() for cache in caches]
        actual = model.forward(tokens, positions, tables, lengths, caches)

    context = official_norm(
        F.linear(features, weights["fc.weight"]), weights["hidden_norm.weight"]
    )
    golden = F.embedding(tokens, model.target.embedding)
    for index in range(2):
        prefix = f"layers.{index}."
        normalized = official_norm(golden, weights[prefix + "input_layernorm.weight"])
        q = F.linear(normalized, weights[prefix + "self_attn.q_proj.weight"]).view(
            1, 7, 2, 4
        )
        q = rotate(
            official_norm(q, weights[prefix + "self_attn.q_norm.weight"]), positions
        )
        block_k = F.linear(
            normalized, weights[prefix + "self_attn.k_proj.weight"]
        ).view(1, 7, 1, 4)
        block_k = rotate(
            official_norm(block_k, weights[prefix + "self_attn.k_norm.weight"]),
            positions,
        )
        block_v = F.linear(
            normalized, weights[prefix + "self_attn.v_proj.weight"]
        ).view(1, 7, 1, 4)
        context_k = F.linear(context, weights[prefix + "self_attn.k_proj.weight"]).view(
            3, 1, 4
        )
        context_k = rotate(
            official_norm(context_k, weights[prefix + "self_attn.k_norm.weight"]),
            torch.arange(3),
        )
        context_v = F.linear(context, weights[prefix + "self_attn.v_proj.weight"]).view(
            3, 1, 4
        )
        assert torch.equal(caches[index][1, 0, :3], context_k)
        assert torch.equal(caches[index][1, 1, :3], context_v)
        keys = (
            torch.cat((context_k, block_k[0])).repeat_interleave(2, 1).transpose(0, 1)
        )
        values = (
            torch.cat((context_v, block_v[0])).repeat_interleave(2, 1).transpose(0, 1)
        )
        # All noise rows can see every noise row, plus exactly the three context
        # rows. Padded slots in page 1 and every other page are excluded.
        attended = (
            F.scaled_dot_product_attention(
                q[0].transpose(0, 1).double(),
                keys.double(),
                values.double(),
                is_causal=False,
            )
            .transpose(0, 1)
            .bfloat16()
            .flatten(-2)[None]
        )
        golden = golden + F.linear(
            attended, weights[prefix + "self_attn.o_proj.weight"]
        )
        normalized = official_norm(
            golden, weights[prefix + "post_attention_layernorm.weight"]
        )
        gate = F.silu(F.linear(normalized, weights[prefix + "mlp.gate_proj.weight"]))
        up = F.linear(normalized, weights[prefix + "mlp.up_proj.weight"])
        golden = golden + F.linear(gate * up, weights[prefix + "mlp.down_proj.weight"])
    golden = official_norm(golden, weights["norm.weight"])
    assert torch.equal(actual, golden)
    for original, after in zip(before, caches, strict=True):
        assert torch.equal(original, after)
