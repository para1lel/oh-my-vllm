"""The fixed CUDA RMS epsilon is guarded by the supported Qwen checkpoint contract."""

from unittest.mock import patch

import pytest
from oh_my_vllm.models import qwen


def checkpoint_config(epsilon):
    text = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "rms_norm_eps": epsilon,
        "linear_conv_kernel_dim": 4,
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "rope_parameters": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
    }
    return {
        "text_config": text,
        "quantization_config": {"weight_block_size": [128, 128]},
    }


class FakeCheckpoint:
    def __init__(self, epsilon):
        self.config = checkpoint_config(epsilon)

    def tensor(self, name):
        raise RuntimeError(f"weight loading reached: {name}")


@pytest.mark.parametrize("epsilon", [1e-5, 0, None])
def test_qwen_rejects_non_model_rms_epsilon_before_loading_weights(epsilon):
    with (
        patch.object(qwen, "Checkpoint", return_value=FakeCheckpoint(epsilon)),
        pytest.raises(ValueError, match=r"supported Qwen3\.8-27B architecture"),
    ):
        qwen.Qwen("/unused", device="cpu")


def test_qwen_accepts_model_rms_epsilon_and_reaches_weight_loading():
    with (
        patch.object(qwen, "Checkpoint", return_value=FakeCheckpoint(1e-6)),
        pytest.raises(RuntimeError, match="weight loading reached"),
    ):
        qwen.Qwen("/unused", device="cpu")
