"""Target taps expose full residual outputs, in configured layer and row order."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.models.qwen import Qwen


class ResidualLayer:
    def __init__(self, index):
        self.index = index
        self.cache = None

    def forward_residual(self, hidden, batch, cache, residual):
        del batch
        self.cache = cache
        full = hidden if residual is None else hidden + residual
        return full * (self.index + 1), full


def final_norm(hidden, residual, weight):
    full = hidden + residual
    return full, F.normalize(full.float(), dim=-1).to(full.dtype) * weight


def test_target_features_match_full_layer_outputs_and_final_forward():
    model = Qwen.__new__(Qwen)
    model.embedding = torch.tensor([[1, 2], [3, -1], [-2, 4]], dtype=torch.bfloat16)
    model.layers = [ResidualLayer(index) for index in range(6)]
    model.norm = torch.tensor([2, 3], dtype=torch.bfloat16)
    model.feature_layer_ids = (0, 2, 5)
    tokens = torch.tensor([2, 0, 1, 2])
    caches = [object() for _ in model.layers]
    full = F.embedding(tokens, model.embedding)
    expected = []
    for index in range(6):
        full = full * (index + 1) + full
        if index in model.feature_layer_ids:
            expected.append(full.clone())
    with patch("oh_my_vllm.models.qwen.add_rms_norm", final_norm):
        hidden, features = model.forward_features(tokens, object(), caches)
        ordinary = model.forward(tokens, object(), caches)
    torch.testing.assert_close(features, torch.cat(expected, -1), rtol=0, atol=0)
    torch.testing.assert_close(hidden, ordinary, rtol=0, atol=0)
    assert all(
        layer.cache is cache for layer, cache in zip(model.layers, caches, strict=True)
    )
    torch.testing.assert_close(features[0], features[3], rtol=0, atol=0)


def test_target_features_require_explicit_taps():
    model = Qwen.__new__(Qwen)
    model.feature_layer_ids = ()
    with pytest.raises(ValueError, match="feature layers were not configured"):
        model.forward_features(torch.tensor([1]), object(), [])
