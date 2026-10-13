"""Selected feature APIs and context capture failures preserve their contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from oh_my_vllm.models.qwen import AttentionBatch, Linear, Qwen
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.mtp_context_graph import MTPContextGraph


@pytest.mark.parametrize("taps", [(63,), (5, 63)])
def test_feature_tap_63_preserves_all_rows_with_selected_logits(monkeypatch, taps):
    import oh_my_vllm.models.qwen as module

    model = Qwen.__new__(Qwen)
    model.feature_layer_ids = taps
    model.embedding = torch.zeros(2, 4)
    model.norm = None
    model.layers = [
        SimpleNamespace(
            forward_residual=lambda x, b, c, r: (x + 1, torch.zeros_like(x))
        )
        for _ in range(64)
    ]
    model._selected_final = Mock(
        side_effect=AssertionError("tap63 requires full hidden")
    )
    monkeypatch.setattr(module, "add_rms_norm", lambda x, r, w: (x + r, x + r))
    hidden, features = model.forward_features(
        torch.zeros(3, dtype=torch.int64),
        SimpleNamespace(output_indices=torch.tensor([2])),
        [None] * 64,
    )
    torch.testing.assert_close(hidden, torch.full((3, 4), 64.0))
    torch.testing.assert_close(
        features, torch.cat([torch.full((3, 4), float(tap + 1)) for tap in taps], -1)
    )
    model._selected_final.assert_not_called()


@pytest.mark.parametrize("failing_call", [1, 2])
def test_context_capture_restores_all_fa_writes_on_failure(monkeypatch, failing_call):
    import oh_my_vllm.worker.mtp_context_graph as module

    monkeypatch.setattr(module, "compile_forward", lambda fn, **kw: fn)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: nullcontext())
    cache = torch.zeros(2, 2, 784, 4, 256, dtype=torch.bfloat16)
    before = cache.clone()
    calls = 0

    def context(*args):
        nonlocal calls
        calls += 1
        cache[1, :, :2] = 7
        if calls == failing_call:
            raise RuntimeError("capture failure")
        return torch.zeros(0, 5120)

    with pytest.raises(RuntimeError, match="capture failure"):
        MTPContextGraph(
            SimpleNamespace(draft_context=context),
            cache,
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, 5120),
            AttentionBatch(torch.arange(2), torch.tensor([784, 785]), None),
            torch.zeros(0, dtype=torch.int64),
            None,
        )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


def test_mtp_context_snapshot_reports_capture_and_hit():
    cache = GraphCache(capacity=1, synchronize=lambda: None)
    assert cache.should_use("mtp_context", (1,))
    cache.get_or_create("mtp_context", (1,), object)
    assert cache.should_use("mtp_context", (1,))
    snapshot = cache.snapshot()
    assert snapshot["mtp_context_resident"] == 1
    assert snapshot["mtp_context_capture"] == 1
    assert snapshot["mtp_context_hit"] == 1


def test_projection_views_preserve_checkpoint_scale_block_alignment():
    weight = torch.empty(256, 128, dtype=torch.float8_e4m3fn)
    scale = torch.arange(2, dtype=torch.float32).view(2, 1)
    linear = Linear(weight, scale)
    view = linear.rows(128, 256)
    assert view.weight.data_ptr() == weight[128:].data_ptr()
    torch.testing.assert_close(view.scale, scale[1:])
    for first, last in ((1, 129), (128, 255), (-128, 128), (128, 384)):
        with pytest.raises(ValueError, match="scale blocks"):
            linear.rows(first, last)
