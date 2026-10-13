"""A failed replan must preserve every captured metadata address."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from oh_my_vllm.models.qwen import Batch
from oh_my_vllm.worker.prefill_graph import PrefillGraph


@pytest.mark.parametrize("failure", ["shape", "planner"])
def test_failure_preserves_storage_then_valid_replay_succeeds(failure):
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.tokens = torch.zeros(2, dtype=torch.int64)
    batch = Batch(
        *(torch.zeros(2) for _ in range(8)), prefill_sequences=1, prefill_tokens=2
    )
    graph.batch = batch
    captured = torch.zeros(2, dtype=torch.int32)
    graph.attention = SimpleNamespace(query_starts=captured)
    graph.graph = Mock()
    graph.hidden, graph.features = object(), None

    def bad_plan(*args):
        graph.attention.query_starts = torch.ones(3, dtype=torch.int32)
        if failure == "planner":
            raise RuntimeError("planning failed")

    graph.attention.plan = bad_plan
    error = ValueError if failure == "shape" else RuntimeError
    with pytest.raises(error):
        graph.replay(torch.ones(2, dtype=torch.int64), batch, ())
    assert graph.attention.query_starts is captured
    assert not captured.any()
    graph.graph.replay.assert_not_called()

    def good_plan(*args):
        graph.attention.query_starts = torch.tensor([0, 2], dtype=torch.int32)

    graph.attention.plan = good_plan
    assert graph.replay(torch.ones(2, dtype=torch.int64), batch, ()) == (
        graph.hidden,
        None,
    )
    assert graph.attention.query_starts is captured
    assert captured.tolist() == [0, 2]
    graph.graph.replay.assert_called_once_with()


@pytest.mark.parametrize("failing_call", [1, 2])
@pytest.mark.parametrize("capture_features", [False, True])
def test_constructor_restores_fa_and_recurrent_state_after_failure(
    monkeypatch, failing_call, capture_features
):
    import importlib
    from contextlib import nullcontext

    module = importlib.import_module("oh_my_vllm.worker.prefill_graph")
    attention = SimpleNamespace(workspace=None)
    fake_attention = SimpleNamespace(native=True, ragged=False, plan=Mock())
    monkeypatch.setattr(module, "PagedAttention", lambda **kw: fake_attention)
    monkeypatch.setattr(module, "compile_forward", lambda fn, **kw: fn)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kw: nullcontext())
    batch = Batch(
        positions=torch.tensor([0, 1]),
        fa_slots=torch.tensor([784, 785]),
        attention=attention,
        starts=torch.tensor([0, 2]),
        sequence_ids=torch.zeros(2),
        state_reads=torch.tensor([0]),
        state_writes=torch.tensor([-1, 1]),
        final_state_writes=torch.tensor([1]),
        prefill_sequences=1,
        prefill_tokens=2,
    )
    recurrent = (torch.zeros(2, 1), torch.zeros(2, 1))
    fa = torch.zeros(2, 2, 784, 1)
    snapshots = [state.clone() for state in (*recurrent, fa)]
    calls = 0

    def forward(*args):
        nonlocal calls
        calls += 1
        recurrent[0][1] = 7
        recurrent[1][1] = 8
        fa[1, :, :2] = 9
        if calls == failing_call:
            raise RuntimeError("capture failure")
        return torch.ones(2), None

    model = SimpleNamespace(forward=forward, forward_features=forward)
    with pytest.raises(RuntimeError, match="capture failure"):
        PrefillGraph(
            model,
            [recurrent, fa],
            torch.tensor([1, 2]),
            batch,
            (),
            capture_features=capture_features,
        )
    for state, before in zip((*recurrent, fa), snapshots, strict=True):
        assert torch.equal(state, before)
