"""The GDN fork joins all work and preserves cache and replay semantics."""

from contextlib import contextmanager

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir.execution import execution_policy
from oh_my_vllm.ir.gdn_prepare import (
    _fake_prepare,
    _kernel_prepare,
    _native_prepare,
    gdn_prepare,
)


def inputs(rows, device):
    # Smaller input width keeps the scheduling/schema cases independent of the
    # 27B checkpoint. All GDN output/state widths match production.
    width = 256 if device == "cuda" else 128
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    weights = torch.randn(16384, width, device=device).to(torch.float8_e4m3fn)
    scales = torch.full((128, width // 128), 0.03, device=device)
    ba = torch.randn(96, width, device=device, dtype=torch.bfloat16) * 0.02
    conv = torch.randn(10240, 4, device=device, dtype=torch.bfloat16) * 0.05
    pool = torch.zeros(2, 10240, 3, device=device, dtype=torch.bfloat16)
    ids = torch.zeros(rows, device=device, dtype=torch.int64)
    starts = torch.tensor([0, rows], device=device, dtype=torch.int32)
    reads = torch.zeros(1, device=device, dtype=torch.int64)
    writes = torch.full((rows,), -1, device=device, dtype=torch.int64)
    writes[-1] = 1
    log = torch.linspace(-1, 1, 48, device=device)
    bias = torch.linspace(-0.1, 0.1, 48, device=device)
    return x, weights, scales, ba, conv, pool, ids, starts, reads, writes, log, bias


@contextmanager
def policy(monkeypatch, enabled):
    with monkeypatch.context() as context:
        context.setenv("OH_MY_VLLM_MULTI_STREAM", str(int(enabled)))
        execution_policy.cache_clear()
        try:
            yield
        finally:
            execution_policy.cache_clear()


def test_native_prepare_declares_mutation_and_outputs():
    args = list(inputs(2, "cpu"))
    # opcheck poisons tensor inputs with multiplication, which CPU FP8 lacks.
    # The reference expands checkpoint weights to FP32 before its arithmetic.
    args[1] = args[1].float()
    result = torch.library.opcheck(
        _native_prepare,
        args,
        test_utils=("test_schema", "test_faketensor"),
    )
    assert set(result.values()) == {"SUCCESS"}
    _native_prepare(*args)
    assert not args[5][0].count_nonzero()
    assert args[5][1].count_nonzero()


def test_prepare_fake_rejects_incompatible_projection():
    args = list(inputs(1, "meta"))
    args[3] = torch.empty(48, 128, device="meta")
    with pytest.raises(RuntimeError):
        _fake_prepare(*args)


@pytest.mark.gpu
@torch.inference_mode()
@pytest.mark.parametrize("rows", [1, 8, 624])
def test_parallel_prepare_matches_serial_outputs_and_state(monkeypatch, rows):
    args = inputs(rows, "cuda")
    with policy(monkeypatch, False):
        expected = _kernel_prepare(*args)
    state = args[5].clone()
    args[5].zero_()
    with policy(monkeypatch, True):
        actual = compile_forward(gdn_prepare)(*args)
    torch.cuda.synchronize()
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(args[5], state, rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_parallel_graph_replay_uses_changed_input_after_allocator_pressure(monkeypatch):
    args = inputs(8, "cuda")
    with policy(monkeypatch, True):
        unit = compile_forward(gdn_prepare)
        unit(*args)
        args[5].zero_()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = unit(*args)
    for iteration in range(5):
        args[0].fill_(iteration * 0.07 - 0.13)
        args[5].zero_()
        with policy(monkeypatch, False):
            expected = tuple(value.clone() for value in _kernel_prepare(*args))
        expected_state = args[5].clone()
        args[5].zero_()
        pressure = [torch.empty(65536, device="cuda") for _ in range(32)]
        del pressure
        graph.replay()
        torch.cuda.synchronize()
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(args[5], expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("branch", ["gates", "convolution"])
def test_side_stream_is_joined_when_a_branch_raises(monkeypatch, branch):
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    module = importlib.import_module("oh_my_vllm.ir.gdn_prepare")
    origin, side = Mock(), Mock()
    tensor = Mock(device=SimpleNamespace(index=0))
    monkeypatch.setattr(
        module, "execution_policy", lambda: SimpleNamespace(multi_stream=True)
    )
    monkeypatch.setattr(module, "branch_stream", lambda index: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: origin)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(module.F, "linear", lambda *args: tensor)
    monkeypatch.setattr(
        module,
        "delta_gates",
        Mock(
            return_value=(tensor, tensor),
            side_effect=RuntimeError("gates") if branch == "gates" else None,
        ),
    )
    packed = Mock()
    packed.split.return_value = tensor, tensor
    monkeypatch.setattr(module, "linear", lambda *args: packed)
    monkeypatch.setattr(
        module,
        "causal_conv",
        Mock(
            side_effect=RuntimeError("convolution") if branch == "convolution" else None
        ),
    )
    with pytest.raises(RuntimeError, match=branch):
        _kernel_prepare._init_fn(*([tensor] * 12))
    side.wait_stream.assert_called_once_with(origin)
    origin.wait_stream.assert_called_once_with(side)
