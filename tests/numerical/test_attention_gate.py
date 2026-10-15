"""Protect BF16 gate rounding from compiler fusion and stale graph inputs."""

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir.attention_gate import _native, attention_gate

from development.kernels import oracles
from tests.support.fp64 import close, cpu, specials


def test_attention_gate_reference_schema():
    inputs = (torch.randn(2, 3, 16, dtype=torch.bfloat16),) * 2
    result = torch.library.opcheck(
        _native,
        inputs,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert set(result.values()) == {"SUCCESS"}
    close(_native(*inputs), oracles.attention_gate(*cpu(inputs)), rtol=0.03, atol=0.03)


@pytest.mark.gpu
def test_attention_gate_all_bf16_values_keep_sigmoid_rounding():
    gate = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 256, 256)
    )
    generator = torch.Generator(device="cuda").manual_seed(941)
    input = torch.randn(
        gate.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    input.view(-1)[:3] = torch.tensor(
        [float("inf"), -float("inf"), float("nan")], device="cuda", dtype=input.dtype
    )
    input.view(-1)[0xC100] = 0.390625  # BF16 gate bits0xC100 encode -8.
    expected = input * gate.sigmoid()
    actual = attention_gate(input, gate)
    independent = oracles.attention_gate(*cpu((input, gate)))
    actual_cpu = cpu(actual)
    close(actual_cpu, independent, rtol=0.03, atol=0.03, equal_nan=True)
    for classify in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual_cpu), classify(independent))
    specials(actual_cpu[independent == 0], independent[independent == 0])
    assert actual_cpu.view(-1)[0xC100].item() == 0.0001316070556640625
    # Exact Torch composition is a separate intermediate-storage property.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.gpu
def test_attention_gate_compiled_graph_uses_changed_packed_views():
    from oh_my_vllm.kernels.cuda_backend import compiled

    generator = torch.Generator(device="cuda").manual_seed(947)
    packed = torch.randn(
        7, 24, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    gate = packed[..., 256:]
    input = packed[..., :256]
    unit = compile_forward(attention_gate, unit="test_attention_gate_rounding")
    unit(input, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit(input, gate)
    for _ in range(3):
        packed.copy_(
            torch.randn(
                packed.shape, device="cuda", dtype=packed.dtype, generator=generator
            )
        )
        graph.replay()
        close(
            output,
            oracles.attention_gate(*cpu((input, gate))),
            rtol=0.03,
            atol=0.03,
        )
        torch.testing.assert_close(output, input * gate.sigmoid(), rtol=0, atol=0)
    canary = torch.full((input.numel() + 2,), 17, device="cuda", dtype=input.dtype)
    destination = canary[1:-1].view(input.shape)
    compiled().attention_gate(input, gate, destination)
    close(
        destination,
        oracles.attention_gate(*cpu((input, gate))),
        rtol=0.03,
        atol=0.03,
    )
    torch.testing.assert_close(destination, input * gate.sigmoid(), rtol=0, atol=0)
    assert canary[0].item() == 17 and canary[-1].item() == 17
    with pytest.raises(Exception, match="Check failed"):
        compiled().attention_gate(input, gate, input)
    with pytest.raises(Exception, match="Check failed"):
        compiled().attention_gate(
            input[..., ::2], gate[..., ::2], destination[..., ::2]
        )
