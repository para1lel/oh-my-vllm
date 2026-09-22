"""GPU numerical tests for project-owned operators, without importing vLLM.

Run only under scripts/with-gpu.sh. CPU FP64 references consume the rounded
inputs passed to the GPU, matching actual-path probe tolerances.
"""

import math
from itertools import pairwise

import pytest
import torch
from oh_my_vllm.kernels import fp8, gdn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(q, k, v, g, beta, state, *, snapshots=True):
    q, k, v, g, beta, state = [
        t.detach().to(device="cpu", dtype=torch.float64)
        for t in (q, k, v, g, beta, state)
    ]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    repeat = v.shape[1] // q.shape[1]
    q = q.repeat_interleave(repeat, 1) / math.sqrt(q.shape[-1])
    k = k.repeat_interleave(repeat, 1)
    outputs, states = [], []
    for index in range(len(q)):
        state = state * g[index].exp()[:, None, None]
        residual = v[index] - torch.einsum("hvk,hk->hv", state, k[index])
        state = state + torch.einsum(
            "hv,hk->hvk", residual * beta[index, :, None], k[index]
        )
        outputs.append(torch.einsum("hvk,hk->hv", state, q[index]))
        if snapshots:
            states.append(state.clone())
    return torch.stack(outputs), torch.stack(states) if snapshots else state[None]


def check(actual, expected, state=False):
    actual = actual.detach().cpu().double()
    assert torch.isfinite(actual).all()
    if state:
        error = actual - expected
        nrmse = (
            error.square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-10)
        )
        relative_max = error.abs().max() / expected.abs().max().clamp_min(1e-10)
        assert nrmse <= 0.01, nrmse
        assert relative_max <= 0.02, relative_max
    else:
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


def inputs(tokens, strided=False):
    q = torch.randn(tokens, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(tokens, 48, 128, device="cuda", dtype=torch.bfloat16)
    if strided:
        packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
        q, k, v = (
            part.reshape(tokens, -1, 128)
            for part in packed.split([2048, 2048, 6144], dim=1)
        )
    g = -torch.rand(tokens, 48, device="cuda")
    beta = torch.rand_like(g)
    return q, k, v, g, beta


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_four_sequence_gdn_fp64(dtype):
    torch.manual_seed(71)
    q, k, v, g, beta = inputs(11, strided=True)
    pool = torch.randn(16, 48, 128, 128, device="cuda", dtype=dtype) * 0.05
    before = pool.clone()
    offsets = [0, 1, 4, 6, 11]
    starts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    reads = torch.arange(4, device="cuda", dtype=torch.int32)
    writes = torch.arange(4, 15, device="cuda", dtype=torch.int32)
    out = gdn.recurrent(q, k, v, g, beta, pool, starts, reads, writes)
    for seq, (left, right) in enumerate(pairwise(offsets)):
        expected, states = reference(
            q[left:right],
            k[left:right],
            v[left:right],
            g[left:right],
            beta[left:right],
            before[seq],
        )
        check(out[left:right], expected)
        check(pool[4 + left : 4 + right], states, state=True)
    torch.testing.assert_close(pool[:4], before[:4])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_ragged_mtp_snapshots_and_prefix_isolation(dtype, strided):
    torch.manual_seed(19)
    q, k, v, g, beta = inputs(6, strided)
    pool = torch.randn(12, 48, 128, 128, device="cuda", dtype=dtype) * 0.05
    before = pool.clone()
    # Distinct source rows, ragged verification lengths 1 and 5, noncontiguous
    # destinations, including one uncommitted token.
    starts = torch.tensor([0, 1, 6], device="cuda", dtype=torch.int32)
    reads = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    writes = torch.tensor([7, 6, -1, 9, 8, 10], device="cuda", dtype=torch.int32)
    out = gdn.recurrent(q, k, v, g, beta, pool, starts, reads, writes)
    for begin, end, source in [(0, 1, 2), (1, 6, 4)]:
        expected, states = reference(
            q[begin:end],
            k[begin:end],
            v[begin:end],
            g[begin:end],
            beta[begin:end],
            before[source],
        )
        check(out[begin:end], expected)
        for index, dest in enumerate(writes[begin:end].cpu().tolist()):
            if dest >= 0:
                check(pool[dest], states[index], state=True)
    torch.testing.assert_close(
        pool[[0, 1, 2, 3, 4, 5, 11]], before[[0, 1, 2, 3, 4, 5, 11]]
    )
    # Resume from an intermediate accepted state, not the last proposed state.
    expected, states = reference(q[:1], k[:1], v[:1], g[:1], beta[:1], pool[9])
    out = gdn.recurrent(
        q[:1],
        k[:1],
        v[:1],
        g[:1],
        beta[:1],
        pool,
        starts[:2],
        torch.tensor([9], device="cuda", dtype=torch.int32),
        torch.tensor([11], device="cuda", dtype=torch.int32),
    )
    check(out, expected)
    check(pool[11], states[0], state=True)


@pytest.mark.parametrize("length", [17, 784, 785])
@pytest.mark.parametrize("strided", [False, True])
def test_prefill_state(length, strided):
    torch.manual_seed(length)
    q, k, v, g, beta = inputs(length, strided)
    initial = torch.randn(1, 48, 128, 128, device="cuda") * 0.05
    starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    out, state = gdn.prefill(q, k, v, g, beta, initial, starts)
    expected, states = reference(q, k, v, g, beta, initial[0], snapshots=False)
    check(out, expected)
    check(state[0], states[-1], state=True)


@pytest.mark.parametrize(
    "rows,features",
    [(rows, 256) for rows in [1, 4, 33, 127, 128, 129, 784, 785]] + [(257, 32768)],
)
def test_fp8_block_scales(rows, features):
    torch.manual_seed(rows)
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    x[:, :128] = 0
    x[:, 128:256] *= 8
    x[:, 256:384] *= 0.001
    weight = (torch.randn(features, 512, device="cuda") * 20).to(torch.float8_e4m3fn)
    scale = torch.rand(features // 128, 4, device="cuda") * 0.01
    out = fp8.linear(x, weight, scale)
    quantized, activation_scale = fp8.quantize(x)
    cpu_x = x.cpu().double().reshape(rows, -1, 128)
    cpu_scale = cpu_x.abs().amax(-1).clamp_min(1e-10) / 448
    torch.testing.assert_close(
        activation_scale.cpu().double(), cpu_scale, rtol=1e-6, atol=1e-15
    )
    # Quantization is specified in FP32; double division can pick the opposite
    # side of an FP8 rounding midpoint. GEMM/state references remain FP64.
    fp32_scale = cpu_x.float().abs().amax(-1).clamp_min(1e-10) / 448
    cpu_quantized = (cpu_x.float() / fp32_scale[..., None]).clamp(-448, 448)
    cpu_quantized = cpu_quantized.to(torch.float8_e4m3fn).reshape_as(x)
    torch.testing.assert_close(
        quantized.cpu().float(), cpu_quantized.float(), rtol=0, atol=0
    )
    restored_x = (
        quantized.cpu().double()
        * activation_scale.cpu().double().repeat_interleave(128, 1)
    )
    restored_w = weight.cpu().double() * scale.cpu().double().repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    expected = restored_x @ restored_w.T
    check(out, expected)


@pytest.mark.parametrize("total,query_length", [(789, 5), (1553, 1025)])
def test_paged_gqa_crosses_784_with_prefix(total, query_length):
    from oh_my_vllm.kernels.attention import PagedAttention, append

    torch.manual_seed(784)
    q = torch.randn(query_length, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache = torch.zeros(4, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    # Noncontiguous pages: 2 then 1. Leave pages 0 and 3 untouched.
    slots = torch.cat(
        (torch.arange(784) + 2 * 784, torch.arange(total - 784) + 784)
    ).to(device="cuda", dtype=torch.int64)
    append(cache, k, v, slots)
    plan = PagedAttention()
    plan.plan(
        torch.tensor([0, query_length], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([total - 784], dtype=torch.int32),
        24,
        4,
        256,
    )
    out = plan(q, cache)
    qc, kc, vc = (t.cpu().double() for t in (q, k, v))
    kc, vc = (t.repeat_interleave(6, 1) for t in (kc, vc))
    scores = torch.einsum("thd,shd->hts", qc, kc) / 16
    positions = torch.arange(total - query_length, total)
    mask = torch.arange(total)[None, :] > positions[:, None]
    scores.masked_fill_(mask[None], float("-inf"))
    expected = torch.einsum("hts,shd->thd", scores.softmax(-1), vc)
    check(out, expected)
    assert torch.count_nonzero(cache[[0, 3]]) == 0


def test_long_prefill_keeps_ragged_requests_isolated():
    from oh_my_vllm.kernels.attention import PagedAttention, append

    cache = torch.zeros(9, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    slots = torch.cat(
        (
            torch.arange(784) + 3 * 784,
            torch.arange(516) + 784,
            torch.arange(3920) + 4 * 784,
        )
    ).cuda()
    k = torch.randn(5220, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, 0.5)
    v[1300:] = -0.75
    append(cache, k, v, slots)
    plan = PagedAttention()
    plan.plan(
        torch.tensor([0, 1024, 1025], dtype=torch.int32),
        torch.tensor([0, 2, 7], dtype=torch.int32),
        torch.tensor([3, 1, 4, 5, 6, 7, 8], dtype=torch.int32),
        torch.tensor([516, 784], dtype=torch.int32),
        24,
        4,
        256,
    )
    query = torch.randn(1025, 24, 256, device="cuda", dtype=torch.bfloat16)
    expected = torch.full(query.shape, 0.5, dtype=torch.float64)
    expected[1024:] = -0.75
    check(plan(query, cache), expected)


@pytest.mark.parametrize("strided", [False, True])
def test_convolution_ragged_snapshots(strided):
    from oh_my_vllm.kernels.convolution import causal_conv

    torch.manual_seed(11)
    x = torch.randn(790, 10240, device="cuda", dtype=torch.bfloat16)
    if strided:
        x = torch.cat((x, torch.zeros_like(x)), dim=1)[:, :10240]
    weights = torch.randn(10240, 4, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(10, 10240, 3, device="cuda", dtype=torch.bfloat16)
    before = pool.clone()
    starts = torch.tensor([0, 784, 789, 790], device="cuda", dtype=torch.int32)
    sequence_ids = torch.tensor(
        [0] * 784 + [1] * 5 + [2], device="cuda", dtype=torch.int32
    )
    reads = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    writes = torch.full((790,), -1, device="cuda", dtype=torch.int32)
    writes[783:] = torch.arange(7, device="cuda", dtype=torch.int32) + 3
    # Prefill writes source slot 3 while the last sequence still needs its
    # original contents. The convolution must snapshot all sources first.
    out = causal_conv(x, weights, pool, sequence_ids, starts, reads, writes)
    for begin, end, source in [(0, 784, 1), (784, 789, 2), (789, 790, 3)]:
        history = torch.cat(
            (before[source].cpu().double(), x[begin:end].cpu().double().T), dim=1
        )
        expected = torch.nn.functional.conv1d(
            history[None], weights.cpu().double()[:, None], groups=10240
        )[0].T
        expected = torch.nn.functional.silu(expected)
        check(out[begin:end], expected)
        for token in range(begin, end):
            dest = int(writes[token])
            if dest >= 0:
                offset = token - begin + 1
                torch.testing.assert_close(
                    pool[dest].cpu().double(), history[:, offset : offset + 3]
                )
    torch.testing.assert_close(pool[[1, 2]], before[[1, 2]])


@pytest.mark.parametrize("strided", [False, True])
def test_normalization_and_partial_rotary(strided):
    from oh_my_vllm.kernels.normalization import rms_norm, rotary

    torch.manual_seed(71)
    x = torch.randn(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda") + 1
    gate = torch.randn_like(x)
    if strided:
        x = torch.cat((x, torch.zeros_like(x)), dim=-1)[..., :256]
        gate = torch.cat((gate, torch.zeros_like(gate)), dim=1)[:, :24]
    xc, wc, gc = (t.cpu().double() for t in (x, weight, gate))
    expected = xc * torch.rsqrt(xc.square().mean(-1, keepdim=True) + 1e-6) * wc
    check(rms_norm(x, weight), expected)
    check(rms_norm(x, weight, gate=gate), expected * torch.nn.functional.silu(gc))
    positions = torch.tensor([0, 784, 32768], device="cuda", dtype=torch.int64)
    frequencies = 10000000.0 ** (-torch.arange(32, dtype=torch.float64) / 32)
    angle = positions.cpu().double()[:, None, None] * frequencies[None, None]
    expected = xc.clone()
    expected[..., :32] = xc[..., :32] * angle.cos() - xc[..., 32:64] * angle.sin()
    expected[..., 32:64] = xc[..., 32:64] * angle.cos() + xc[..., :32] * angle.sin()
    check(rotary(x, positions), expected)
    for operation in [
        lambda: rms_norm(x, weight[:128]),
        lambda: rms_norm(x, weight, gate=gate[:1]),
        lambda: rotary(x, positions[:1]),
        lambda: rotary(x, positions.float()),
    ]:
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("rows", [16, 17, 20, 24, 32, 33])
def test_fp8_real_width_tile_boundary_is_deterministic(rows):
    torch.manual_seed(rows)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 5120, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(2, 40, device="cuda") * 0.01
    quantized, activation_scale = fp8.quantize(x)
    restored_x = (
        quantized.cpu().double()
        * activation_scale.cpu().double().repeat_interleave(128, 1)
    )
    restored_w = weight.cpu().double() * scale.cpu().double().repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    expected = restored_x @ restored_w.T
    first = fp8.linear(x, weight, scale)
    check(first, expected)
    for _ in range(3):
        repeated = fp8.linear(x, weight, scale)
        torch.testing.assert_close(repeated, first, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 5, 20, 33])
def test_vocabulary_projection_fp64(rows):
    from oh_my_vllm.models.qwen import Qwen

    torch.manual_seed(rows)
    model = object.__new__(Qwen)
    model.head = torch.randn(248320, 5120, device="cuda", dtype=torch.bfloat16) * 0.01
    hidden = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    actual = model.logits(hidden)
    columns = [0, 1, 127, 128, 8191, 16384, 248319]
    expected = hidden.cpu().double() @ model.head[columns].cpu().double().T
    check(actual[:, columns], expected)
    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(hidden, model.head).float(),
        atol=0.03,
        rtol=0.03,
    )


@pytest.mark.parametrize("rows", [1, 5, 129])
def test_residual_normalization_preserves_bf16_sum(rows):
    from oh_my_vllm.kernels.normalization import add_rms_norm

    torch.manual_seed(rows)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    residual[:, ::3] = -x[:, ::3]
    weight = torch.randn(5120, device="cuda") + 1
    summed, normalized = add_rms_norm(x, residual, weight)
    expected_sum = (x.cpu().double() + residual.cpu().double()).to(torch.bfloat16)
    torch.testing.assert_close(summed.cpu(), expected_sum, atol=0, rtol=0)
    ref = expected_sum.double()
    expected = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + 1e-6)
    check(normalized, expected * weight.cpu().double())


@pytest.mark.parametrize("rows", [1, 5, 20, 128, 129])
def test_fused_silu_quantization_preserves_rounding(rows):
    from oh_my_vllm.kernels.elementwise import silu_mul

    torch.manual_seed(rows)
    packed = torch.randn(rows, 34816, device="cuda", dtype=torch.bfloat16) * 5
    for column_major in (False, True):
        expected = fp8.quantize(silu_mul(packed), column_major=column_major)
        actual = fp8.quantize(packed, column_major=column_major, silu_gate=True)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.float(), b.float(), atol=0, rtol=0)
