"""Direct CUDA FFI calls reject unsupported storage before any kernel launch."""

import subprocess
import sys

import pytest
import torch
from oh_my_vllm.kernels.cuda_backend import compiled

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_quantize_rejects_unsupported_input_dtype(dtype):
    _assert_rejected_in_child("quantize", dtype)


@pytest.mark.parametrize("scenario", ["silu_float16", "silu_float32"])
def test_fused_silu_quantize_rejects_non_bf16_input(scenario):
    _assert_rejected_in_child("quantize", scenario)


def _assert_rejected_in_child(entry, dtype):
    scenario = str(dtype).split(".")[-1]
    process = subprocess.run(
        [sys.executable, "-m", "tests.gpu_cuda_contract_case", entry, scenario],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert f"rejected {entry}/{scenario}" in process.stdout


@pytest.mark.parametrize(
    "scenario",
    [
        "out_float32",
        "scales_bfloat16",
        "out_short",
        "scales_short",
        "empty",
        "out_cpu",
        "scales_cpu",
        "over_i32",
    ],
)
def test_quantize_rejects_bad_outputs(scenario):
    _assert_rejected_in_child("quantize", scenario)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_keeps_supported_input_dtypes(dtype):
    x = torch.randn(1, 128, device="cuda").to(dtype)
    out = torch.empty(1, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, 1, device="cuda")
    compiled().quantize(x, out, scales, False, False)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(scales).all() and scales.item() > 0


def test_fused_silu_quantize_keeps_bf16_input():
    x = torch.ones(1, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, 1, device="cuda")
    compiled().quantize(x, out, scales, False, True)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(scales).all() and scales.item() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_recurrent_rejects_unsupported_state_dtype(dtype):
    _assert_rejected_in_child("recurrent", dtype)


@pytest.mark.parametrize(
    "scenario",
    [
        "q_int8",
        "k_int8",
        "v_int8",
        "decay_bfloat16",
        "beta_bfloat16",
        "starts_int16",
        "reads_int16",
        "writes_int16",
        "empty",
        "pool_short",
    ],
)
def test_recurrent_rejects_bad_inputs_and_short_state(scenario):
    _assert_rejected_in_child("recurrent", scenario)


@pytest.mark.parametrize("scenario", ["out_float32", "out_short", "out_cpu"])
def test_recurrent_rejects_bad_output(scenario):
    _assert_rejected_in_child("recurrent", scenario)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vector", [False, True])
def test_recurrent_keeps_supported_state_dispatch(dtype, vector):
    q = torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    if not vector:
        q = q.as_strided(q.shape, (2049, 128, 1))
    v = torch.zeros(1, 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(1, 48, device="cuda")
    pool = torch.zeros(2, 48, 128, 128, device="cuda", dtype=dtype)
    starts = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    reads = torch.tensor([0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([1], device="cuda", dtype=torch.int32)
    out = torch.empty_like(v)
    compiled().recurrent(q, q, v, gate, gate, pool, starts, reads, writes, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("storage_offset", [0, 1])
def test_recurrent_state_vector_alignment(dtype, storage_offset):
    q = torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(1, 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(1, 48, device="cuda")
    shape = (2, 48, 128, 128)
    storage = torch.empty(
        2 * 48 * 128 * 128 + storage_offset, device="cuda", dtype=dtype
    )
    pool = storage[storage_offset:].view(shape)
    assert pool.data_ptr() % dtype.itemsize == 0
    pool[0] = torch.arange(pool[0].numel(), device="cuda").view_as(pool[0]) % 127
    pool[1].zero_()
    before = pool[0].clone()
    starts = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    reads = torch.tensor([0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([1], device="cuda", dtype=torch.int32)
    out = torch.empty_like(v)
    compiled().recurrent(q, q, v, gate, gate, pool, starts, reads, writes, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(pool[1], before, rtol=0, atol=0)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_recurrent_direct_ffi_preserves_skip_snapshot_sentinel():
    q = torch.zeros(1, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(1, 48, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(1, 48, device="cuda")
    pool = torch.ones(2, 48, 128, 128, device="cuda")
    before = pool.clone()
    starts = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    reads = torch.tensor([0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([-1], device="cuda", dtype=torch.int32)
    out = torch.empty_like(v)
    compiled().recurrent(q, q, v, gate, gate, pool, starts, reads, writes, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "scenario",
    [
        "decay_bfloat16",
        "decay_short",
        "empty_decay_bfloat16",
        "decay_cpu",
        "empty_decay_cpu",
        "beta_bfloat16",
        "beta_short",
        "beta_cpu",
    ],
)
def test_gates_rejects_bad_outputs(scenario):
    _assert_rejected_in_child("gates", scenario)


def test_direct_empty_gate_call_does_not_launch_zero_grid():
    projected = torch.empty(0, 96, device="cuda", dtype=torch.bfloat16)
    params = torch.zeros(48, device="cuda")
    decay = torch.empty(0, 48, device="cuda")
    beta = torch.empty_like(decay)
    compiled().gates(projected, params, params, decay, beta)
    assert decay.numel() == beta.numel() == 0


def test_direct_nonempty_gate_call_keeps_expected_values():
    projected = torch.zeros(1, 96, device="cuda", dtype=torch.bfloat16)
    params = torch.zeros(48, device="cuda")
    decay = torch.empty(1, 48, device="cuda")
    beta = torch.empty_like(decay)
    compiled().gates(projected, params, params, decay, beta)
    torch.cuda.synchronize()
    expected_decay = torch.full_like(decay, -torch.log(torch.tensor(2.0)).item())
    torch.testing.assert_close(decay, expected_decay, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(beta, torch.full_like(beta, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize(
    "scenario",
    [
        "over_i32",
        "input_fp32",
        "output_fp32",
        "out_short",
        "out_cpu",
        "input_strided",
        "output_strided",
    ],
)
def test_direct_silu_mul_rejects_invalid_contract(scenario):
    _assert_rejected_in_child("silu_mul", scenario)


def test_direct_silu_mul_keeps_expected_output_and_empty_noop():
    x = torch.ones(1, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    compiled().silu_mul(x, out)
    empty_x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    empty_out = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    compiled().silu_mul(empty_x, empty_out)
    torch.cuda.synchronize()
    expected = torch.nn.functional.silu(torch.ones_like(out))
    torch.testing.assert_close(out.float(), expected.float(), rtol=0.003, atol=0.003)
    assert empty_out.numel() == 0
