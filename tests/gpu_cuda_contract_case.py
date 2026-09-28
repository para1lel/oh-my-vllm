"""Isolate malformed direct FFI calls from the parent GPU test process."""

import sys

import torch
from oh_my_vllm.kernels.cuda_backend import compiled


def quantize_case(scenario):
    input_dtype = (
        getattr(torch, scenario.removeprefix("silu_"))
        if scenario in ("float64", "int32", "silu_float16", "silu_float32")
        else torch.bfloat16
    )
    rows = 0 if scenario == "empty" else 1
    silu = scenario.startswith("silu_")
    x = torch.ones(rows, 256 if silu else 128, device="cuda", dtype=input_dtype)
    output_dtype = torch.float32 if scenario == "out_float32" else torch.float8_e4m3fn
    output_width = 64 if scenario == "out_short" else 128
    out_device = "cpu" if scenario == "out_cpu" else "cuda"
    out = torch.zeros(rows, output_width, device=out_device, dtype=output_dtype)
    scale_dtype = torch.bfloat16 if scenario == "scales_bfloat16" else torch.float32
    scale_device = "cpu" if scenario == "scales_cpu" else "cuda"
    scale_width = 0 if scenario == "scales_short" else 1
    scales = torch.full(
        (rows, scale_width), -1.0, device=scale_device, dtype=scale_dtype
    )
    expected = (
        "tensors must share one CUDA device"
        if scenario in ("out_cpu", "scales_cpu")
        else "requires BF16, FP16, or FP32 input"
        if scenario in ("float64", "int32")
        else "fused SiLU quantize requires BF16 input"
        if silu
        else "requires FP8 output and FP32 scales"
        if scenario in ("out_float32", "scales_bfloat16")
        else "output shape or layout is invalid"
    )

    def invoke():
        compiled().quantize(x, out, scales, False, silu)

    def unchanged():
        torch.testing.assert_close(out.float(), torch.zeros_like(out.float()))
        assert scales.numel() == 0 or scales.item() == -1

    return invoke, unchanged, expected


def recurrent_case(scenario):
    rows = 0 if scenario == "empty" else 1
    q_dtype = torch.int8 if scenario == "q_int8" else torch.bfloat16
    k_dtype = torch.int8 if scenario == "k_int8" else torch.bfloat16
    v_dtype = torch.int8 if scenario == "v_int8" else torch.bfloat16
    q = torch.zeros(rows, 16, 128, device="cuda", dtype=q_dtype)
    k = torch.zeros(rows, 16, 128, device="cuda", dtype=k_dtype)
    v = torch.zeros(rows, 48, 128, device="cuda", dtype=v_dtype)
    gate_dtype = torch.bfloat16 if scenario == "decay_bfloat16" else torch.float32
    beta_dtype = torch.bfloat16 if scenario == "beta_bfloat16" else torch.float32
    gate = torch.zeros(rows, 48, device="cuda", dtype=gate_dtype)
    beta = torch.zeros(rows, 48, device="cuda", dtype=beta_dtype)
    pool_dtype = (
        getattr(torch, scenario)
        if scenario in ("float16", "float64", "int32")
        else torch.bfloat16
    )
    pool_width = 1 if scenario == "pool_short" else 128
    pool = torch.zeros(2, 48, 128, pool_width, device="cuda", dtype=pool_dtype)
    starts_dtype = torch.int16 if scenario == "starts_int16" else torch.int32
    reads_dtype = torch.int16 if scenario == "reads_int16" else torch.int32
    writes_dtype = torch.int16 if scenario == "writes_int16" else torch.int32
    starts = torch.tensor([0, rows], device="cuda", dtype=starts_dtype)
    reads = torch.tensor([] if rows == 0 else [0], device="cuda", dtype=reads_dtype)
    writes = torch.tensor([] if rows == 0 else [1], device="cuda", dtype=writes_dtype)
    output_dtype = torch.float32 if scenario == "out_float32" else torch.bfloat16
    output_width = 1 if scenario == "out_short" else 128
    output_device = "cpu" if scenario == "out_cpu" else "cuda"
    out = torch.full(
        (rows, 48, output_width), 5, device=output_device, dtype=output_dtype
    )
    expected = (
        "tensors must share one CUDA device"
        if scenario == "out_cpu"
        else "requires nonempty BF16 q/k/v rows"
        if scenario in ("q_int8", "k_int8", "v_int8", "empty")
        else "requires contiguous FP32 gates"
        if scenario in ("decay_bfloat16", "beta_bfloat16")
        else "requires nonempty int32/int64 metadata"
        if scenario in ("starts_int16", "reads_int16", "writes_int16")
        else "state shape or layout is invalid"
        if scenario == "pool_short"
        else "state requires BF16 or FP32"
        if scenario in ("float16", "float64", "int32")
        else "output requires contiguous BF16 value shape"
    )

    def invoke():
        compiled().recurrent(q, k, v, gate, beta, pool, starts, reads, writes, out)

    def unchanged():
        torch.testing.assert_close(out, torch.full_like(out, 5), rtol=0, atol=0)

    return invoke, unchanged, expected


def gates_case(scenario):
    rows = 0 if scenario in ("empty_decay_bfloat16", "empty_decay_cpu") else 1
    projected = torch.zeros(rows, 96, device="cuda", dtype=torch.bfloat16)
    params = torch.zeros(48, device="cuda")
    decay_dtype = torch.bfloat16 if scenario.endswith("bfloat16") else torch.float32
    width = 1 if scenario == "decay_short" else 48
    decay_device = "cpu" if scenario in ("decay_cpu", "empty_decay_cpu") else "cuda"
    decay = torch.full((rows, width), -1, device=decay_device, dtype=decay_dtype)
    beta_dtype = torch.bfloat16 if scenario == "beta_bfloat16" else torch.float32
    beta_width = 1 if scenario == "beta_short" else 48
    beta_device = "cpu" if scenario == "beta_cpu" else "cuda"
    beta = torch.full((rows, beta_width), -1, device=beta_device, dtype=beta_dtype)

    def invoke():
        compiled().gates(projected, params, params, decay, beta)

    def unchanged():
        assert torch.all(decay == -1) and torch.all(beta == -1)

    expected = (
        "tensors must share one CUDA device"
        if scenario in ("decay_cpu", "empty_decay_cpu", "beta_cpu")
        else "requires BF16 projection and FP32 [rows,48] outputs"
    )
    return invoke, unchanged, expected


def main() -> None:
    entry, scenario = sys.argv[1:]
    cases = {
        "quantize": quantize_case,
        "recurrent": recurrent_case,
        "gates": gates_case,
    }
    invoke, unchanged, expected = cases[entry](scenario)
    try:
        invoke()
        torch.cuda.synchronize()
    except Exception as error:
        if expected not in str(error):
            raise
    else:
        raise AssertionError(f"{entry}/{scenario} was not rejected")
    unchanged()
    print(f"rejected {entry}/{scenario}")


if __name__ == "__main__":
    main()
