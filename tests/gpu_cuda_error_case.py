"""Isolated CUDA fault and graph-capture cases for KRN-09."""

import sys
from pathlib import Path

import torch
from oh_my_vllm.kernels import cuda_backend
from oh_my_vllm.kernels.cuda_backend import compiled


def _probe_module():
    from tvm_ffi import load_module
    from tvm_ffi.cpp import build_inline

    root = Path(__file__).resolve().parents[1]
    source = (
        root / "python/oh_my_vllm/kernels/cuda_backend/kernels.cu"
    ).read_text() + (root / "tests/fixtures/cuda_error_probe.cu").read_text()
    digest = cuda_backend._build_input_digest(source, str(torch.__version__))
    path = build_inline(
        "oh_my_vllm_cuda_error_probe",
        cuda_sources=source,
        functions=[
            "variant_launch_count",
            "diagnostic_prior_launch_then_rms",
            "diagnostic_prior_execution_then_rms",
            "diagnostic_current_execution",
            "diagnostic_peek_error",
            "diagnostic_clear_error",
        ],
        backend="cuda",
        extra_include_paths=[str(root / "python/oh_my_vllm/kernels/cuda_backend")],
        extra_cuda_cflags=[
            *cuda_backend._BASE_CUDA_FLAGS,
            cuda_backend._pdl_flag(),
            f"-DOH_MY_VLLM_PROBE_INPUT_{digest[:20]}",
        ],
    )
    return load_module(path)


def main(scenario):
    x = torch.ones(1, 1, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(5120, device="cuda")
    out = torch.empty_like(x)
    if scenario == "prior_launch":
        module = _probe_module()
        before = module.variant_launch_count(0, True)
        try:
            module.diagnostic_prior_launch_then_rms(x, weight, x, out)
        except Exception as exc:
            assert "norm prior CUDA error before launch" in str(exc), str(exc)
        else:
            raise AssertionError("prior launch error was not detected")
        assert module.variant_launch_count(0, True) == before
        assert module.diagnostic_peek_error() != 0
        assert module.diagnostic_clear_error() != 0
        assert module.diagnostic_peek_error() == 0
    elif scenario == "normal_preserves_error":
        module = _probe_module()
        before = module.variant_launch_count(0, True)
        try:
            module.diagnostic_prior_launch_then_rms(x, weight, x, out)
        except Exception as exc:
            assert "norm CUDA launch or prior asynchronous error" in str(exc), str(exc)
        else:
            raise AssertionError("default launch check missed the existing error")
        assert module.variant_launch_count(0, True) == before
        assert module.diagnostic_peek_error() != 0
        assert module.diagnostic_clear_error() != 0
        assert module.diagnostic_peek_error() == 0
    elif scenario == "prior_execution":
        module = _probe_module()
        try:
            module.diagnostic_prior_execution_then_rms(x, weight, x, out)
        except Exception as exc:
            assert "norm prior CUDA" in str(exc), str(exc)
            assert "before launch" in str(exc), str(exc)
        else:
            raise AssertionError("prior asynchronous fault was not detected")
    elif scenario == "current_execution":
        module = _probe_module()
        try:
            module.diagnostic_current_execution(x)
        except Exception as exc:
            assert "diagnostic_current_execution CUDA" in str(exc), str(exc)
            assert "after launch" in str(exc), str(exc)
        else:
            raise AssertionError("current asynchronous fault was not detected")
    elif scenario == "eager":
        compiled().rms(x, weight, x, out, 1e-6, False)
        torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    elif scenario in ("capture_debug", "capture_normal"):
        module = compiled()
        graph = torch.cuda.CUDAGraph()
        caught = None
        with torch.cuda.graph(graph):
            try:
                module.rms(x, weight, x, out, 1e-6, False)
            except Exception as exc:
                caught = str(exc)
        if scenario == "capture_debug":
            assert caught and "CUDA debug sync requires eager execution" in caught
        else:
            assert caught is None, caught
            graph.replay()
            torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    else:
        raise ValueError(f"unknown KRN-09 scenario: {scenario}")
    print(f"passed {scenario}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
