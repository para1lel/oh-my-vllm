"""Workspace ownership fences and allocator validation without a GPU."""

import os
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from oh_my_vllm.worker import graph_capture
from oh_my_vllm.worker.model_runner import OhMyVllmWorker


@pytest.mark.parametrize(
    ("backend", "settings", "allowed"),
    [
        ("native", "", True),
        ("native", "graph_capture_record_stream_reuse:True", True),
        ("native", "roundup_power2_divisions:[256:1,512:2]", True),
        ("native", "max_split_size_mb:128", False),
        (
            "native",
            "graph_capture_record_stream_reuse:True, max_split_size_mb:64",
            False,
        ),
        ("cudaMallocAsync", "backend:cudaMallocAsync", False),
    ],
)
def test_validate_effective_allocator_settings(backend, settings, allowed):
    with (
        patch("torch.cuda.get_allocator_backend", return_value=backend),
        patch("torch._C._accelerator_getAllocatorSettings", return_value=settings),
        patch("torch._C._cuda_cudaCachingAllocator_is_enabled", return_value=True),
    ):
        if allowed:
            graph_capture.validate_graph_allocator()
        else:
            with pytest.raises(ValueError, match="native allocator"):
                graph_capture.validate_graph_allocator()


@pytest.mark.parametrize(
    "failure", [None, "synchronize", "clear", "begin", "body", "end"]
)
def test_capture_fences_previous_work_and_preserves_errors(failure):
    events = []
    graph, pool = object(), object()
    error = RuntimeError("capture failure")

    def stage(name):
        events.append(name)
        if failure == name:
            raise error

    @contextmanager
    def capture(actual_graph, *, pool):
        assert actual_graph is graph
        assert pool is expected_pool
        stage("begin")
        try:
            yield
        finally:
            stage("end")

    expected_pool = pool
    with (
        patch.object(graph_capture, "validate_graph_allocator"),
        patch("torch.cuda.synchronize", side_effect=lambda: stage("synchronize")),
        patch(
            "torch._C._cuda_clearCublasWorkspaces", side_effect=lambda: stage("clear")
        ),
        patch("torch.cuda.graph", side_effect=capture),
        patch("torch.backends.cuda.cublas_workspace_size") as blas_budget,
        patch("torch.backends.cuda.cublaslt_workspace_size") as lt_budget,
    ):
        if failure:
            with (
                pytest.raises(RuntimeError) as caught,
                graph_capture.capture_graph(graph, pool=pool),
            ):
                stage("body")
            assert caught.value is error
        else:
            with graph_capture.capture_graph(graph, pool=pool):
                stage("body")
        blas_budget.assert_not_called()
        lt_budget.assert_not_called()
    expected = ["synchronize", "clear", "begin", "body", "end"]
    if failure in ("synchronize", "clear", "begin"):
        expected = expected[: expected.index(failure) + 1]
    assert events == expected


def test_invalid_allocator_does_not_clear_existing_workspace():
    with (
        patch.object(graph_capture, "validate_graph_allocator", side_effect=ValueError),
        patch("torch.cuda.synchronize") as synchronize,
        patch("torch._C._cuda_clearCublasWorkspaces") as clear,
        patch("torch.cuda.graph") as graph,
    ):
        with pytest.raises(ValueError), graph_capture.capture_graph(object()):
            pytest.fail("invalid allocator entered capture")
        synchronize.assert_not_called()
        clear.assert_not_called()
        graph.assert_not_called()


@pytest.mark.parametrize("disable", ["environment", "runtime"])
def test_effective_caching_disabled_is_rejected_with_real_torch(disable):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTORCH_ALLOC_CONF"] = "backend:native"
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if disable == "environment":
        environment["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        setup = ""
    else:
        environment.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
        environment.pop("PYTORCH_NO_HIP_MEMORY_CACHING", None)
        setup = "torch._C._cuda_cudaCachingAllocator_enable(False)\n"
    program = (
        "import torch\n"
        "from oh_my_vllm.worker.graph_capture import validate_graph_allocator\n"
        + setup
        + "assert torch.cuda.get_allocator_backend() == 'native'\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert not torch._C._cuda_cudaCachingAllocator_is_enabled()\n"
        "try:\n"
        "    validate_graph_allocator()\n"
        "except ValueError as error:\n"
        "    assert 'enabled native allocator' in str(error)\n"
        "else:\n"
        "    raise AssertionError('uncached allocation entered graph execution')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("eager", [False, True])
def test_allocator_is_validated_before_worker_device_startup(monkeypatch, eager):
    if eager:
        monkeypatch.setenv("OH_MY_VLLM_ENFORCE_EAGER", "1")
    else:
        monkeypatch.delenv("OH_MY_VLLM_ENFORCE_EAGER", raising=False)
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    validate = Mock(side_effect=ValueError("unsupported allocator"))
    with (
        patch("oh_my_vllm.worker.model_runner.validate_graph_allocator", validate),
        patch("oh_my_vllm.worker.model_runner.identity", return_value={}),
        patch("torch.cuda.set_device") as device,
    ):
        if eager:
            worker.init_device()
            validate.assert_not_called()
            device.assert_called_once_with(0)
        else:
            with pytest.raises(ValueError, match="unsupported allocator"):
                worker.init_device()
            device.assert_not_called()
