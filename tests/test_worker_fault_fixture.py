"""The external-only fault injector fires once across worker reconstruction."""

import importlib.util
from pathlib import Path

import pytest
import torch
from oh_my_vllm.worker.model_runner import OhMyVllmWorker


@pytest.mark.parametrize("kind", ["oom", "device", "internal"])
def test_explicit_fixture_fault_is_once_per_marker(tmp_path, monkeypatch, kind):
    path = Path(__file__).parent / "fixtures" / "worker_faults.py"
    spec = importlib.util.spec_from_file_location("test_owned_worker_fault", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Patch registration restores the original method after the installer wraps
    # it; the test never installs injection into an actual running worker.
    monkeypatch.setattr(OhMyVllmWorker, "_commit_plans", lambda self: "committed")
    marker = tmp_path / "fault.once"
    module.install_fault("after_commit", kind, marker)
    expected = {
        "oom": torch.cuda.OutOfMemoryError,
        "device": torch.AcceleratorError,
        "internal": RuntimeError,
    }[kind]
    first = OhMyVllmWorker.__new__(OhMyVllmWorker)
    with pytest.raises(expected, match="test-owned"):
        first._commit_plans()
    assert marker.is_file()
    replacement = OhMyVllmWorker.__new__(OhMyVllmWorker)
    assert replacement._commit_plans() == "committed"


def test_capture_fixture_preserves_shape_specific_headroom(tmp_path, monkeypatch):
    from oh_my_vllm.worker.graph_cache import GraphCache

    path = Path(__file__).parent / "fixtures" / "worker_faults.py"
    spec = importlib.util.spec_from_file_location("test_owned_capture_fault", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Register the original for restoration after the external installer wraps it.
    monkeypatch.setattr(GraphCache, "get_or_create", GraphCache.get_or_create)
    marker = tmp_path / "capture.once"
    module.install_fault("graph_capture", "internal", marker)

    def cache():
        return GraphCache(
            capacity=4,
            min_capture_free_bytes=64,
            free_bytes=lambda: 32,
            synchronize=lambda: None,
        )

    created = []

    def create():
        value = object()
        created.append(value)
        return value

    with pytest.raises(RuntimeError, match="test-owned internal fault"):
        cache().get_or_create("prefill_dspark", (1,), create, headroom_bytes=16)
    assert len(created) == 1 and marker.is_file()
    replacement = cache()
    result = replacement.get_or_create(
        "prefill_dspark", (1,), create, headroom_bytes=16
    )
    assert result is created[-1] and len(created) == 2
    assert replacement.snapshot()["prefill_dspark_max_capture_headroom_bytes"] == 16
