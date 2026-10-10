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
