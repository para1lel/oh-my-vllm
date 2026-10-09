"""A repeated operator attempt must preserve previous measurement bytes."""

import importlib.util
import sys
from pathlib import Path

import pytest


def test_existing_output_fails_before_hardware_initialization(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "kernel_collector", root / "benchmarks/kernels.py"
    )
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    output = tmp_path / "results.json"
    previous = b'{"passed": false, "attempt": "retained"}\n'
    output.write_bytes(previous)
    monkeypatch.setenv("OH_MY_VLLM_KERNEL_BACKEND", "cuda")
    monkeypatch.setattr(sys, "argv", ["kernels.py", "--output", str(output)])

    def hardware_must_not_start():
        pytest.fail("hardware initialization preceded evidence reservation")

    monkeypatch.setattr(collector, "hardware_identity", hardware_must_not_start)
    with pytest.raises(FileExistsError):
        collector.main()
    assert output.read_bytes() == previous
