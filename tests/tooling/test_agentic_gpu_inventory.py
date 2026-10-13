"""GPU inventory selects the serving worker's pinned UUID and rejects PID reuse."""

import subprocess
import sys

import pytest

from benchmarks import agentic_soak_support as support


@pytest.mark.parametrize("mismatch", [None, "device", "birth"])
def test_inventory_is_scoped_to_pinned_worker_uuid(monkeypatch, mismatch):
    uuid = "GPU-" + "test-device"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env={"CUDA_VISIBLE_DEVICES": uuid},
    )
    original_record = support._process_record
    queried = []
    inventory_done = [False]

    def query(command, **kwargs):
        queried.append(command)
        assert kwargs["timeout"] == 10
        inventory_done[0] = True
        selected = "GPU-foreign" if mismatch == "device" else uuid
        return f"{selected}, {process.pid}, 100\n"

    def record(pid):
        result = original_record(pid)
        if result and mismatch == "birth" and inventory_done[0]:
            result["start_ticks"] += 1
        return result

    monkeypatch.setattr(support.subprocess, "check_output", query)
    monkeypatch.setattr(support, "_process_record", record)
    try:
        if mismatch:
            with pytest.raises(ValueError):
                support.gpu_processes(process.pid)
        else:
            result = support.gpu_processes(process.pid)
            assert result["gpu_uuid"] == uuid
            assert result["processes"][0]["pid"] == process.pid
        assert queried[0][1:3] == ["-i", uuid]
    finally:
        process.terminate()
        process.wait(timeout=10)
