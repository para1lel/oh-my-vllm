import pytest
from oh_my_vllm.worker.torch_profile import BenchmarkProfiler, kernel_timeline


def test_timeline_uses_device_events_and_keeps_overlap():
    rows = kernel_timeline(
        {
            "traceEvents": [
                {
                    "cat": "cpu_op",
                    "name": "aten::mm",
                    "args": {"External id": 7},
                    "ts": 0,
                    "dur": 10000,
                },
                {
                    "cat": "kernel",
                    "ph": "X",
                    "name": "gemm",
                    "ts": 105,
                    "dur": 10,
                    "args": {"External id": 7, "stream": 3, "device": 0},
                },
                {
                    "cat": "kernel",
                    "ph": "X",
                    "name": "copy",
                    "ts": 100,
                    "dur": 20,
                    "args": {"stream": 4, "device": 0},
                },
            ]
        }
    )
    assert [row["start_us"] for row in rows] == [0, 5]
    assert [row["duration_us"] for row in rows] == [20, 10]
    assert [row["stream"] for row in rows] == [4, 3]
    assert rows[1]["operator"] == "aten::mm"
    with pytest.raises(RuntimeError, match="CUDA kernel"):
        kernel_timeline(
            {"traceEvents": [{"cat": "cpu_op", "name": "enqueue", "ts": 0, "dur": 50}]}
        )


@pytest.mark.parametrize(
    "skip,steps", [(-1, 2), (10001, 2), (0, 0), (0, 9), (True, 2), (0, False)]
)
def test_window_bounds_reject_invalid_configuration(tmp_path, skip, steps):
    with pytest.raises(ValueError):
        BenchmarkProfiler(
            {"path": str(tmp_path / "trace.json"), "skip_steps": skip, "steps": steps}
        )


def test_disabled_profiler_and_skipped_window_never_start_cuda(tmp_path):
    disabled = BenchmarkProfiler(None)
    with disabled.step(1, 32768):
        pass
    disabled.close()
    assert disabled.profile is None
    path = tmp_path / "trace.json"
    skipped = BenchmarkProfiler({"path": str(path), "skip_steps": 1, "steps": 2})
    with skipped.step(1, 5):
        pass
    assert skipped.profile is None
    skipped.close()
    assert skipped.completed
    with pytest.raises(ValueError, match="new external"):
        BenchmarkProfiler({"path": str(path), "skip_steps": 0, "steps": 2})


@pytest.mark.parametrize("original_error", [False, True])
def test_export_failure_releases_profiler_and_preserves_execution_error(
    tmp_path, monkeypatch, original_error
):
    from unittest.mock import MagicMock

    import torch

    recorder = MagicMock()
    recorder.export_chrome_trace.side_effect = OSError("export failed")
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda name: __import__("contextlib").nullcontext(),
    )
    profiler = BenchmarkProfiler(
        {"path": str(tmp_path / "failed.json"), "skip_steps": 0, "steps": 1}
    )
    profiler.profile = recorder
    expected = RuntimeError("model failed") if original_error else OSError
    with (
        pytest.raises(
            type(expected) if original_error else expected,
            match="model failed" if original_error else "export failed",
        ) as caught,
        profiler.step(1, 1),
    ):
        if original_error:
            raise expected
    if original_error:
        assert caught.value is expected
    assert profiler.completed
    assert profiler.profile is None
    recorder.__exit__.assert_called_once()
    profiler.close()
    recorder.__exit__.assert_called_once()
