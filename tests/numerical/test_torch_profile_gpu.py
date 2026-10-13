import json
import logging

import pytest
import torch
from oh_my_vllm.kernels.elementwise import silu_mul
from oh_my_vllm.worker.torch_profile import BenchmarkProfiler, kernel_timeline

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_profile_records_real_graph_kernels_in_bounded_window(tmp_path, caplog):
    x = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        silu_mul(x)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = silu_mul(x)
    path = tmp_path / "timeline.json"
    profiler = BenchmarkProfiler({"path": str(path), "skip_steps": 1, "steps": 2})
    with caplog.at_level(logging.INFO, logger="oh_my_vllm.worker.torch_profile"):
        for step in range(5):
            with profiler.step(step, 2):
                graph.replay()
        profiler.close()
    assert profiler.recorded == 2
    rows = kernel_timeline(json.loads(path.read_text()))
    assert rows and all(row["duration_us"] > 0 for row in rows)
    assert all(row["stream"] is not None and row["device"] == 0 for row in rows)
    assert any("silu" in row["kernel"] for row in rows)
    logged = [
        record.fields
        for record in caplog.records
        if record.message == "torch_profile_kernel"
    ]
    assert logged == rows
    torch.testing.assert_close(output, silu_mul(x), atol=0, rtol=0)
