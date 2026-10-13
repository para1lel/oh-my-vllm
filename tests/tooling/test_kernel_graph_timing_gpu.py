"""Graph timing excludes host gaps and retains the complete operator chain."""

import re
import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.kernels.dspark_attention import rms_norm

from development.kernels import timing

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("side", ["before", "after"])
def test_graph_events_exclude_host_gaps_and_enclose_all_operations(
    monkeypatch, tmp_path, side
):
    original_graph = torch.cuda.CUDAGraph
    graphs = []
    delay = 0.0

    class DelayedGraph(original_graph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_debug_mode()
            graphs.append(self)

        def replay(self):
            if side == "before":
                time.sleep(delay)
            super().replay()
            if side == "after":
                time.sleep(delay)

    cuda = SimpleNamespace(
        CUDAGraph=DelayedGraph,
        Event=torch.cuda.Event,
        graph=torch.cuda.graph,
        synchronize=torch.cuda.synchronize,
    )
    monkeypatch.setattr(timing, "torch", SimpleNamespace(cuda=cuda))
    x = torch.randn(7, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)

    def run():
        return rms_norm(x, weight)

    try:
        baseline = timing.measure(run, run)
        delay = 0.02
        delayed = timing.measure(run, run)

        def median(rows):
            return statistics.median(
                value for pairs in rows for pair in pairs for value in pair
            )

        # A host gap in the interval adds 0.2 ms per operation at 100 repeats.
        assert median(delayed) < median(baseline) + 0.05

        for index, graph in enumerate(graphs):
            path = tmp_path / f"graph-{index}.dot"
            graph.debug_dump(str(path))
            dot = path.read_text()
            nodes = dict(re.findall(r'"(graph_\d+_node_\d+)"\[(.*?)\];', dot, re.S))
            edges = re.findall(r'"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"', dot)
            assert len(nodes) == 102 and len(edges) == 101
            event_nodes = {
                key for key, label in nodes.items() if "EVENT_RECORD" in label
            }
            kernel_nodes = {key for key, label in nodes.items() if "{KERNEL" in label}
            assert len(event_nodes) == 2 and len(kernel_nodes) == 100
            starts = set(nodes) - {target for _, target in edges}
            assert len(starts) == 1
            chain = dict(edges)
            order, node = [], next(iter(starts))
            while node is not None:
                assert node not in order
                order.append(node)
                node = chain.get(node)
            assert len(order) == 102
            assert order[0] in event_nodes and order[-1] in event_nodes
            assert all(node in kernel_nodes for node in order[1:-1])
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            errors = []
            for graph in graphs:
                try:
                    graph.reset()
                except RuntimeError as error:
                    errors.append(error)
            # Reset even when a traceback retains the local debug subclass.
            graphs.clear()
            if errors:
                raise RuntimeError("debug graph cleanup failed") from errors[0]
