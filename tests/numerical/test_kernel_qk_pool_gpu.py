"""Shared Q/K timing preserves complete calls, input bytes, and output addresses."""

import re

import pytest
import torch

from development.kernels import timing
from development.kernels.cases import cases
from development.kernels.fixtures import fixture
from development.kernels.output_verification import verify

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

QK_CASES = [case for case in cases() if case["configuration"]["operation"] == "qk"]


@pytest.mark.parametrize("case", QK_CASES, ids=lambda case: case["id"])
def test_formal_qk_shapes_keep_full_chain_and_identical_output_addresses(
    monkeypatch, tmp_path, case
):
    reference, candidate, witnesses = fixture(case["configuration"], observe=True)
    verify("qk", reference, candidate, witnesses)
    del reference, candidate, witnesses
    reference, candidate, inputs = fixture(case["configuration"], qk_inputs=True)
    original_graph = torch.cuda.CUDAGraph
    graphs = []

    class DebugGraph(original_graph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_debug_mode()
            graphs.append(self)

        def reset(self):
            try:
                path = tmp_path / f"qk-{graphs.index(self)}.dot"
                if not path.exists():
                    self.debug_dump(str(path))
            finally:
                super().reset()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", DebugGraph)
    try:
        audit = {}
        raw = timing.measure(
            reference, candidate, qk_inputs=inputs, audit=audit, rounds=1, pairs=2
        )
        assert len(raw) == 1 and len(raw[0]) == 2
        assert audit["matching_capture_output_addresses"]
        assert audit["input_storage_unchanged"]
        assert audit["checked_calls_per_backend"] == 100
        assert audit["output_allocations"] >= 2
        assert len(audit["pointer_sequence_sha256"]) == 64
        assert len(graphs) == 2
        for index in range(len(graphs)):
            path = tmp_path / f"qk-{index}.dot"
            # The helper resets graph resources before returning. Its test
            # subclass saves the complete DOT immediately before that reset.
            dot = path.read_text()
            nodes = dict(re.findall(r'"(graph_\d+_node_\d+)"\[(.*?)\];', dot, re.S))
            edges = re.findall(r'"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"', dot)
            assert len(nodes) == 102 and len(edges) == 101
            starts = set(nodes) - {target for _, target in edges}
            assert len(starts) == 1
            chain = dict(edges)
            order, node = [], next(iter(starts))
            while node is not None:
                assert node not in order
                order.append(node)
                node = chain.get(node)
            assert len(order) == 102
            assert "EVENT_RECORD" in nodes[order[0]]
            assert "EVENT_RECORD" in nodes[order[-1]]
            assert all("{KERNEL" in nodes[node] for node in order[1:-1])
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


def test_shared_qk_timing_rejects_aliases_and_input_mutation():
    q = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    with pytest.raises(ValueError, match="must not alias"):
        timing.measure(
            lambda: (q, k),
            lambda: (q, k),
            qk_inputs=(q, k),
            audit={},
            rounds=1,
            pairs=1,
            repeats=2,
        )

    def mutate():
        q.add_(1)
        return q.clone(), k.clone()

    with pytest.raises(ValueError, match="input storage"):
        timing.measure(
            mutate, mutate, qk_inputs=(q, k), audit={}, rounds=1, pairs=1, repeats=2
        )


def test_shared_qk_timing_rejects_fixed_outputs_and_invalid_counts():
    q = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    fixed = (q.clone(), k.clone())
    with pytest.raises(ValueError, match="fresh outputs"):
        timing.measure(
            lambda: fixed,
            lambda: fixed,
            qk_inputs=(q, k),
            audit={},
            rounds=1,
            pairs=1,
            repeats=2,
        )
    with pytest.raises(ValueError, match="fresh outputs"):
        timing.measure(
            lambda: fixed,
            lambda: (q.clone(), k.clone()),
            qk_inputs=(q, k),
            audit={},
            rounds=1,
            pairs=1,
            repeats=2,
        )
    reference, candidate, inputs = fixture(
        {"operation": "qk", "tokens": 2}, qk_inputs=True
    )
    timing.measure(
        reference, candidate, qk_inputs=inputs, audit={}, rounds=1, pairs=1, repeats=2
    )
    for name in ("rounds", "pairs", "repeats"):
        with pytest.raises(ValueError, match="positive"):
            timing.measure(
                lambda: fixed, lambda: fixed, qk_inputs=(q, k), audit={}, **{name: 0}
            )
