"""CPU contracts for bounded context admission and transactional graph capture."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from oh_my_vllm.worker.dspark_graph import DSparkContextGraph, DSparkGraph
from test_dspark_worker import make_worker


@pytest.mark.parametrize("rows", [1, 32])
def test_context_graph_admission_reuses_shape_and_passes_current_metadata(rows):
    worker, model = make_worker()
    worker.compile_model = True
    features = torch.zeros(rows, 16, dtype=torch.bfloat16)
    positions = torch.arange(rows)
    slots = positions + 784
    calls = []

    class Graph:
        MAX_ROWS = 32

        def __init__(self, unit, caches, *inputs, pool):
            calls.append((unit, caches, pool))
            self.unit, self.caches = unit, caches

        def replay(self, *inputs):
            return self.unit(*inputs, self.caches)

    with (
        patch("oh_my_vllm.worker.dspark_graph.DSparkContextGraph", Graph),
        patch("torch.cuda.graph_pool_handle", return_value=object()),
    ):
        assert worker._inject(features, positions, slots) is None
        assert worker._inject(features + 1, positions + 3, slots + 3) is None
    assert len(calls) == 1
    assert calls[0][1] is worker.cache
    assert worker.context_graph_cache.contains("dspark_context", (rows,))
    assert not worker.graph_cache.has_family("dspark_context")
    assert worker.graph_pool is None
    assert worker.context_graph_pool is calls[0][2]
    actual_features, actual_positions, actual_slots = model.injections[-1]
    torch.testing.assert_close(actual_features, features + 1)
    torch.testing.assert_close(actual_positions, positions + 3)
    torch.testing.assert_close(actual_slots, slots + 3)


@pytest.mark.parametrize("rows,compiled", [(33, True), (1, False)])
def test_large_prefill_and_eager_context_keep_the_existing_unit(rows, compiled):
    worker, model = make_worker()
    worker.compile_model = compiled
    with patch("torch.cuda.graph_pool_handle", side_effect=AssertionError):
        worker._inject(
            torch.zeros(rows, 16).bfloat16(), torch.arange(rows), torch.arange(rows)
        )
    assert len(model.injections) == 1
    assert worker.context_graph_cache.clock == 0
    assert worker.context_graph_cache.graphs == {}


def test_context_headroom_denial_preserves_compiled_execution():
    worker, model = make_worker()
    worker.compile_model = True
    worker.context_graph_cache.free_bytes = lambda: 0
    with patch("torch.cuda.graph_pool_handle", side_effect=AssertionError):
        worker._inject(
            torch.zeros(1, 16).bfloat16(), torch.tensor([0]), torch.tensor([784])
        )
    assert len(model.injections) == 1
    assert worker.context_graph_cache.counters["dspark_context_headroom_eager"] == 1


def test_context_shape_churn_keeps_admission_and_proposal_budgets_independent():
    worker, _ = make_worker()
    worker.compile_model = True
    worker.context_graph_cache.capacity = 8
    worker.context_graph_cache.synchronize = lambda: None
    proposal_owner = object()
    worker.graph_cache.get_or_create("dspark", (True, 1, 4096), lambda: proposal_owner)

    class Graph:
        MAX_ROWS = 32

        def __init__(self, unit, caches, *inputs, pool):
            self.unit, self.caches = unit, caches

        def replay(self, *inputs):
            return self.unit(*inputs, self.caches)

    with (
        patch("oh_my_vllm.worker.dspark_graph.DSparkContextGraph", Graph),
        patch("torch.cuda.graph_pool_handle", return_value=object()),
    ):
        for rows in range(1, 33):
            worker._inject(
                torch.zeros(rows, 16).bfloat16(), torch.arange(rows), torch.arange(rows)
            )
        assert len(worker.context_graph_cache.graphs) == 8
        assert worker.context_graph_cache.counters["dspark_context_budget_eager"] == 24
        for _ in range(4):
            worker._inject(
                torch.zeros(9, 16).bfloat16(), torch.arange(9), torch.arange(9)
            )
    assert len(worker.context_graph_cache.graphs) == 8
    assert worker.context_graph_cache.contains("dspark_context", (9,))
    assert worker.context_graph_cache.counters["dspark_context_eviction"] == 1
    assert worker.graph_cache.graphs[("dspark", (True, 1, 4096))] is proposal_owner


def test_context_capture_failure_resets_new_pool_and_obeys_retry_cooldown():
    worker, model = make_worker()
    worker.compile_model = True
    inputs = (torch.zeros(1, 16).bfloat16(), torch.tensor([0]), torch.tensor([784]))
    with (
        patch("torch.cuda.graph_pool_handle", return_value=object()),
        patch(
            "oh_my_vllm.worker.dspark_graph.DSparkContextGraph",
            side_effect=RuntimeError("capture"),
        ) as capture,
    ):
        capture.MAX_ROWS = 32
        with pytest.raises(RuntimeError, match="capture"):
            worker._inject(*inputs)
        assert worker.context_graph_pool is None
        assert not worker.context_graph_cache.graphs
        worker._inject(*inputs)
    assert len(model.injections) == 1
    assert worker.context_graph_cache.counters["dspark_context_capture_failure"] == 1
    assert (
        worker.context_graph_cache.counters["dspark_context_capture_failure_eager"] == 1
    )


def _cpu_capture(unit, caches, features, positions, slots):
    stream = SimpleNamespace(wait_stream=lambda _other: None)
    graph = SimpleNamespace(replay=lambda: None)
    with (
        patch.object(DSparkContextGraph, "_validate_inputs"),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.Stream", return_value=stream),
        patch("torch.cuda.current_stream", return_value=stream),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch("torch.cuda.synchronize"),
    ):
        return DSparkContextGraph(unit, caches, features, positions, slots)


@pytest.mark.parametrize("fail_call", [None, 1, 3])
def test_context_capture_restores_only_destination_rows_even_on_failure(
    fail_call, caplog
):
    caplog.set_level("INFO", logger="oh_my_vllm.worker.dspark_graph")
    caches = [
        torch.full((3, 2, 784, 1, 4), 123, dtype=torch.bfloat16) for _ in range(5)
    ]
    features = torch.arange(8).view(2, 4).bfloat16()
    positions = torch.tensor([783, 784])
    slots = torch.tensor([784 + 783, 2 * 784])
    before = [cache.clone() for cache in caches]
    calls = 0

    def unit(rows, _positions, writes, pools):
        nonlocal calls
        calls += 1
        for pool in pools:
            pool[writes // 784, :, writes % 784] = rows[:, None, None, :]
        if calls == fail_call:
            raise RuntimeError("after context writes")

    if fail_call is None:
        graph = _cpu_capture(unit, caches, features, positions, slots)
        assert graph.outputs is None
        assert calls == 3
        assert graph.caches == tuple(caches)
    else:
        with pytest.raises(RuntimeError, match="after context writes"):
            _cpu_capture(unit, caches, features, positions, slots)
    for cache, original in zip(caches, before, strict=True):
        torch.testing.assert_close(cache, original, atol=0, rtol=0)
    captures = [
        record.message
        for record in caplog.records
        if "Capturing CUDA graph" in record.message
    ]
    assert len(captures) == (0 if fail_call == 1 else 1)
    if captures:
        assert "family=dspark_context key=(2,) rows=2" in captures[0]


def test_proposal_capture_logs_actual_graph_family_key_and_rows(caplog):
    caplog.set_level("INFO", logger="oh_my_vllm.worker.dspark_graph")
    stream = SimpleNamespace(wait_stream=lambda _other: None)
    tokens = SimpleNamespace(
        device=torch.device("cuda"),
        clone=lambda: torch.zeros(1, 7, dtype=torch.int64),
        numel=lambda: 7,
    )
    with (
        patch("torch.cuda.CUDAGraph"),
        patch("torch.cuda.Stream", return_value=stream),
        patch("torch.cuda.current_stream", return_value=stream),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch("torch.cuda.synchronize"),
    ):
        graph = DSparkGraph(
            lambda *_args: None,
            [],
            tokens,
            torch.zeros(1, 7, dtype=torch.int64),
            torch.zeros(1, 6, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            key=(True, 1, 4096),
        )
    assert graph.outputs is None
    captures = [
        record.message
        for record in caplog.records
        if "Capturing CUDA graph" in record.message
    ]
    assert captures == [
        "Capturing CUDA graph: family=dspark key=(True, 1, 4096) rows=7"
    ]


@pytest.mark.parametrize("change", ["features", "positions", "slots"])
def test_context_replay_rejects_changed_metadata_before_any_copy(change):
    graph = DSparkContextGraph.__new__(DSparkContextGraph)
    graph.features = torch.zeros(2, 4, dtype=torch.bfloat16)
    graph.positions = torch.zeros(2, dtype=torch.int64)
    graph.slots = torch.zeros(2, dtype=torch.int64)
    graph.graph = SimpleNamespace(replay=lambda: pytest.fail("replay must not start"))
    inputs = [
        torch.ones(2, 4).bfloat16(),
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.int64),
    ]
    index = ["features", "positions", "slots"].index(change)
    inputs[index] = inputs[index][:1]
    with (
        patch.object(DSparkContextGraph, "_validate_inputs"),
        pytest.raises(ValueError, match="changed"),
    ):
        graph.replay(*inputs)
    assert torch.count_nonzero(graph.features) == 0
    assert torch.count_nonzero(graph.positions) == 0
    assert torch.count_nonzero(graph.slots) == 0


@pytest.mark.parametrize("rows", [0, 33])
def test_context_capture_rejects_outside_small_batch_contract(rows):
    with pytest.raises(ValueError, match=r"1\.\.32"):
        DSparkContextGraph._validate_inputs(
            torch.zeros(rows, 4).bfloat16(), torch.arange(rows), torch.arange(rows)
        )


def test_context_capture_rejects_wrong_dtype_and_device():
    with pytest.raises(ValueError, match="BF16"):
        DSparkContextGraph._validate_inputs(
            torch.zeros(1, 4), torch.tensor([0]), torch.tensor([784])
        )
    with pytest.raises(ValueError, match="int64"):
        DSparkContextGraph._validate_inputs(
            torch.zeros(1, 4).bfloat16(),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([784]),
        )
    with pytest.raises(ValueError, match="CUDA device"):
        DSparkContextGraph._validate_inputs(
            torch.zeros(1, 4).bfloat16(), torch.tensor([0]), torch.tensor([784])
        )


def test_context_budget_keeps_all_supported_row_shapes_resident():
    worker, _ = make_worker()
    worker.compile_model = True
    worker.context_graph_cache.free_bytes = lambda: 8 << 30

    class Graph:
        MAX_ROWS = 32

        def __init__(self, unit, caches, *inputs, pool):
            self.unit, self.caches = unit, caches

        def replay(self, *inputs):
            return self.unit(*inputs, self.caches)

    with (
        patch("oh_my_vllm.worker.dspark_graph.DSparkContextGraph", Graph),
        patch("torch.cuda.graph_pool_handle", return_value=object()),
    ):
        for _ in range(3):
            for rows in range(1, 33):
                worker._inject(
                    torch.zeros(rows, 16).bfloat16(),
                    torch.arange(rows),
                    torch.arange(rows),
                )
        stats = worker.context_graph_cache.snapshot()
        assert stats["dspark_context_capture"] == 32
        assert stats["dspark_context_resident"] == 32
        assert stats["dspark_context_eviction"] == 0
        assert stats["dspark_context_recent_recapture"] == 0
        assert stats["dspark_context_hit"] == 64
        assert stats["dspark_context_budget_eager"] == 0
