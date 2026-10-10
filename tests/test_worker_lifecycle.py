"""Worker transaction fencing and complete resource-owner teardown on CPU."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from oh_my_vllm.worker.errors import WorkerUnavailable, failure_kind, transactional_step
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.model_runner import OhMyVllmWorker
from oh_my_vllm.worker.protocol import SchedulerOutput


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (torch.cuda.OutOfMemoryError("allocation failed"), "oom"),
        (MemoryError("host allocation failed"), "oom"),
        (RuntimeError("CUDA error: out of memory"), "oom"),
        (RuntimeError("CUDA error: illegal memory access"), "device"),
        (RuntimeError("cublas execution failed"), "device"),
        (ValueError("invalid input"), "internal"),
    ],
)
def test_failure_classification(error, kind):
    assert failure_kind(error) == kind


def test_execute_fences_invalid_schedule_before_next_command():
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {}
    worker.mtp = None
    # Invalid wire data is an escaping step failure. Rust rebuilds the worker.
    scheduled = SchedulerOutput([], [], [], 1)
    with pytest.raises(ValueError, match="token budget"):
        worker.execute_model(scheduled)
    assert worker._failure == "internal"
    with pytest.raises(WorkerUnavailable, match="rebuild required"):
        worker.execute_model(SchedulerOutput([], [], [], 0))
    with pytest.raises(WorkerUnavailable):
        worker.register_request(1, [7])
    with patch.object(torch.cuda, "synchronize") as synchronize:
        with pytest.raises(WorkerUnavailable):
            worker.resource_snapshot()
        with pytest.raises(WorkerUnavailable):
            worker.shutdown()
        synchronize.assert_not_called()


def test_registration_oom_fences_but_validation_keeps_worker_available():
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {}
    worker.samplers = {}
    from oh_my_vllm.worker.sampling import SamplingParams

    with pytest.raises(ValueError, match="invalid prompt"):
        worker.register_request(1, [], SamplingParams(2))
    assert getattr(worker, "_failure", None) is None
    with (
        patch(
            "oh_my_vllm.worker.model_runner.RequestSampler",
            side_effect=MemoryError("host OOM"),
        ),
        pytest.raises(MemoryError),
    ):
        worker.register_request(1, [7], SamplingParams(2))
    assert worker._failure == "oom"
    with pytest.raises(WorkerUnavailable):
        worker.register_request(2, [8], SamplingParams(2))


def test_prepare_registration_rejection_releases_generation():
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {}
    worker.samplers = {}
    worker.serving = SimpleNamespace(generations={})

    def prepare(request_id, _request):
        worker.serving.generations[request_id] = object()
        return [], None

    worker.serving.prepare = prepare
    with pytest.raises(ValueError, match="invalid prompt"):
        worker.prepare_request(1, {})
    assert worker.serving.generations == {}


@pytest.mark.parametrize("stage", ["grammar", "sampler", "state_copy", "proposal"])
def test_post_commit_oom_fences_incomplete_step(stage):
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {1: [5]}
    worker.samplers = {1: Mock()}
    worker.sources = {}
    worker.computed = {}
    worker.mtp = None
    worker.dspark = None
    generation = Mock(finished=None, reasoning_tokens=0)
    generation.consume.return_value = ([7], "seven")
    worker.serving = SimpleNamespace(generations={1: generation}, tokenizer=object())
    plan = SimpleNamespace(
        request=SimpleNamespace(request_id=1, num_computed_tokens=10),
        sample_indices=[0],
        drafts=[],
        commit=Mock(return_value=(1, 3, [])),
    )
    error = torch.cuda.OutOfMemoryError(f"fault at {stage}")
    if stage == "grammar":
        generation.consume.side_effect = error
    elif stage == "sampler":
        worker.samplers[1].commit.side_effect = error
    calls = []

    @transactional_step
    def step(self):
        calls.append("executed")
        self._commit_plans([plan], [0, 1], torch.empty(1, 1), [[7, 1]], {})
        if stage in ("state_copy", "proposal"):
            raise error

    with pytest.raises(torch.cuda.OutOfMemoryError):
        step(worker)
    assert worker._failure == "oom"
    with pytest.raises(WorkerUnavailable):
        step(worker)
    assert calls == ["executed"]
    if stage in ("state_copy", "proposal"):
        assert worker.histories[1] == [5, 7]
        assert worker.computed[1] == 11


def test_shutdown_releases_bound_methods_graph_pools_and_model():
    class Model:
        def forward(self):
            return None

    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    model = Model()
    reference = weakref.ref(model)
    worker.model = model
    worker.forward_unit = model.forward
    worker.logits_unit = model.forward
    worker.feature_unit = model.forward
    worker.histories = {1: [7]}
    worker.samplers = {1: object()}
    worker.sources = {1: 3}
    worker.computed = {1: 9}
    worker.serving = SimpleNamespace(generations={1: object()})
    worker.graph_cache = GraphCache(synchronize=Mock())
    worker.prefill_graph_cache = GraphCache(synchronize=Mock())
    worker.graph_cache.graphs[("target", (1,))] = model
    worker.mtp = None
    worker.dspark = None
    del model
    worker._release_resources()
    gc.collect()
    assert reference() is None
    assert worker.forward_unit is worker.logits_unit is worker.feature_unit is None
    assert not worker.histories and not worker.graph_cache.graphs
    assert all(
        getattr(worker, name) is None
        for name in (
            "graph_pool",
            "dspark_graph_pool",
            "prefill_graph_pool",
            "prefill_dspark_graph_pool",
        )
    )


def test_resource_snapshot_has_bounded_counts_and_separate_allocator_metrics():
    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {1: [7]}
    worker.samplers = {1: object()}
    worker.sources = {}
    worker.computed = {}
    worker.serving = SimpleNamespace(generations={1: object()})
    worker.graph_cache = GraphCache(synchronize=Mock())
    worker.prefill_graph_cache = GraphCache(synchronize=Mock())
    worker.mtp = None
    worker.dspark = None
    with (
        patch.object(torch.cuda, "synchronize") as synchronize,
        patch.object(torch.cuda, "mem_get_info", return_value=(100, 200)),
        patch.object(torch.cuda, "memory_allocated", return_value=11),
        patch.object(torch.cuda, "memory_reserved", return_value=22),
        patch.object(torch.cuda, "max_memory_allocated", return_value=33),
        patch.object(torch.cuda, "max_memory_reserved", return_value=44),
    ):
        stats = worker.resource_snapshot()
    synchronize.assert_called_once_with()
    assert stats["requests"]["histories"] == 1
    assert stats["requests"]["dspark_proposals"] == 0
    assert stats["graphs"]["target"]["capacity"] == 32
    assert stats["gpu"]["allocated_bytes"] == 11
    assert stats["gpu"]["reserved_bytes"] == 22
    assert stats["cpu"]["rss_bytes"] > 0
