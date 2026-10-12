"""Native CUDA graph destruction, pool retirement, and recapture memory checks."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.worker.graph_cache import GraphCache


def _default_segment_unused_capacity():
    return max(
        (
            segment["total_size"] - segment["allocated_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["segment_pool_id"] == (0, 0)
        ),
        default=0,
    )


@pytest.mark.gpu
def test_allocator_reclaim_retains_shared_graph_pool_and_live_output():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    pool = [torch.cuda.graph_pool_handle()]

    class Owner:
        def __init__(self):
            self.input = torch.ones(1 << 20, device="cuda")
            self.graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, pool=pool[0]):
                self.output = self.input * 2

    release = []
    cache = GraphCache(
        free_bytes=lambda: torch.cuda.mem_get_info()[0],
        release_memory=torch.cuda.empty_cache,
        release_family=lambda family: release.append(family),
        min_capture_free_bytes=0,
    )
    owner = cache.get_or_create("prefill_dspark", (1,), Owner)
    retained_pool = pool[0]
    torch.cuda.synchronize()
    # An inactive block can share a segment with a live tensor from an earlier
    # test. Request more than the unused capacity of every default-pool segment
    # so scratch receives a new segment that empty_cache can release in full.
    default_unused = _default_segment_unused_capacity()
    scratch = torch.empty(default_unused + (64 << 20), device="cuda", dtype=torch.uint8)
    scratch.fill_(1)
    torch.cuda.synchronize()
    del scratch
    cache.min_capture_free_bytes = torch.cuda.mem_get_info()[0] + (32 << 20)
    assert cache.should_use("prefill_dspark", (2,))
    assert cache.get_or_create("prefill_dspark", (1,), Owner) is owner
    assert pool[0] == retained_pool and not release
    assert cache.snapshot()["prefill_dspark_pressure_eviction"] == 0
    assert cache.snapshot()["allocator_reclaimed_bytes"] >= 32 << 20
    owner.input.fill_(3)
    owner.graph.replay()
    assert torch.equal(owner.output, torch.full_like(owner.output, 6))
    cache.clear()
    del owner, retained_pool
    pool[0] = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_allocator_reclaim_retains_graph_with_fragmented_default_pool():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    default_unused = _default_segment_unused_capacity()
    slab = torch.empty(default_unused + (256 << 20), device="cuda", dtype=torch.uint8)
    slab.fill_(1)
    torch.cuda.synchronize()
    del slab
    live = torch.full(
        (default_unused + (128 << 20),), 3, device="cuda", dtype=torch.uint8
    )
    try:
        segment = next(
            segment
            for segment in torch.cuda.memory_snapshot()
            if segment["address"]
            <= live.data_ptr()
            < segment["address"] + segment["total_size"]
        )
        assert segment["segment_pool_id"] == (0, 0)
        assert segment["total_size"] - segment["allocated_size"] >= 128 << 20
        test_allocator_reclaim_retains_shared_graph_pool_and_live_output()
        assert torch.equal(live, torch.full_like(live, 3))
    finally:
        del live
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.gpu
def test_pressure_retirement_waits_for_asynchronous_replay():
    side = torch.cuda.Stream()
    retired = []
    pools = [torch.cuda.graph_pool_handle()]

    class Owner:
        def __init__(self):
            self.input = torch.ones(32, device="cuda")
            self.graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, pool=pools[0]):
                self.output = self.input * 2

        def __del__(self):
            retired.append(side.query())

    cache = GraphCache(
        free_bytes=lambda: 10,
        release_memory=torch.cuda.empty_cache,
        release_family=lambda _family: pools.__setitem__(0, None),
        min_capture_free_bytes=0,
    )
    owner = cache.get_or_create("target_dspark", (1,), Owner)
    reference = weakref.ref(owner)
    with torch.cuda.stream(side):
        torch.cuda._sleep(20_000_000)
        owner.graph.replay()
    del owner
    cache.min_capture_free_bytes = 11
    cache.trim_for_headroom()
    assert reference() is None and retired == [True]
    assert pools[0] is None and not cache.graphs


@pytest.mark.gpu
def test_graph_pressure_recapture_returns_allocated_bytes_to_baseline():
    pool = [None]

    class Owner:
        def __init__(self):
            if pool[0] is None:
                pool[0] = torch.cuda.graph_pool_handle()
            self.input = torch.ones(1 << 20, device="cuda")
            self.graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, pool=pool[0]):
                self.output = self.input * 2

    def release_family(_family):
        pool[0] = None

    cache = GraphCache(
        2,
        synchronize=torch.cuda.synchronize,
        free_bytes=lambda: torch.cuda.mem_get_info()[0],
        release_memory=torch.cuda.empty_cache,
        release_family=release_family,
        min_capture_free_bytes=0,
        recent_capacity=8,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    for index in range(24):
        cache.min_capture_free_bytes = 0
        assert cache.should_use("target", (index,))
        owner = cache.get_or_create("target", (index,), Owner)
        owner.graph.replay()
        assert torch.equal(owner.output, torch.full_like(owner.output, 2))
        del owner
        # The graph owns at least 8 MiB of input and captured output. Require
        # another 4 MiB of driver headroom so pressure must retire that owner.
        cache.min_capture_free_bytes = torch.cuda.mem_get_info()[0] + (4 << 20)
        cache.trim_for_headroom()
        gc.collect()
        torch.cuda.synchronize()
        assert not cache.graphs and pool[0] is None
        assert torch.cuda.memory_allocated() <= baseline + (16 << 10)
    snapshot = cache.snapshot()
    assert snapshot["target_pressure_eviction"] == 24
    assert snapshot["capture_history"] <= snapshot["capture_history_capacity"]
    cache.clear()


@pytest.mark.gpu
def test_real_allocator_oom_after_commit_fences_further_steps():
    from oh_my_vllm.worker.errors import WorkerUnavailable, transactional_step
    from oh_my_vllm.worker.model_runner import OhMyVllmWorker
    from oh_my_vllm.worker.sampler import RequestSampler
    from oh_my_vllm.worker.sampling import SamplingParams

    worker = OhMyVllmWorker.__new__(OhMyVllmWorker)
    worker.histories = {1: [1]}
    worker.samplers = {
        1: RequestSampler(SamplingParams(4, temperature=0), [1], "cuda", vocab_size=4)
    }
    worker.sources = {}
    worker.computed = {}
    worker.serving = None
    worker.mtp = None
    worker.dspark = None
    plan = SimpleNamespace(
        request=SimpleNamespace(request_id=1, num_computed_tokens=1),
        sample_indices=[0],
        drafts=[],
        commit=lambda _count: (1, 3, []),
    )

    @transactional_step
    def step(self):
        self._commit_plans(
            [plan], [0, 1], torch.zeros(1, 4, device="cuda"), [[2, 1]], {}
        )
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        torch.empty(free + cached + (1 << 30), dtype=torch.uint8, device="cuda")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        step(worker)
    assert worker.histories[1] == [1, 2] and worker.computed[1] == 2
    assert worker._failure == "oom"
    with pytest.raises(WorkerUnavailable):
        step(worker)
    del worker
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_compiler_retirement_preserves_an_existing_cuda_graph(monkeypatch):
    from oh_my_vllm.ir import core
    from oh_my_vllm.ir.cache import CompilerCache

    cache = CompilerCache(4, per_code=2)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)
    unit = core.compile_forward(
        lambda x, offset: x + offset, unit="graph_retirement_test"
    )
    value = torch.ones(32, device="cuda")
    unit(value, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit(value, 1)
    execution_owners = tuple(unit.execution_owners)
    for offset in range(2, 12):
        assert torch.equal(unit(value, offset), torch.full_like(value, offset + 1))
    assert cache.snapshot()["cache_resets"] > 0
    # Remove registry roots and the original closure, like a graph constructor.
    del unit
    cache.clear()
    gc.collect()
    value.fill_(3)
    graph.replay()
    assert torch.equal(output, torch.full_like(value, 4))
    assert execution_owners
    del graph, output, value, execution_owners
    torch.cuda.synchronize()
    cache.clear()
