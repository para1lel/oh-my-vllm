"""Real captured executables survive compiler source and metadata retirement."""

import pytest
import torch
from oh_my_vllm.ir import core
from oh_my_vllm.ir.cache import CompilerCache
from oh_my_vllm.ir.metadata_cache import CompilerMetadata


@pytest.mark.gpu
def test_owned_metadata_retirement_preserves_capture_and_cold_recapture(monkeypatch):
    metadata = CompilerMetadata(fx_capacity=1, fake_capacity=16)
    cache = CompilerCache(2, per_code=1, gc_batch=1, metadata=metadata)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)

    def function(x, offset):
        return x.square() + offset

    unit = core.compile_forward(function, unit="metadata_capture_test")
    value = torch.ones(32, device="cuda")
    unit(value, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit(value, 1)
    lease = tuple(unit.execution_owners)
    for offset in range(2, 12):
        assert torch.equal(unit(value, offset), torch.full_like(value, offset + 1))
        snapshot = cache.snapshot()
        assert snapshot["fx_source_entries"] <= snapshot["fx_source_capacity"]
        assert snapshot["fake_tensor_entries"] <= snapshot["fake_tensor_capacity"]
    assert metadata.fx_evictions > 0 and cache.full_gc_collections > 0
    del unit
    # Shutdown/retirement itself must collect cycles. No extra test-side GC.
    cache.clear()
    value.fill_(3)
    graph.replay()
    assert torch.equal(output, torch.full_like(value, 10))
    assert lease
    unit = core.compile_forward(function, unit="metadata_cold_recapture_test")
    unit(value, 2)
    torch.cuda.synchronize()
    recaptured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(recaptured):
        new_output = unit(value, 2)
    new_lease = tuple(unit.execution_owners)
    value.fill_(4)
    graph.replay()
    recaptured.replay()
    assert torch.equal(output, torch.full_like(value, 17))
    assert torch.equal(new_output, torch.full_like(value, 18))
    del graph, recaptured, output, new_output, value, lease, new_lease, unit
    torch.cuda.synchronize()
    cache.clear()
    torch.cuda.empty_cache()
