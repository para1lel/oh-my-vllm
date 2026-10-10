"""Real Dynamo churn through bounded fullgraph caches, without GPU work."""

import gc
import sys
import weakref
from types import ModuleType

import pytest
import torch
from oh_my_vllm.ir import core
from oh_my_vllm.ir.cache import CompilerCache


def test_scoped_retirement_prevents_cumulative_fullgraph_failure(monkeypatch):
    cache = CompilerCache(4, per_code=2)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(core, "_RECOMPILE_LIMIT", 4)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 4)
    monkeypatch.setattr(torch._dynamo.config, "accumulated_recompile_limit", 4)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)
    monkeypatch.setattr(core, "lower_to_inductor", lambda graph, _, **kw: graph.forward)

    # A Python extent influences output semantics, so Dynamo must specialize it.
    def function(x, extent):
        return x + extent

    unit = core.compile_forward(function, unit="bounded_churn_test")
    before = core.compiled_graph_counts().get("bounded_churn_test", 0)
    for extent in range(24):
        assert torch.equal(unit(torch.ones(2), extent), torch.full((2,), 1.0 + extent))
    assert core.compiled_graph_counts()["bounded_churn_test"] - before >= 12
    snapshot = cache.snapshot()
    assert snapshot["code_cache_entries"] <= 4
    assert snapshot["cache_resets"] >= 5
    assert sum(len(state.frame_ids) for _, state in cache.codes.values()) <= 4
    cache.clear()


def test_module_registry_eviction_releases_only_registry_roots():
    from torch._inductor.codecache import PyCodeCache

    # Supply isolated registry objects, not the global modules of other tests.
    with pytest.MonkeyPatch.context() as patch:
        modules = []
        references = []
        for index in range(5):
            module = ModuleType(f"test_bounded_inductor_{index}")
            module.__file__ = f"test_bounded_{index}.py"
            modules.append(module)
            references.append(weakref.ref(module))
            patch.setitem(sys.modules, module.__name__, module)
        del module
        patch.setattr(PyCodeCache, "modules", modules)
        patch.setattr(PyCodeCache, "modules_no_attr", {m.__file__: m for m in modules})
        patch.setattr(PyCodeCache, "linemaps", {m.__file__: [] for m in modules})
        cache = CompilerCache(2)
        cache.modules.update((id(module), module) for module in modules[:4])
        foreign = modules[4]
        retained_callable_owner = modules[1]
        cache.trim_modules()
        gc.collect()
        assert len(PyCodeCache.modules) == len(PyCodeCache.linemaps) == 3
        assert foreign in PyCodeCache.modules
        assert references[0]() is None
        assert references[1]() is retained_callable_owner
        del retained_callable_owner
        gc.collect()
        assert references[1]() is None
        cache.clear()
        assert PyCodeCache.modules == [foreign]
        assert sys.modules[foreign.__name__] is foreign


def test_dispatch_metadata_is_bounded_without_changing_selection(monkeypatch):
    from collections import OrderedDict, deque

    monkeypatch.setattr(core, "_SELECTIONS", OrderedDict())
    monkeypatch.setattr(core, "_SELECTION_CAPACITY", 3)
    monkeypatch.setattr(core, "_REWRITES", deque(maxlen=2))
    for index in range(100):
        core._record_selection("fp8_linear", "cuda", "compile", str(index))
        core._REWRITES.append(core.Rewrite("silu_mul+fp8_linear", index))
    assert len(core.selected_implementations()) == 3
    assert [row.static_key for row in core.selected_implementations()] == [
        "97",
        "98",
        "99",
    ]
    assert [row.count for row in core.applied_rewrites()] == [98, 99]
    assert {row.provider for row in core.selected_implementations()} == {"cuda"}


def test_dead_compile_units_release_owned_backend_registry_roots(monkeypatch):
    from torch._dynamo import eval_frame

    cache = CompilerCache(4)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    before = dict(eval_frame.cached_backends)
    for _ in range(32):
        unit = core.compile_forward(lambda x: x + 1, unit="ephemeral_unit_test")
        del unit
        gc.collect()
    assert cache.snapshot()["backend_entries"] == 0
    assert all(
        eval_frame.cached_backends.get(key) is value for key, value in before.items()
    )


def test_real_inductor_churn_retires_owned_modules_and_keeps_fullgraph(monkeypatch):
    cache = CompilerCache(2, per_code=1)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)

    def function(x, offset):
        return x.sin() + offset

    unit = core.compile_forward(function, unit="real_inductor_churn_test")
    value = torch.ones(7)
    for offset in range(73331, 73337):
        assert torch.equal(unit(value, offset), value.sin() + offset)
        assert cache.snapshot()["module_entries"] <= 2
    snapshot = cache.snapshot()
    assert snapshot["cache_resets"] >= 5
    assert snapshot["module_evictions"] >= 4
    del unit
    cache.clear()


def test_long_lived_unit_bounds_executable_owners_and_preserves_capture_lease(
    monkeypatch,
):
    cache = CompilerCache(4, per_code=2)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)
    references = []

    def lower(graph, _inputs, **_kwargs):
        compiled = graph.forward
        references.append(weakref.ref(compiled))
        return compiled

    monkeypatch.setattr(core, "lower_to_inductor", lower)
    unit = core.compile_forward(
        lambda x, offset: x + offset, unit="bounded_owners_test"
    )
    unit(torch.ones(2), 0)
    captured_owners = tuple(unit.execution_owners)
    for offset in range(1, 30):
        assert torch.equal(unit(torch.ones(2), offset), torch.full((2,), offset + 1.0))
        assert len(unit.execution_owners) <= cache.per_code
        gc.collect()
        assert sum(reference() is not None for reference in references) <= 3
    assert captured_owners
    del captured_owners, unit
    cache.clear()
    gc.collect()
    assert not any(reference() is not None for reference in references)


def test_graph_inventory_grows_only_for_explicit_semantic_traces(monkeypatch):
    from unittest.mock import Mock

    from oh_my_vllm.performance import execution

    observer = Mock()
    monkeypatch.setattr(execution, "observe_graph", observer)
    monkeypatch.setattr(
        torch._dynamo, "lookup_backend", lambda _name: lambda gm, _args: gm.forward
    )
    monkeypatch.delenv("OH_MY_VLLM_SEMANTIC_TRACE", raising=False)
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    core.lower_to_inductor(graph, [torch.ones(2)], unit="quiet_inventory_test")
    observer.assert_not_called()
    monkeypatch.setenv("OH_MY_VLLM_SEMANTIC_TRACE", "test-owned-trace.json")
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    core.lower_to_inductor(graph, [torch.ones(2)], unit="recorded_inventory_test")
    observer.assert_called_once_with("recorded_inventory_test", graph.graph)
