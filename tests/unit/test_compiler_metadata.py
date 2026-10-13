"""Compiler ownership and rebuildable metadata budgets, without CUDA work."""

import gc
import importlib
import linecache
import threading
import weakref

import pytest
import torch
from oh_my_vllm.ir import core
from oh_my_vllm.ir.cache import CompilerCache
from oh_my_vllm.ir.metadata_cache import CompilerMetadata
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture
def isolated_metadata(monkeypatch):
    fx = importlib.import_module("torch.fx.graph_module")
    monkeypatch.setattr(fx._loader, "eval_cache", {})
    monkeypatch.setattr(linecache, "cache", {})
    monkeypatch.setattr(FakeTensorMode, "cache", {})
    traceback = importlib.import_module("torch.fx.traceback")
    monkeypatch.setattr(traceback, "_FX_METADATA_REGISTRY", {})
    return fx._loader


def test_fx_count_and_byte_limits_preserve_running_and_serializable_graphs(
    isolated_metadata,
):
    metadata = CompilerMetadata(fx_capacity=2, fx_byte_capacity=2500)
    foreign = torch.fx.symbolic_trace(lambda x: x + 10)
    foreign_key = foreign.forward.__code__.co_filename
    with metadata.scope():
        graphs = [torch.fx.symbolic_trace(lambda x: x.sin() + 1) for _ in range(12)]
    old_key = graphs[0].forward.__code__.co_filename
    assert old_key in metadata.fx
    metadata.trim()
    assert old_key not in isolated_metadata.eval_cache
    assert old_key not in linecache.cache
    assert metadata.fx_bytes <= 2500
    assert len(metadata.fx) <= 2
    assert foreign_key in isolated_metadata.eval_cache
    # Source retirement does not remove graph code or executable functions.
    graph = graphs[0]
    value = torch.arange(3.0)
    assert torch.equal(graph(value), value.sin() + 1)
    import pickle

    restored = pickle.loads(pickle.dumps(graph))
    assert torch.equal(restored(value), graph(value))
    with metadata.scope():
        graph.recompile()
    metadata.trim()
    assert torch.equal(graph(value), value.sin() + 1)


def test_fx_oversized_source_is_retired_without_discarding_foreign_source(
    isolated_metadata,
):
    metadata = CompilerMetadata(fx_byte_capacity=64)
    foreign = isolated_metadata.cache("foreign source", {})
    with metadata.scope():
        key = isolated_metadata.cache("a" * 5000, {})
    metadata.trim()
    assert key not in isolated_metadata.eval_cache
    assert foreign in isolated_metadata.eval_cache
    assert metadata.snapshot()["fx_source_bytes"] == 0


def test_foreign_thread_sources_are_not_claimed(isolated_metadata):
    metadata = CompilerMetadata(fx_capacity=1)
    keys = []

    def foreign_compile():
        graph = torch.fx.symbolic_trace(lambda x: x * 3)
        keys.append(graph.forward.__code__.co_filename)

    with metadata.scope():
        thread = threading.Thread(target=foreign_compile)
        thread.start()
        owned = torch.fx.symbolic_trace(lambda x: x + 2)
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert owned.forward.__code__.co_filename in metadata.fx
    assert keys[0] not in metadata.fx
    metadata.trim(clear=True)
    assert keys[0] in isolated_metadata.eval_cache


@pytest.mark.parametrize("same_source", [False, True])
def test_foreign_explicit_source_replacement_releases_ownership(
    isolated_metadata, same_source
):
    metadata = CompilerMetadata()
    source = "owned source"
    with metadata.scope():
        key = isolated_metadata.cache(source, {}, {"co_filename": "owned.py"})
    replacement = source if same_source else "foreign source"
    isolated_metadata.cache(replacement, {}, {"co_filename": key})
    metadata.trim(clear=True)
    assert isolated_metadata.eval_cache[key] is replacement
    assert key in linecache.cache
    assert key not in metadata.fx


def test_compilation_scope_includes_dynamo_source_created_before_lower(
    isolated_metadata, monkeypatch
):
    metadata = CompilerMetadata(fx_capacity=2)
    cache = CompilerCache(4, per_code=1, metadata=metadata)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(torch._dynamo.config, "specialize_int", True)
    observed = []

    def lower(graph, _inputs, **_kwargs):
        # Dynamo supplies a lazy GraphModule: its public forward may still be
        # the shared lazy trampoline, but it generated source before lower.
        observed.append(tuple(metadata.fx))
        assert any("def forward" in row[0] for row in metadata.fx.values())
        return graph.forward

    monkeypatch.setattr(core, "lower_to_inductor", lower)
    unit = core.compile_forward(lambda x, shift: x + shift, unit="metadata_pre_lower")
    for shift in range(8):
        assert torch.equal(unit(torch.ones(2), shift), torch.full((2,), shift + 1.0))
        assert len(metadata.fx) <= 2
    assert len(observed) == 8
    del unit
    cache.clear()
    assert not metadata.fx


def test_fx_foreign_linecache_replacement_is_preserved(isolated_metadata):
    metadata = CompilerMetadata()
    with metadata.scope():
        key = isolated_metadata.cache("owned source", {})
    foreign_lines = (8, None, ["foreign\n"], key)
    linecache.cache[key] = foreign_lines
    metadata.trim(clear=True)
    assert key not in isolated_metadata.eval_cache
    assert linecache.cache[key] is foreign_lines


def test_fx_owned_materialized_linecache_is_retired(isolated_metadata):
    metadata = CompilerMetadata()
    with metadata.scope():
        # Synthetic names must look like loader-backed filenames, not <...>.
        key = isolated_metadata.cache("def example():\n    return 3\n", {})
        assert linecache.getlines(key)
        assert len(linecache.cache[key]) == 4
    metadata.trim(clear=True)
    assert key not in linecache.cache


@pytest.mark.parametrize("other_owner", [False, True])
def test_fx_materialized_source_outside_its_scope_keeps_exact_ownership(
    isolated_metadata, other_owner
):
    metadata = CompilerMetadata()
    with metadata.scope():
        key = isolated_metadata.cache("def example():\n    return 3\n", {})
    if other_owner:
        with CompilerMetadata().scope():
            assert linecache.getlines(key)
    else:
        assert linecache.getlines(key)
    assert len(linecache.cache[key]) == 4
    assert metadata.snapshot()["owned_fx_linecache_entries"] == 1
    metadata.trim(clear=True)
    assert key not in linecache.cache


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("same_source", [False, True])
def test_owned_fx_source_transfer_preserves_linecache_ownership(
    isolated_metadata, materialized, same_source
):
    first = CompilerMetadata()
    second = CompilerMetadata()
    key = "project_transfer.py"
    source = "def example():\n    return 3\n"
    with first.scope():
        isolated_metadata.cache(source, {}, {"co_filename": key})
    if materialized:
        assert linecache.getlines(key)
    lines = linecache.cache[key]
    replacement = source if same_source else "def example():\n    return 4\n"
    with second.scope():
        isolated_metadata.cache(replacement, {}, {"co_filename": key})
    assert key not in first.fx
    assert second.snapshot()["owned_fx_linecache_entries"] == 1
    first.trim(clear=True)
    assert isolated_metadata.eval_cache[key] is replacement
    assert linecache.cache[key] is lines
    second.trim(clear=True)
    assert key not in isolated_metadata.eval_cache and key not in linecache.cache


def test_metadata_snapshots_are_safe_with_concurrent_foreign_writes(
    isolated_metadata,
):
    from torch._subclasses import fake_tensor

    metadata = CompilerMetadata()
    with metadata.scope():
        isolated_metadata.cache("owned", {})
    stop = threading.Event()
    failures = []

    def foreign_writes():
        try:
            for index in range(3000):
                if stop.is_set():
                    return
                key = fake_tensor._DispatchCacheKey(("foreign", index))
                entry = fake_tensor._DispatchCacheBypassEntry("foreign metadata")
                fake_tensor._set_cache_key(FakeTensorMode.cache, key, entry)
                isolated_metadata.cache(str(index), {})
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=foreign_writes)
    thread.start()
    try:
        for _ in range(300):
            snapshot = metadata.snapshot()
            assert snapshot["fx_source_entries"] == 1
            assert snapshot["foreign_fake_tensor_entries"] >= 0
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and not failures


def test_metadata_snapshots_tolerate_torch_unhooked_weakref_eviction(
    isolated_metadata,
):
    from torch._subclasses import fake_tensor

    metadata = CompilerMetadata()
    keys = [fake_tensor._DispatchCacheKey(("foreign", i)) for i in range(6000)]
    entry = fake_tensor._DispatchCacheBypassEntry("foreign metadata")
    FakeTensorMode.cache.update((key, entry) for key in keys)
    started = threading.Event()

    def finalizers():
        started.wait(timeout=10)
        for key in keys:
            fake_tensor.evict_fake_tensor_cache_key(key)

    thread = threading.Thread(target=finalizers)
    thread.start()
    try:
        started.set()
        for _ in range(100):
            snapshot = metadata.snapshot()
            assert (
                snapshot["total_fake_tensor_entries"]
                == snapshot["foreign_fake_tensor_entries"]
            )
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()


def test_fake_dispatch_rebuild_preserves_values_metadata_views_and_inplace(
    isolated_metadata,
):
    metadata = CompilerMetadata(fake_capacity=2, fake_byte_capacity=4096)
    mode = FakeTensorMode()
    with metadata.scope(), mode:
        for size in range(3, 20):
            tensor = torch.empty(size, 2)
            view = tensor.view(2, size)
            clone = tensor.clone()
            assert view.shape == (2, size)
            assert view.stride() == (size, 1)
            assert torch._C._is_alias_of(view, tensor)
            assert not torch._C._is_alias_of(clone, tensor)
            assert tensor.add_(2) is tensor
            assert len(metadata.fake) <= 2
            assert metadata.fake_bytes <= 4096
        metadata.trim(clear=True)
        assert not metadata.fake
        view = tensor.view(2, size)
        assert torch._C._is_alias_of(view, tensor)
        assert tensor.add_(1) is tensor
    assert metadata.fake_evictions > 10


def test_fake_byte_limit_foreign_replacement_and_local_cache_are_preserved(
    isolated_metadata,
):
    from torch._subclasses import fake_tensor

    metadata = CompilerMetadata(fake_byte_capacity=32)
    key = fake_tensor._DispatchCacheKey(("owned",))
    entry = fake_tensor._DispatchCacheBypassEntry("metadata rebuild")
    foreign_key = fake_tensor._DispatchCacheKey(("foreign",))
    FakeTensorMode.cache[foreign_key] = entry
    local = {}
    with metadata.scope():
        fake_tensor._set_cache_key(FakeTensorMode.cache, key, entry)
        fake_tensor._set_cache_key(local, key, entry)
        fake_tensor._set_cache_key(FakeTensorMode.cache, foreign_key, entry)
    assert key not in FakeTensorMode.cache
    assert local[key] is entry
    assert FakeTensorMode.cache[foreign_key] is entry
    assert not metadata.fake


def test_fake_foreign_thread_write_and_same_entry_overwrite_are_not_evicted(
    isolated_metadata,
):
    from torch._subclasses import fake_tensor

    metadata = CompilerMetadata(fake_capacity=8)
    entry = fake_tensor._DispatchCacheBypassEntry("rebuild")
    owned_key = fake_tensor._DispatchCacheKey(("owned",))
    foreign_key = fake_tensor._DispatchCacheKey(("foreign",))
    with metadata.scope():
        fake_tensor._set_cache_key(FakeTensorMode.cache, owned_key, entry)
        thread = threading.Thread(
            target=fake_tensor._set_cache_key,
            args=(FakeTensorMode.cache, foreign_key, entry),
        )
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert owned_key in metadata.fake
    assert foreign_key not in metadata.fake
    fake_tensor._set_cache_key(FakeTensorMode.cache, owned_key, entry)
    metadata.trim(clear=True)
    assert FakeTensorMode.cache[owned_key] is entry
    assert FakeTensorMode.cache[foreign_key] is entry


def test_full_gc_runs_only_at_non_capture_boundaries_and_reclaims_old_cycles(
    isolated_metadata, monkeypatch
):
    metadata = CompilerMetadata(fx_capacity=1)
    cache = CompilerCache(gc_batch=2, metadata=metadata)
    capturing = True
    monkeypatch.setattr(cache, "_capturing", lambda: capturing)

    class Cycle:
        pass

    obj = Cycle()
    obj.self = obj
    reference = weakref.ref(obj)
    # Promote while still strongly owned. Production must later collect the
    # generation-2 cycle, without a test collection after its owner is dropped.
    gc.collect(2)
    del obj
    cache.pending_retirements = 2
    with metadata.scope():
        keys = [isolated_metadata.cache(str(i), {}) for i in range(3)]
    cache.after_call()
    assert reference() is not None
    assert cache.full_gc_collections == 0
    assert all(key in isolated_metadata.eval_cache for key in keys)
    capturing = False
    cache.after_call()
    assert reference() is None
    assert cache.full_gc_collections == 1
    assert len(metadata.fx) == 1
    cache.after_call()
    assert cache.full_gc_collections == 1


def test_nested_scope_defers_retirement_and_restores_outer_owner(isolated_metadata):
    outer = CompilerMetadata(fx_capacity=1)
    inner = CompilerMetadata(fx_capacity=1)
    cache = CompilerCache(gc_batch=1, metadata=inner)
    with outer.scope():
        first = isolated_metadata.cache("first", {})
        with inner.scope():
            for value in range(3):
                isolated_metadata.cache(str(value), {})
        cache.pending_retirements = 1
        cache.after_call()
        assert cache.full_gc_collections == 0
        assert len(inner.fx) == 3
        last = isolated_metadata.cache("last", {})
    cache.after_call()
    assert len(inner.fx) == 1
    assert first in outer.fx and last in outer.fx


def test_gc_finalizer_metadata_is_foreign_after_invocation_scope_ends(
    isolated_metadata, monkeypatch
):
    metadata = CompilerMetadata(fx_capacity=1)
    cache = CompilerCache(gc_batch=1, metadata=metadata)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)
    monkeypatch.setattr(core, "lower_to_inductor", lambda graph, _, **kw: graph.forward)
    keys = []

    class Finalizer:
        def __del__(self):
            keys.append(isolated_metadata.cache("foreign finalizer source", {}))

    obj = Finalizer()
    obj.self = obj
    gc.collect(2)
    del obj
    cache.pending_retirements = 1
    unit = core.compile_forward(lambda x: x + 1, unit="foreign_finalizer_test")
    unit(torch.ones(2))
    assert len(keys) == 1
    assert keys[0] not in metadata.fx
    assert keys[0] in isolated_metadata.eval_cache


def test_optional_fx_profiler_registry_has_owned_count_and_byte_limits(
    isolated_metadata,
):
    traceback = importlib.import_module("torch.fx.traceback")
    metadata = CompilerMetadata(fx_capacity=2, fx_byte_capacity=4096)
    foreign = {"node_metadata": {0: {"name": "foreign"}}}
    traceback._register_fx_metadata("foreign", foreign)
    with metadata.scope():
        for index in range(12):
            traceback._register_fx_metadata(str(index), {"node_metadata": index})
    metadata.trim()
    assert len(metadata.profiler) == 2
    assert metadata.profiler_bytes <= 4096
    assert traceback._FX_METADATA_REGISTRY["foreign"] is foreign
    # A foreign overwrite of exactly the same value transfers ownership too.
    key = next(iter(metadata.profiler))
    value = metadata.profiler[key][0]
    traceback._register_fx_metadata(key, value)
    metadata.trim(clear=True)
    assert traceback._FX_METADATA_REGISTRY[key] is value
    assert not metadata.profiler


def test_fx_preserves_foreign_linecache_that_precedes_its_source(isolated_metadata):
    metadata = CompilerMetadata()
    key = "shared_source.py"
    foreign_lines = (8, None, ["foreign\n"], key)
    linecache.cache[key] = foreign_lines
    with metadata.scope():
        isolated_metadata.cache("owned source", {}, {"co_filename": key})
    metadata.trim(clear=True)
    assert key not in isolated_metadata.eval_cache
    assert linecache.cache[key] is foreign_lines


def test_foreign_loader_can_compile_without_metadata_lock_inversion(
    isolated_metadata,
):
    metadata = CompilerMetadata()
    completed = threading.Event()

    class Loader:
        def get_source(self, _name):
            def compile_source():
                # Another thread may already hold the Dynamo compile lock.
                # It must still enter the owned FX insertion hook here.
                with metadata.scope():
                    isolated_metadata.cache("owned source from another thread", {})
                completed.set()

            thread = threading.Thread(target=compile_source, daemon=True)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive(), "linecache callback held metadata lock"
            return "foreign loader source\n"

    key = "foreign_loader.py"
    linecache.lazycache(key, {"__name__": key, "__loader__": Loader()})
    assert linecache.getlines(key) == ["foreign loader source\n"]
    assert completed.is_set()


def test_owned_source_with_foreign_linecache_loader_is_not_locked(
    isolated_metadata,
):
    metadata = CompilerMetadata()

    class Loader:
        def get_source(self, _name):
            completed = threading.Event()

            def compile_source():
                with metadata.scope():
                    isolated_metadata.cache("another owned source", {})
                completed.set()

            thread = threading.Thread(target=compile_source, daemon=True)
            thread.start()
            thread.join(timeout=2)
            assert completed.is_set(), "foreign lazy loader ran under metadata lock"
            return "foreign source\n"

    key = "owned_eval_foreign_loader.py"
    linecache.lazycache(key, {"__name__": key, "__loader__": Loader()})
    with metadata.scope():
        isolated_metadata.cache("owned source", {}, {"co_filename": key})
        assert linecache.getlines(key) == ["foreign source\n"]
    foreign_lines = linecache.cache[key]
    metadata.trim(clear=True)
    assert linecache.cache[key] is foreign_lines


def test_foreign_lazy_loader_property_can_compile_without_lock_inversion(
    isolated_metadata,
):
    metadata = CompilerMetadata()

    class Loader:
        @property
        def get_source(self):
            completed = threading.Event()

            def compile_source():
                with metadata.scope():
                    isolated_metadata.cache("owned source in loader property", {})
                completed.set()

            thread = threading.Thread(target=compile_source, daemon=True)
            thread.start()
            thread.join(timeout=2)
            assert completed.is_set(), "foreign loader property held metadata lock"
            return lambda _name: "foreign source\n"

    key = "foreign_property.py"
    assert linecache.lazycache(key, {"__name__": key, "__loader__": Loader()})
    foreign_lines = linecache.cache[key]
    metadata.trim(clear=True)
    assert linecache.cache[key] is foreign_lines


def test_real_enriched_fx_graphs_obey_profiler_budget_and_still_run(
    isolated_metadata, monkeypatch
):
    from torch.fx.experimental import _config

    monkeypatch.setattr(_config, "enrich_profiler_metadata", True)
    metadata = CompilerMetadata(fx_capacity=2)
    graphs = []
    with metadata.scope():
        for index in range(8):
            graph = torch.fx.symbolic_trace(lambda x: x + 1)
            # Different node metadata yields different profiler source keys.
            next(node for node in graph.graph.nodes if node.op == "call_function").meta[
                "stack_trace"
            ] = str(index)
            graph.recompile()
            graphs.append(graph)
    assert len(metadata.profiler) >= 8
    metadata.trim()
    assert len(metadata.profiler) <= 2
    assert metadata.profiler_bytes <= metadata.fx_byte_capacity
    assert torch.equal(graphs[0](torch.ones(3)), torch.full((3,), 2.0))


def test_exception_boundary_retires_owned_metadata(monkeypatch, isolated_metadata):
    metadata = CompilerMetadata(fx_capacity=1)
    cache = CompilerCache(metadata=metadata)
    monkeypatch.setattr(core, "COMPILER_CACHE", cache)

    def lower(graph, _inputs, **_kwargs):
        for index in range(3):
            isolated_metadata.cache(str(index), {})
        raise ValueError("test compiler failure")

    monkeypatch.setattr(core, "lower_to_inductor", lower)
    unit = core.compile_forward(lambda x: x + 2, unit="metadata_failure")
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed):
        unit(torch.ones(2))
    assert len(metadata.fx) <= 1
    del unit
    cache.clear()
