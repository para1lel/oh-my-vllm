"""Bound only source and dispatch metadata produced by owned compilations.

The pinned FX loader keeps source text after its code objects are executable.
The global FakeTensor dispatch cache keeps metadata, not tensor storage. A miss
reconstructs metadata through normal fake dispatch, including alias semantics.
ShapeEnv-local caches remain under their original lifetime and ownership.
"""

import dataclasses
import functools
import importlib
import linecache
import sys
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

_OWNER = ContextVar("oh_my_vllm_compiler_metadata_owner", default=None)
_INSERT_LINES = ContextVar("oh_my_vllm_fx_inserted_lines", default=None)
_FX_OWNERS = weakref.WeakValueDictionary()
_FAKE_OWNERS = weakref.WeakValueDictionary()
_PROFILER_OWNERS = weakref.WeakValueDictionary()
_LOCK = threading.RLock()
_INSTALLED = False


def _metadata_bytes(value, seen=None) -> int:
    """Account for owned Python metadata, without traversing shared runtime roots."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        return size + sum(_metadata_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return size + sum(
            _metadata_bytes(item, seen) for pair in value.items() for item in pair
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            _metadata_bytes(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    # Types, operator handles, dtype/device descriptors are shared. Count their
    # reference's object size but do not walk their modules, globals or storage.
    return size


def _install_hooks() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        fx = importlib.import_module("torch.fx.graph_module")
        fake = importlib.import_module("torch._subclasses.fake_tensor")
        traceback = importlib.import_module("torch.fx.traceback")
        original_fx = fx._loader.cache
        original_fake = fake._set_cache_key
        original_lines = linecache.updatecache
        original_lazy = linecache.lazycache
        original_profiler = traceback._register_fx_metadata

        @functools.wraps(original_lazy)
        def lazy_lines(filename, module_globals):
            inserted = _INSERT_LINES.get()
            if inserted is None:
                return original_lazy(filename, module_globals)
            with _LOCK:
                previous = linecache.cache.get(filename)
                owner = _FX_OWNERS.get(filename)
                old = owner.fx.get(filename) if owner is not None else None
                owned = previous is None or (old is not None and old[2] is previous)
                result = original_lazy(filename, module_globals)
                if inserted is not None and owned:
                    inserted[filename] = linecache.cache.get(filename)
                return result

        @functools.wraps(original_profiler)
        def register_profiler(key, metadata):
            with _LOCK:
                previous = _PROFILER_OWNERS.get(key)
                foreign = key in traceback._FX_METADATA_REGISTRY and previous is None
                original_profiler(key, metadata)
                owner = _OWNER.get()
                if previous is not None and previous is not owner:
                    previous.forget_profiler(key)
                if owner is not None and not foreign:
                    owner.track_profiler(key, metadata)

        @functools.wraps(original_lines)
        def update_lines(filename, module_globals=None):
            # Source ownership follows its registered key, even if inspect or
            # traceback hydrates it after the compile context has exited.
            owner = _FX_OWNERS.get(filename)
            if owner is None or filename not in owner.fx:
                # Arbitrary foreign loaders can acquire compiler/import locks.
                # Do not hold the metadata lock while running user callbacks.
                return original_lines(filename, module_globals)
            with _LOCK:
                previous = owner.fx.get(filename)
                current = linecache.cache.get(filename)
                owned_lines = previous is not None and current is previous[2]
                # Only hydrate our pinned FX loader's lazy source callback.
                # A materialized/foreign entry can call a different loader.
                known_loader = (
                    owned_lines
                    and current is not None
                    and len(current) == 1
                    and isinstance(current[0], functools.partial)
                    and getattr(current[0].func, "__self__", None) is fx._loader
                )
                if known_loader:
                    result = original_lines(filename, module_globals)
                    source, size, _ = owner.fx[filename]
                    owner.fx[filename] = source, size, linecache.cache.get(filename)
                    return result
            return original_lines(filename, module_globals)

        @functools.wraps(original_fx)
        def fx_cache(source, globals, co_fields=None):
            with _LOCK:
                explicit = None
                if co_fields and (
                    "co_firstlineno" not in co_fields or "co_name" not in co_fields
                ):
                    explicit = co_fields.get("co_filename")
                foreign = (
                    explicit in fx._loader.eval_cache and explicit not in _FX_OWNERS
                )
                inserted = {}
                token = _INSERT_LINES.set(inserted)
                try:
                    key = original_fx(source, globals, co_fields)
                finally:
                    _INSERT_LINES.reset(token)
                previous = _FX_OWNERS.get(key)
                owner = _OWNER.get()
                if previous is not None and previous is not owner:
                    previous.forget_fx(key)
                if owner is not None and not foreign:
                    owner.track_fx(key, source, inserted.get(key))
                return key

        @functools.wraps(original_fake)
        def fake_cache(cache, key, entry):
            # ShapeEnv entries can contain symbolic graph state. Only the
            # process-global, reconstructible dispatch cache is managed here.
            if cache is not fake.FakeTensorMode.cache:
                return original_fake(cache, key, entry)
            with _LOCK:
                previous = _FAKE_OWNERS.get(key)
                foreign = key in cache and previous is None
                original_fake(cache, key, entry)
                owner = _OWNER.get()
                if previous is not None and previous is not owner:
                    previous.forget_fake(key)
                if owner is not None and not foreign:
                    owner.track_fake(key, entry)

        fx._loader.cache = fx_cache
        fake._set_cache_key = fake_cache
        linecache.updatecache = update_lines
        linecache.lazycache = lazy_lines
        traceback._register_fx_metadata = register_profiler
        _INSTALLED = True


class CompilerMetadata:
    def __init__(
        self,
        *,
        fx_capacity: int = 256,
        fx_byte_capacity: int = 16 * 1024 * 1024,
        fake_capacity: int = 8192,
        fake_byte_capacity: int = 32 * 1024 * 1024,
    ) -> None:
        if min(fx_capacity, fx_byte_capacity, fake_capacity, fake_byte_capacity) <= 0:
            raise ValueError("compiler metadata budgets must be positive")
        self.fx_capacity = fx_capacity
        self.fx_byte_capacity = fx_byte_capacity
        self.fake_capacity = fake_capacity
        self.fake_byte_capacity = fake_byte_capacity
        self.fx = OrderedDict()
        self.fake = OrderedDict()
        self.profiler = OrderedDict()
        self.fx_bytes = self.fake_bytes = 0
        self.fx_evictions = self.fake_evictions = 0
        self.profiler_bytes = self.profiler_evictions = 0

    @staticmethod
    def scope_active() -> bool:
        return _OWNER.get() is not None

    @contextmanager
    def scope(self):
        _install_hooks()
        token = _OWNER.set(self)
        try:
            yield
        finally:
            _OWNER.reset(token)

    def track_fx(self, key, source, lines) -> None:
        self.forget_fx(key)
        # Account for the source and a fully materialized linecache copy, even
        # while its actual linecache entry is still a small lazy-loader tuple.
        size = sys.getsizeof(key) + sys.getsizeof(source)
        size += sys.getsizeof(source.splitlines(True)) + sum(
            sys.getsizeof(line) for line in source.splitlines(True)
        )
        self.fx[key] = source, size, lines
        self.fx_bytes += size
        _FX_OWNERS[key] = self

    def track_fake(self, key, entry) -> None:
        self.forget_fake(key)
        size = _metadata_bytes((key, entry))
        self.fake[key] = entry, size
        self.fake_bytes += size
        _FAKE_OWNERS[key] = self
        # Eviction only removes reusable metadata. Dispatch already produced
        # its output and does not need the just-inserted entry for execution.
        self.trim_fake()

    def track_profiler(self, key, metadata) -> None:
        self.forget_profiler(key)
        size = _metadata_bytes((key, metadata))
        self.profiler[key] = metadata, size
        self.profiler_bytes += size
        _PROFILER_OWNERS[key] = self

    def forget_profiler(self, key) -> None:
        previous = self.profiler.pop(key, None)
        if previous is not None:
            self.profiler_bytes -= previous[1]
        if _PROFILER_OWNERS.get(key) is self:
            _PROFILER_OWNERS.pop(key, None)

    def forget_fx(self, key) -> None:
        previous = self.fx.pop(key, None)
        if previous is not None:
            self.fx_bytes -= previous[1]
        if _FX_OWNERS.get(key) is self:
            _FX_OWNERS.pop(key, None)

    def forget_fake(self, key) -> None:
        previous = self.fake.pop(key, None)
        if previous is not None:
            self.fake_bytes -= previous[1]
        if _FAKE_OWNERS.get(key) is self:
            _FAKE_OWNERS.pop(key, None)

    def trim_fake(self, *, clear=False) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        while self.fake and (
            clear
            or len(self.fake) > self.fake_capacity
            or self.fake_bytes > self.fake_byte_capacity
        ):
            key = next(iter(self.fake))
            entry, _ = self.fake[key]
            if FakeTensorMode.cache.get(key) is entry:
                # Torch's GraphModule weakref finalizer also removes entries,
                # without going through the insertion lock.
                FakeTensorMode.cache.pop(key, None)
            self.forget_fake(key)
            self.fake_evictions += 1

    def trim(self, *, clear=False) -> int:
        """Retire source only after compilation has finished using inspection."""
        fx = importlib.import_module("torch.fx.graph_module")
        traceback = importlib.import_module("torch.fx.traceback")
        with _LOCK:
            before = self.fx_evictions + self.fake_evictions + self.profiler_evictions
            while self.fx and (
                clear
                or len(self.fx) > self.fx_capacity
                or self.fx_bytes > self.fx_byte_capacity
            ):
                key = next(iter(self.fx))
                source, _, lines = self.fx[key]
                if fx._loader.eval_cache.get(key) is source:
                    fx._loader.eval_cache.pop(key)
                if lines is not None and linecache.cache.get(key) is lines:
                    linecache.cache.pop(key, None)
                self.forget_fx(key)
                self.fx_evictions += 1
            self.trim_fake(clear=clear)
            while self.profiler and (
                clear
                or len(self.profiler) > self.fx_capacity
                or self.profiler_bytes > self.fx_byte_capacity
            ):
                key = next(iter(self.profiler))
                metadata, _ = self.profiler[key]
                if traceback._FX_METADATA_REGISTRY.get(key) is metadata:
                    traceback._FX_METADATA_REGISTRY.pop(key)
                self.forget_profiler(key)
                self.profiler_evictions += 1
            return (
                self.fx_evictions
                + self.fake_evictions
                + self.profiler_evictions
                - before
            )

    def snapshot(self) -> dict[str, int]:
        # Foreign threads also use the pinned insertion hooks. Keep a stats
        # request from racing their dictionary writes or ownership transfers.
        with _LOCK:
            return self._snapshot()

    def _snapshot(self) -> dict[str, int]:
        fx = importlib.import_module("torch.fx.graph_module")
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Torch's own weakref cleanup can mutate these dictionaries without our
        # hook lock. Materialize keys before any Python-level iteration.
        fx_keys = tuple(fx._loader.eval_cache)
        fake_keys = tuple(FakeTensorMode.cache)
        return {
            "fx_source_entries": len(self.fx),
            "fx_source_capacity": self.fx_capacity,
            "fx_source_bytes": self.fx_bytes,
            "fx_source_byte_capacity": self.fx_byte_capacity,
            "fx_source_evictions": self.fx_evictions,
            "total_fx_source_entries": len(fx_keys),
            "foreign_fx_source_entries": sum(key not in self.fx for key in fx_keys),
            "owned_fx_linecache_entries": sum(
                row[2] is not None and linecache.cache.get(key) is row[2]
                for key, row in self.fx.items()
            ),
            "fake_tensor_entries": len(self.fake),
            "fake_tensor_capacity": self.fake_capacity,
            "fake_tensor_bytes": self.fake_bytes,
            "fake_tensor_byte_capacity": self.fake_byte_capacity,
            "fake_tensor_evictions": self.fake_evictions,
            "total_fake_tensor_entries": len(fake_keys),
            "foreign_fake_tensor_entries": sum(
                key not in self.fake for key in fake_keys
            ),
            "fx_profiler_entries": len(self.profiler),
            "fx_profiler_capacity": self.fx_capacity,
            "fx_profiler_bytes": self.profiler_bytes,
            "fx_profiler_byte_capacity": self.fx_byte_capacity,
            "fx_profiler_evictions": self.profiler_evictions,
        }
