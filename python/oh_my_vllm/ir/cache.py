"""Bound the pinned compiler's live code caches without changing fullgraph mode.

Manual CUDA graphs own their captured executable and buffers. Dropping Dynamo
specializations at a call boundary does not destroy these graphs. A later host
invocation compiles again rather than falling back to eager execution.
"""

import gc
import sys
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from types import CodeType

import torch

from .metadata_cache import CompilerMetadata


@dataclass
class CodeState:
    variants: int = 0
    frame_ids: set[int] = field(default_factory=set)


class CompilerCache:
    def __init__(
        self,
        capacity: int = 256,
        *,
        per_code: int = 64,
        code_capacity: int = 32,
        gc_batch: int = 8,
        metadata: CompilerMetadata | None = None,
    ) -> None:
        if min(capacity, per_code, code_capacity, gc_batch) <= 0:
            raise ValueError("compiler cache budgets must be positive")
        self.capacity = capacity
        self.per_code = per_code
        self.code_capacity = code_capacity
        # Code objects compare structurally. Use identity, as Dynamo does.
        self.codes: OrderedDict[int, tuple[CodeType, CodeState]] = OrderedDict()
        self.modules: OrderedDict[int, object] = OrderedDict()
        self.backends: dict[int, weakref.ReferenceType] = {}
        self.resets = 0
        self.module_evictions = 0
        self.metadata = metadata or CompilerMetadata()
        self.gc_batch = gc_batch
        self.pending_retirements = 0
        self.full_gc_collections = 0
        self._metadata_retirements = 0

    @staticmethod
    def _capturing() -> bool:
        return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()

    def _reset(self, ident: int) -> None:
        from torch._dynamo import convert_frame
        from torch._dynamo.utils import guard_failures

        code, state = self.codes.pop(ident)
        with convert_frame.compile_lock:
            torch._dynamo.reset_code(code)
            # reset_code starts a fresh frame ID, but 2.14 retains the old
            # cumulative counter. Retire only frame IDs recorded by our backend.
            for frame_id in state.frame_ids:
                convert_frame.FRAME_COMPILE_COUNTER.pop(frame_id, None)
            guard_failures.pop(code, None)
            for tracker in (convert_frame.input_codes, convert_frame.output_codes):
                tracker.seen[:] = [ref for ref in tracker.seen if ref() is not None]
        self.resets += 1
        self.pending_retirements += 1

    def before_call(self, code: CodeType) -> None:
        ident = id(code)
        state = self.codes.get(ident)
        saturated = state is not None and state[1].variants >= self.per_code
        total = sum(value.variants for _, value in self.codes.values())
        if saturated or total >= self.capacity:
            # A capture's second invocation must keep its warmed specialization.
            # The next host invocation performs the pending retirement instead.
            if self._capturing():
                return
            if saturated:
                self._reset(ident)
            while (
                sum(value.variants for _, value in self.codes.values()) >= self.capacity
            ):
                self._reset(next(iter(self.codes)))
        if ident not in self.codes:
            if len(self.codes) >= self.code_capacity:
                if self._capturing():
                    raise RuntimeError(
                        "a new forward unit must warm before CUDA capture"
                    )
                self._reset(next(iter(self.codes)))
            self.codes[ident] = code, CodeState()
        self.codes.move_to_end(ident)

    def bind_backend(self, compiled, backend) -> None:
        """Do not let Dynamo's reset registry retain dead capture closures."""
        from torch._dynamo import eval_frame

        for ident, wrapper in eval_frame.cached_backends.items():
            if getattr(wrapper, "compiler_fn", None) is backend:
                reference = weakref.ref(wrapper)
                self.backends[ident] = reference
                weakref.finalize(compiled, self._release_backend, ident, reference)

    def _release_backend(self, ident: int, reference) -> None:
        from torch._dynamo import eval_frame

        if eval_frame.cached_backends.get(ident) is reference():
            eval_frame.cached_backends.pop(ident, None)
        self.backends.pop(ident, None)

    def compiled(self, code: CodeType, module_start: int) -> None:
        from torch._guards import CompileContext
        from torch._inductor.codecache import PyCodeCache

        state = self.codes[id(code)][1]
        state.variants += 1
        compile_id = CompileContext.current_compile_id()
        if compile_id is not None:
            state.frame_ids.add(compile_id.frame_id)
        for module in PyCodeCache.modules[module_start:]:
            self.modules[id(module)] = module
        self.trim_modules()

    def trim_modules(self) -> None:

        surplus = len(self.modules) - self.capacity
        if surplus <= 0:
            return
        retired = [self.modules.popitem(last=False)[1] for _ in range(surplus)]
        self._release_modules(retired)
        self.module_evictions += len(retired)
        self.pending_retirements += len(retired)

    def after_call(self, *, force=False) -> None:
        """Retire metadata and cyclic compiler owners at a safe host boundary."""
        if self._capturing() or self.metadata.scope_active():
            return
        self.metadata.trim(clear=force)
        retired = (
            self.metadata.fx_evictions
            + self.metadata.fake_evictions
            + self.metadata.profiler_evictions
        )
        self.pending_retirements += retired - self._metadata_retirements
        self._metadata_retirements = retired
        if force or self.pending_retirements >= self.gc_batch:
            # Generated functions own globals which can own the same function.
            # Dynamo's generation-1 collection cannot retire older cycles.
            gc.collect(2)
            self.full_gc_collections += 1
            self.pending_retirements = 0

    @staticmethod
    def _release_modules(retired) -> None:
        from torch._inductor.codecache import PyCodeCache

        retired_ids = {id(module) for module in retired}
        PyCodeCache.modules[:] = [
            module for module in PyCodeCache.modules if id(module) not in retired_ids
        ]
        retained_paths = {module.__file__ for module in PyCodeCache.modules}
        for module in retired:
            path = module.__file__
            if PyCodeCache.modules_no_attr.get(path) is module:
                PyCodeCache.modules_no_attr.pop(path)
            if sys.modules.get(module.__name__) is module:
                sys.modules.pop(module.__name__)
            if path not in retained_paths:
                PyCodeCache.linemaps.pop(path, None)
        PyCodeCache.stack_frames_for_code.cache_clear()

    def snapshot(self) -> dict[str, int]:
        from torch._inductor.codecache import PyCodeCache

        return {
            **self.metadata.snapshot(),
            "code_cache_entries": sum(
                state.variants for _, state in self.codes.values()
            ),
            "code_cache_capacity": self.capacity,
            "code_objects": len(self.codes),
            "code_object_capacity": self.code_capacity,
            "per_code_capacity": self.per_code,
            "module_entries": len(self.modules),
            "module_capacity": self.capacity,
            "total_module_entries": len(PyCodeCache.modules),
            "foreign_module_entries": sum(
                id(module) not in self.modules for module in PyCodeCache.modules
            ),
            "backend_entries": len(self.backends),
            "module_evictions": self.module_evictions,
            "cache_resets": self.resets,
            "full_gc_collections": self.full_gc_collections,
            "gc_retirement_batch": self.gc_batch,
            "pending_retirements": self.pending_retirements,
        }

    def clear(self) -> None:
        for ident in list(self.codes):
            self._reset(ident)
        # Live callable owners are released by worker shutdown first. Remove
        # the compiler's registry roots without deleting persistent disk files.
        self._release_modules(list(self.modules.values()))
        self.modules.clear()
        for ident, reference in list(self.backends.items()):
            self._release_backend(ident, reference)
        self.after_call(force=True)


COMPILER_CACHE = CompilerCache()
