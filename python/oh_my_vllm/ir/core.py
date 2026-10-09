"""Register semantic operators and lower their FX nodes to selected providers.

An operation has a native PyTorch reference, an opaque torch.library operation,
and independently registered implementations. The reference is available only
when explicitly selected as a provider. Both eager and compiled paths use the
same static capability check and priority list. Configuration freezes at first
selection so a compiled graph cannot silently retain an obsolete provider.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg

LOG = logging.getLogger(__name__)

# The manual target/draft/proposal graph cache deliberately retains many static
# token extents. Dynamo shares recompile counters by Python code object even
# when each graph has a distinct closure. Raise both its resident and lifetime
# limits; do not let Dynamo silently run eager when either limit is reached.
# Host specialization and manual graph closures can still exhaust this bound.
_RECOMPILE_LIMIT = 4096


@dataclass(frozen=True)
class Implementation:
    name: str
    op: Callable[..., Any]
    target: Any
    supports: Callable[..., bool]


@dataclass(frozen=True)
class Selection:
    operation: str
    provider: str
    phase: str
    static_key: str


@dataclass(frozen=True)
class Rewrite:
    pattern: str
    count: int


@dataclass(frozen=True)
class TensorSpec:
    """Metadata only; symbolic dimensions stay dynamic until a predicate uses them."""

    shape: tuple[int | torch.SymInt, ...]
    stride: tuple[int | torch.SymInt, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def ndim(self) -> int:
        return len(self.shape)


_OPERATIONS: dict[str, Operation] = {}
_TARGETS: dict[Any, Operation] = {}
_SELECTIONS: dict[tuple[str, str, str], Selection] = {}
_REWRITES: list[Rewrite] = []
_COMPILED_GRAPHS: Counter[str] = Counter()


class Operation:
    """One logical operation with explicit eager and compile-time dispatch."""

    def __init__(
        self,
        name: str,
        ir_op: Callable[..., Any],
        reference: Callable[..., Any],
        *,
        default_priority: Sequence[str],
    ) -> None:
        if name in _OPERATIONS:
            raise ValueError(f"IR operation already registered: {name}")
        if not default_priority:
            raise ValueError(f"IR operation {name} needs a production provider")
        if getattr(ir_op, "_qualname", None) != f"oh_my_vllm_ir::{name}":
            raise ValueError(f"IR operation {name} has a mismatched torch.library op")
        self.name = name
        self.ir_op = ir_op
        self.reference = reference
        self.default_priority = tuple(default_priority)
        self.priority = self.default_priority
        self.implementations: dict[str, Implementation] = {}
        self._frozen = False
        self.target = getattr(torch.ops.oh_my_vllm_ir, name).default
        native = getattr(reference, "_qualname", None)
        if native != f"oh_my_vllm_native::{name}":
            raise ValueError(f"IR operation {name} needs a native reference op")
        self.register_native(reference)
        _OPERATIONS[name] = self
        _TARGETS[self.target] = self

    def register_impl(
        self,
        name: str,
        op: Callable[..., Any],
        *,
        supports: Callable[..., bool] | None = None,
        available: bool = True,
    ) -> None:
        """Register a torch.library implementation before first selection."""
        if self._frozen:
            raise RuntimeError(f"IR operation {self.name} is already in use")
        if name in self.implementations:
            raise ValueError(f"duplicate IR provider {self.name}/{name}")
        if not available:
            return
        qualname = getattr(op, "_qualname", None)
        if qualname is None or "::" not in qualname:
            raise TypeError(f"IR provider {self.name}/{name} needs a torch.library op")
        namespace, op_name = qualname.split("::", 1)
        if namespace == "oh_my_vllm_ir":
            raise ValueError("an IR semantic op cannot be its own provider")
        target = getattr(getattr(torch.ops, namespace), op_name).default
        semantic_schema = str(self.target._schema).partition("(")[2]
        provider_schema = str(target._schema).partition("(")[2]
        if provider_schema != semantic_schema:
            raise ValueError(
                f"IR provider {self.name}/{name} has a different argument, "
                "return, or mutation schema"
            )
        self.implementations[name] = Implementation(
            name, op, target, supports or (lambda *args, **kwargs: True)
        )

    def register_native(self, op: Callable[..., Any]) -> None:
        """Expose the reference as an explicit, schema-checked debug provider."""
        self.register_impl("native", op)

    def set_priority(self, providers: Sequence[str]) -> None:
        if self._frozen:
            raise RuntimeError(f"IR operation {self.name} is already in use")
        if not providers or len(set(providers)) != len(providers):
            raise ValueError(f"IR priority for {self.name} must be unique and nonempty")
        unknown = set(providers) - self.implementations.keys()
        if unknown:
            raise ValueError(f"unknown IR provider for {self.name}: {sorted(unknown)}")
        self.priority = tuple(providers)

    def select(self, *args: Any, **kwargs: Any) -> Implementation:
        return self._select(args, kwargs, "eager")

    def _select(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        phase: str,
        *,
        record: bool = True,
    ) -> Implementation:
        self._frozen = True
        static_args = _to_static(args)
        static_kwargs = _to_static(kwargs)
        for name in self.priority:
            implementation = self.implementations.get(name)
            if implementation is None:
                continue
            if implementation.supports(*static_args, **static_kwargs):
                key = _static_key(static_args, static_kwargs)
                if record:
                    _record_selection(self.name, name, phase, key)
                LOG.debug("IR selected %s/%s for %s %s", self.name, name, phase, key)
                return implementation
        raise RuntimeError(
            f"no supported IR implementation for {self.name}; priority={self.priority}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return self.ir_op(*args, **kwargs)
        return self.select(*args, **kwargs).op(*args, **kwargs)


def operations() -> Mapping[str, Operation]:
    return dict(_OPERATIONS)


def configure_priorities(priorities: Mapping[str, Sequence[str]]) -> None:
    """Apply all worker startup overrides before any operation is selected."""
    unknown = priorities.keys() - _OPERATIONS.keys()
    if unknown:
        raise ValueError(f"unknown IR operations: {sorted(unknown)}")
    for name, providers in priorities.items():
        operation = _OPERATIONS[name]
        if operation._frozen:
            raise RuntimeError(f"IR operation {name} is already in use")
        if not providers or len(set(providers)) != len(providers):
            raise ValueError(f"IR priority for {name} must be unique and nonempty")
        unknown_providers = set(providers) - operation.implementations.keys()
        if unknown_providers:
            raise ValueError(
                f"unknown IR providers for {name}: {sorted(unknown_providers)}"
            )
    for name, providers in priorities.items():
        _OPERATIONS[name].set_priority(providers)


def selected_implementations() -> tuple[Selection, ...]:
    """Successful compiled decisions and eager invocations, keyed by metadata."""
    return tuple(_SELECTIONS.values())


def applied_rewrites() -> tuple[Rewrite, ...]:
    """Graph rewrites committed by successful Inductor compilations."""
    return tuple(_REWRITES)


def compiled_graph_counts() -> Mapping[str, int]:
    """Successful Inductor compilations by declared forward unit in this process."""
    return dict(_COMPILED_GRAPHS)


def _record_selection(operation: str, provider: str, phase: str, key: str) -> None:
    _SELECTIONS[(phase, operation, key)] = Selection(operation, provider, phase, key)


def _to_static(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return TensorSpec(
            # Converting SymInt to int installs equality guards even when the
            # provider ignores that dimension. Predicate comparisons may still
            # install the guards needed to make provider selection sound.
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
    if isinstance(value, tuple):
        return tuple(_to_static(item) for item in value)
    if isinstance(value, list):
        return [_to_static(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_static(item) for key, item in value.items()}
    return value


def _static_key(args: Any, kwargs: Any) -> str:
    def convert(value: Any) -> Any:
        if isinstance(value, TensorSpec):
            return (
                "tensor",
                str(value.dtype),
                str(value.device),
                value.shape,
                value.stride,
            )
        if isinstance(value, (tuple, list)):
            return tuple(convert(item) for item in value)
        if isinstance(value, dict):
            return tuple((key, convert(item)) for key, item in sorted(value.items()))
        return repr(value)

    return repr((convert(args), convert(kwargs)))


def _example(node: torch.fx.Node) -> Any:
    if "example_value" not in node.meta:
        # Dynamo omits example_value for a void custom op. Its registered
        # mutation schema still retains the effectful node and validates that
        # each selected provider also returns None; no alias/clone is needed.
        schema = getattr(node.target, "_schema", None)
        if node.op == "call_function" and schema is not None and not schema.returns:
            return None
        raise RuntimeError(f"IR compiler lacks metadata for FX node {node.name}")
    return node.meta["example_value"]


def _output_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            str(value.dtype),
            str(value.device),
            tuple(map(str, value.shape)),
            tuple(map(str, value.stride())),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_output_metadata(item) for item in value)
    return value


def _require_fake(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not isinstance(value, FakeTensor):
            raise RuntimeError("IR lowering requires FakeTensor metadata")
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_fake(item)
    elif isinstance(value, dict):
        for item in value.values():
            _require_fake(item)


def _rewrite_silu_fp8_linear(graph_module: torch.fx.GraphModule) -> int:
    """Fuse a single-use SiLU multiply into FP8 linear's equivalent gate path."""
    count = 0
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function":
            continue
        operation = _TARGETS.get(node.target)
        if operation is None or operation.name != "fp8_linear":
            continue
        if operation.priority != operation.default_priority:
            continue
        if len(node.args) != 4 or node.args[3] is not False:
            continue
        activation = node.args[0]
        if not isinstance(activation, torch.fx.Node) or len(activation.users) != 1:
            continue
        # Intervening FX nodes may mutate an alias of packed. Adjacency is a
        # conservative proof that the fused kernel reads the same input value.
        if activation.next is not node:
            continue
        source = _TARGETS.get(activation.target)
        if source is None or source.name != "silu_mul":
            continue
        if source.priority != source.default_priority:
            continue
        if len(activation.args) != 1:
            continue
        packed = activation.args[0]
        if not isinstance(packed, torch.fx.Node):
            continue
        example = packed.meta.get("example_value")
        if example is None or example.dtype != torch.bfloat16:
            continue
        source_impl = source._select((example,), {}, "compile", record=False)
        fused_args = (
            example,
            _example(node.args[1]),
            _example(node.args[2]),
            True,
        )
        fused_impl = operation._select(fused_args, {}, "compile", record=False)
        if source_impl.name != fused_impl.name or source_impl.name == "native":
            continue
        node.args = (activation.args[0], *node.args[1:3], True)
        graph_module.graph.erase_node(activation)
        count += 1
    return count


def lower_to_inductor(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    unit: str = "compiled",
) -> Callable[..., Any]:
    """Select providers from FX metadata, then compile the lowered graph."""
    from oh_my_vllm.performance.execution import observe_graph

    observe_graph(unit, graph_module.graph)
    fused_silu = _rewrite_silu_fp8_linear(graph_module)
    selections: list[tuple[str, str, str]] = []
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        operation = _TARGETS.get(node.target)
        if operation is None:
            continue
        args = map_arg(node.args, _example)
        kwargs = map_arg(node.kwargs, _example)
        _require_fake(args)
        _require_fake(kwargs)
        selected = operation._select(args, kwargs, "compile", record=False)
        expected = _output_metadata(_example(node))
        fake_output = selected.op(*args, **kwargs)
        _require_fake(fake_output)
        actual = _output_metadata(fake_output)
        if actual != expected:
            raise RuntimeError(
                f"IR provider {operation.name}/{selected.name} "
                "has different fake output"
            )
        node.target = selected.target
        selections.append(
            (
                operation.name,
                selected.name,
                _static_key(_to_static(args), _to_static(kwargs)),
            )
        )
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and str(node.target).startswith("oh_my_vllm_ir."):
            raise RuntimeError(f"unlowered IR operation in FX graph: {node.target}")
    graph_module.graph.lint()
    graph_module.recompile()
    compiled = torch._dynamo.lookup_backend("inductor")(graph_module, example_inputs)
    for operation, provider, key in selections:
        _record_selection(operation, provider, "compile", key)
    if fused_silu:
        _REWRITES.append(Rewrite("silu_mul+fp8_linear", fused_silu))
    return compiled


def compile_forward(
    function: Callable[..., Any], *, unit: str | None = None
) -> Callable[..., Any]:
    """Compile one declared GPU forward unit; graph breaks are errors."""
    from .execution import execution_policy

    execution_policy()
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, _RECOMPILE_LIMIT
    )
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, _RECOMPILE_LIMIT
    )
    torch._inductor.config.triton.cudagraphs = False
    label = unit or getattr(function, "__qualname__", type(function).__name__)

    def lower(graph_module: torch.fx.GraphModule, inputs: Sequence[Any]):
        LOG.info("Compilation started: IR unit %s", label)
        compiled = lower_to_inductor(graph_module, inputs, unit=label)
        _COMPILED_GRAPHS[label] += 1
        count = _COMPILED_GRAPHS[label]
        if count % 256 == 0:
            LOG.warning(
                "IR unit %s compiled %d graph variants; inspect shape churn and "
                "manual graph cache/allocator pressure",
                label,
                count,
            )
        return compiled

    return torch.compile(function, backend=lower, fullgraph=True)
