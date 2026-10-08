"""The IR must lower opaque semantic nodes before Inductor sees the graph."""

import logging

import pytest
import torch
from oh_my_vllm.ir import (
    Operation,
    TensorSpec,
    compile_forward,
    compiled_graph_counts,
    configure_priorities,
    selected_implementations,
)


@torch.library.custom_op("oh_my_vllm_ir::test_scale", mutates_args=())
def _ir_scale(x: torch.Tensor) -> torch.Tensor:
    return _scale.select(x).op(x)


@_ir_scale.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_native::test_scale", mutates_args=())
def _scale_native(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_scale_native.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_impl::test_scale_two", mutates_args=())
def _scale_two(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_scale_two.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_impl::test_scale_three", mutates_args=())
def _scale_three(x: torch.Tensor) -> torch.Tensor:
    return x * 3


@_scale_three.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_impl::test_scale_mutating", mutates_args=("x",))
def _scale_mutating(x: torch.Tensor) -> torch.Tensor:
    x.mul_(2)
    return x.clone()


@torch.library.custom_op("oh_my_vllm_ir::test_contract", mutates_args=())
def _ir_contract(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_ir_contract.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_native::test_contract", mutates_args=())
def _contract_native(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_contract_native.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


_scale = Operation("test_scale", _ir_scale, _scale_native, default_priority=("two",))
_scale.register_impl(
    "two",
    _scale_two,
    supports=lambda x: isinstance(x, TensorSpec) and x.shape[-1] == 4,
)
_scale.register_impl("three", _scale_three)
_contract = Operation(
    "test_contract", _ir_contract, _contract_native, default_priority=("native",)
)


@torch.library.custom_op("oh_my_vllm_ir::test_fake", mutates_args=())
def _ir_fake(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_ir_fake.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_native::test_fake", mutates_args=())
def _native_fake(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_native_fake.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_impl::test_fake_bad", mutates_args=())
def _bad_fake(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_bad_fake.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], x.shape[1] + 1), device=x.device, dtype=x.dtype)


_fake = Operation("test_fake", _ir_fake, _native_fake, default_priority=("bad",))
_fake.register_impl("bad", _bad_fake)


@torch.library.custom_op("oh_my_vllm_ir::test_self", mutates_args=())
def _ir_self(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_ir_self.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("oh_my_vllm_native::test_self", mutates_args=())
def _native_self(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_native_self.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


_self = Operation("test_self", _ir_self, _native_self, default_priority=("native",))


@torch.library.custom_op("oh_my_vllm_ir::test_unregistered", mutates_args=())
def _unregistered(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_unregistered.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def test_ir_compiled_and_eager_select_same_provider():
    x = torch.ones(2, 4)
    expected = _scale(x)
    before = compiled_graph_counts().get("test_scale_forward", 0)
    compiled = compile_forward(
        lambda value: _scale(value) + 1, unit="test_scale_forward"
    )
    assert torch.equal(expected, torch.full_like(x, 2))
    assert torch.equal(compiled(x), torch.full_like(x, 3))
    assert compiled_graph_counts()["test_scale_forward"] == before + 1
    records = [
        record
        for record in selected_implementations()
        if record.operation == "test_scale"
    ]
    assert {record.phase for record in records} == {"eager", "compile"}
    assert {record.provider for record in records} == {"two"}


def test_compile_log_is_emitted_once_per_actual_lowering(caplog, monkeypatch):
    from oh_my_vllm.ir import core

    # Keep this test focused on backend entry, independent of Inductor latency.
    monkeypatch.setattr(core, "lower_to_inductor", lambda graph, _: graph.forward)
    unit = compile_forward(lambda x: x.sin() + 7, unit="test_compile_log")
    with caplog.at_level(logging.INFO, logger=core.__name__):
        x = torch.ones(2, 4)
        unit(x)
        unit(x)
        first = [r for r in caplog.records if "Compilation started:" in r.getMessage()]
        assert [r.getMessage() for r in first] == [
            "Compilation started: IR unit test_compile_log"
        ]
        unit(torch.ones(3, 5))
        count = len(
            [r for r in caplog.records if "Compilation started:" in r.getMessage()]
        )
        assert count == 2
        unit(torch.ones(3, 5))
        assert (
            len([r for r in caplog.records if "Compilation started:" in r.getMessage()])
            == count
        )


def test_ir_rejects_unsupported_shape_without_fallback():
    with pytest.raises(RuntimeError, match="no supported IR implementation"):
        _scale(torch.ones(2, 3))


@pytest.mark.parametrize("transposed", [False, True])
def test_ir_reuses_symbolic_shapes_and_strides(transposed):
    label = f"test_dynamic_scale_{transposed}"
    compiled = compile_forward(lambda x: _scale(x) + 1, unit=label)

    def input_for(rows):
        return torch.ones(4, rows).T if transposed else torch.ones(rows, 4)

    # Allow Dynamo's initial static graph and automatic dynamic generalization.
    for rows in (2, 3):
        x = input_for(rows)
        torch.testing.assert_close(compiled(x), x * 2 + 1)
    warmed = compiled_graph_counts()[label]
    for rows in (4, 5, 6, 7):
        x = input_for(rows)
        torch.testing.assert_close(compiled(x), x * 2 + 1)
    assert compiled_graph_counts()[label] == warmed

    # A dimension consulted by the provider must retain its guard: the cached
    # graph must not run the width-four provider on an unsupported width.
    with pytest.raises(Exception, match="no supported IR implementation"):
        compiled(torch.ones(7, 5))


def test_ir_capability_receives_only_static_metadata():
    assert _scale.select(torch.ones(2, 4)).name == "two"
    assert _scale.select(-torch.ones(2, 4)).name == "two"


def test_ir_symbolic_provider_predicate_keeps_range_guard():
    @torch.library.custom_op("oh_my_vllm_native::test_row_range", mutates_args=())
    def native(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @torch.library.custom_op("oh_my_vllm_ir::test_row_range", mutates_args=())
    def semantic(x: torch.Tensor) -> torch.Tensor:
        return operation.select(x).op(x)

    for op in (native, semantic):
        op.register_fake(lambda x: torch.empty_like(x))
    operation = Operation("test_row_range", semantic, native, default_priority=("two",))
    operation.register_impl("two", _scale_two, supports=lambda x: x.shape[0] < 5)
    label = "test_symbolic_predicate"
    compiled = compile_forward(lambda x: operation(x), unit=label)
    for rows in (2, 3):
        x = torch.ones(rows, 4)
        torch.testing.assert_close(compiled(x), x * 2)
    warmed = compiled_graph_counts()[label]
    torch.testing.assert_close(compiled(torch.ones(4, 4)), torch.full((4, 4), 2.0))
    assert compiled_graph_counts()[label] == warmed
    with pytest.raises(Exception, match="no supported IR implementation"):
        compiled(torch.ones(5, 4))


def test_ir_configuration_freezes_after_selection():
    _scale(torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="already in use"):
        configure_priorities({"test_scale": ["three"]})


def test_ir_rejects_unknown_configuration_atomically():
    with pytest.raises(ValueError, match="unknown IR operations"):
        configure_priorities({"unknown": ["provider"]})


def test_ir_custom_op_fake_contract():
    result = torch.library.opcheck(
        _ir_scale,
        (torch.ones(2, 4),),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
    assert set(result.values()) == {"SUCCESS"}


def test_ir_rejects_provider_with_different_mutation_contract():
    with pytest.raises(ValueError, match="different argument, return, or mutation"):
        _contract.register_impl("mutating", _scale_mutating)


def test_ir_rejects_semantic_op_as_provider():
    with pytest.raises(ValueError, match="cannot be its own provider"):
        _self.register_impl("recursive", _ir_self)


def test_ir_rejects_mismatched_semantic_target():
    with pytest.raises(ValueError, match=r"mismatched torch\.library op"):
        Operation(
            "test_wrong", _ir_contract, _contract_native, default_priority=("native",)
        )


def test_ir_rejects_provider_with_wrong_fake_output():
    compiled = compile_forward(lambda x: _fake(x))
    with pytest.raises(Exception, match="different fake output"):
        compiled(torch.ones(2, 4))
    assert not any(
        record.operation == "test_fake" and record.phase == "compile"
        for record in selected_implementations()
    )


def test_ir_rejects_unregistered_semantic_node():
    compiled = compile_forward(lambda x: _unregistered(x))
    with pytest.raises(Exception, match="unlowered IR operation"):
        compiled(torch.ones(2, 4))


def test_pinned_compiler_keeps_enough_static_graph_specializations():
    assert torch.__version__.split("+", 1)[0] == "2.14.0"
    compile_forward(lambda x: x + 1)
    assert torch._dynamo.config.recompile_limit >= 4096
    assert torch._dynamo.config.accumulated_recompile_limit >= 4096
    assert not torch._inductor.config.triton.cudagraphs


@pytest.mark.parametrize("override", ["silu_mul", "fp8_linear"])
def test_fusion_respects_explicit_provider_priority(monkeypatch, override):
    from oh_my_vllm.ir import fp8, pointwise
    from oh_my_vllm.ir.core import _rewrite_silu_fp8_linear

    graph = torch.fx.Graph()
    packed = graph.placeholder("packed")
    weight = graph.placeholder("weight")
    scale = graph.placeholder("scale")
    activated = graph.call_function(pointwise._SILU.target, (packed,))
    output = graph.call_function(fp8._LINEAR.target, (activated, weight, scale, False))
    graph.output(output)
    module = torch.fx.GraphModule({}, graph)
    operation = pointwise._SILU if override == "silu_mul" else fp8._LINEAR
    monkeypatch.setattr(operation, "priority", ("native",))
    assert _rewrite_silu_fp8_linear(module) == 0
    assert activated in tuple(module.graph.nodes)


@pytest.mark.parametrize(
    ("dtype", "intervening_write"),
    [(torch.float32, False), (torch.bfloat16, True)],
)
def test_fusion_requires_bf16_and_no_intervening_write(dtype, intervening_write):
    from oh_my_vllm.ir import fp8, pointwise
    from oh_my_vllm.ir.core import _rewrite_silu_fp8_linear

    graph = torch.fx.Graph()
    packed = graph.placeholder("packed")
    weight = graph.placeholder("weight")
    scale = graph.placeholder("scale")
    packed.meta["example_value"] = torch.empty(1, 10240, dtype=dtype)
    weight.meta["example_value"] = torch.empty(256, 5120, dtype=torch.float8_e4m3fn)
    scale.meta["example_value"] = torch.empty(2, 40)
    activated = graph.call_function(pointwise._SILU.target, (packed,))
    if intervening_write:
        graph.call_method("add_", (packed, 1))
    output = graph.call_function(fp8._LINEAR.target, (activated, weight, scale, False))
    graph.output(output)
    module = torch.fx.GraphModule({}, graph)
    assert _rewrite_silu_fp8_linear(module) == 0
