"""Fullgraph provider lowering preserves explicit void cache mutations."""

import torch
from oh_my_vllm.ir import Operation, compile_forward


@torch.library.custom_op("oh_my_vllm_ir::test_void_write", mutates_args=("cache",))
def _ir_write(cache: torch.Tensor, value: torch.Tensor) -> None:
    return _WRITE.select(cache, value).op(cache, value)


@torch.library.custom_op("oh_my_vllm_native::test_void_write", mutates_args=("cache",))
def _native_write(cache: torch.Tensor, value: torch.Tensor) -> None:
    cache.copy_(value)


def _fake_write(cache, value):
    torch._check(cache.shape == value.shape)


_ir_write.register_fake(_fake_write)
_native_write.register_fake(_fake_write)
_WRITE = Operation(
    "test_void_write", _ir_write, _native_write, default_priority=("native",)
)


def test_compiled_void_operation_keeps_mutation_and_return_shape():
    def run(cache, value):
        _WRITE(cache, value)
        return cache.square()

    unit = compile_forward(run, unit="test_void_mutation")
    cache = torch.zeros(4)
    value = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(unit(cache, value), value.square(), atol=0, rtol=0)
    torch.testing.assert_close(cache, value, atol=0, rtol=0)
    value.add_(1)
    torch.testing.assert_close(unit(cache, value), value.square(), atol=0, rtol=0)
    torch.testing.assert_close(cache, value, atol=0, rtol=0)
