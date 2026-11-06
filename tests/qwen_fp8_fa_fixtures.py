from types import SimpleNamespace

from oh_my_vllm.operators.fa_fp8 import api


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


class Device:
    def __init__(self):
        self.allocations = []

    def empty(self, shape, dtype):
        tensor = Tensor(shape, dtype)
        self.allocations.append(tensor)
        return tensor


def bindings(plan):
    weights = api.Weights(*(object() for _ in range(7)))
    batch = api.BatchView(object(), object(), object())
    state = api.CacheView(
        Tensor((5, 2, 128, 4, 256), "uint8"),
        Tensor((5, 2, 128, 4), "float32"),
        Tensor((plan.rows,), "int32"),
        Tensor((plan.batch, plan.table_width), "int32"),
        Tensor((plan.rows,), "bool"),
    )
    return weights, api.Inputs(object(), object()), batch, state


def native_module():
    calls = []
    return SimpleNamespace(fa_fp8_run=lambda *args: calls.append(args)), calls
