"""The formal timing collector must reject divergent whole-operation outputs."""

import pytest
import torch
from oh_my_vllm.kernels.backend import NAME

from development.kernels.cases import cases
from development.kernels.fixtures import fixture
from development.kernels.output_verification import verify


def test_verification_checks_returns_and_written_state():
    old = torch.tensor([1.0])
    new = torch.tensor([1.0])
    old_state = torch.tensor([[3.0]])
    new_state = torch.tensor([[3.0]])
    witnesses = (("written_state", lambda: old_state, lambda: new_state),)
    result = verify("recurrent", lambda: old, lambda: new, witnesses)
    assert result == {
        "passed": True,
        "compared_tensors": 2,
        "witnessed_writes": ["written_state"],
    }
    new_state[0, 0] = 9
    with pytest.raises(AssertionError, match="written_state"):
        verify("recurrent", lambda: old, lambda: new, witnesses)


def test_verification_rejects_missing_or_mismatched_returns():
    with pytest.raises(AssertionError, match="no output"):
        verify("append", lambda: None, lambda: None)
    copied = torch.tensor([7.0])
    result = verify(
        "append",
        lambda: None,
        lambda: None,
        (("written_cache", lambda: copied, lambda: copied.clone()),),
    )
    assert result["compared_tensors"] == 1
    with pytest.raises(AssertionError, match="dtype mismatch"):
        verify("norm", lambda: torch.ones(1), lambda: torch.ones(1).half())


def test_verification_uses_exact_copy_tolerances():
    value = torch.tensor([1.0])
    changed = torch.tensor([1.001])
    with pytest.raises(AssertionError, match="written_value"):
        verify(
            "prepare_attention",
            lambda: value,
            lambda: value,
            (("written_value", lambda: value, lambda: changed),),
        )
    with pytest.raises(AssertionError, match=r"return\[0\]"):
        verify("add_norm", lambda: (value, value), lambda: (changed, value))
    with pytest.raises(AssertionError, match=r"return\[0\]"):
        verify("add_norm_fp8_linear", lambda: (value, value), lambda: (changed, value))


@pytest.mark.parametrize("column_major", [False, True])
def test_quant_scale_stride_is_part_of_the_contract(column_major):
    row_major = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    column_layout = torch.empty_strided((2, 3), (1, 2))
    column_layout.copy_(row_major)
    expected = column_layout if column_major else row_major
    wrong = row_major if column_major else column_layout
    data = torch.ones(2, 384, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="scale stride mismatch"):
        verify("quant", lambda: (data, expected), lambda: (data, wrong))


def test_recurrent_state_tolerance_is_per_slot():
    old_state = torch.tensor([[[1.0] * 10], [[100.0] * 10]])
    new_state = old_state.clone()
    new_state[0, 0, 0] += 0.5
    witnesses = (("written_state", lambda: old_state, lambda: new_state),)
    with pytest.raises(AssertionError, match=r"written_state\[0\]"):
        verify("recurrent", lambda: old_state, lambda: old_state, witnesses)


def _smallest_formal_case(operation):
    matching = [c for c in cases() if c["configuration"]["operation"] == operation]
    return min(
        matching,
        key=lambda case: case["configuration"].get(
            "tokens", sum(case["configuration"].get("counts", ()))
        ),
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or NAME != "cuda", reason="CUDA required"
)
@pytest.mark.parametrize(
    "operation",
    [
        "prepare_attention",
        "convolution",
        "recurrent",
        "quant",
        "attention",
        "append",
        "dspark_rms_norm",
        "dspark_norm_rope",
        "dspark_append",
        "dspark_attention",
    ],
)
def test_formal_fixture_compares_both_backends(operation):
    config = (
        {"operation": "append", "tokens": 2}
        if operation == "append"
        else _smallest_formal_case(operation)["configuration"]
    )
    reference, candidate, witnesses = fixture(config, observe=True)
    result = verify(operation, reference, candidate, witnesses)
    assert result["passed"]
    assert result["compared_tensors"] >= 1
    if operation in (
        "prepare_attention",
        "convolution",
        "recurrent",
        "append",
        "dspark_append",
    ):
        assert result["witnessed_writes"]
