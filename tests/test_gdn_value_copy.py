"""Packed GDN values retain their bits through copies and graph replay."""

import pytest
import torch
from oh_my_vllm.kernels.gdn import _contiguous_values


def test_contiguous_values_reuse_storage_and_cpu_strides_copy():
    dense = torch.arange(24).reshape(2, 3, 4).bfloat16()
    assert _contiguous_values(dense) is dense
    strided = torch.arange(40).bfloat16().reshape(2, 5, 4)[:, 1:4]
    assert not strided.is_contiguous()
    actual = _contiguous_values(strided)
    assert actual.is_contiguous()
    assert actual.data_ptr() != strided.data_ptr()
    torch.testing.assert_close(actual, strided, rtol=0, atol=0)


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.gpu
@GPU
@pytest.mark.parametrize("rows", [1, 7, 624, 32144, 32768])
@pytest.mark.parametrize("stride", [10240, 16384])
def test_packed_values_copy_all_bits_and_preserve_input(rows, stride):
    torch.manual_seed(784)
    packed = torch.randn(rows, stride, device="cuda", dtype=torch.bfloat16)
    packed[0, 4096:4100] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0], device="cuda"
    )
    values = packed[:, 4096:10240].view(rows, 48, 128)
    before = packed.clone()
    actual = _contiguous_values(values)
    assert actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), values.contiguous().view(torch.int16))
    assert torch.equal(packed.view(torch.int16), before.view(torch.int16))


@pytest.mark.gpu
@GPU
def test_value_copy_joins_side_stream_and_replays_changed_inputs():
    packed = torch.zeros(7, 10240, device="cuda", dtype=torch.bfloat16)
    values = packed[:, 4096:].view(7, 48, 128)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _contiguous_values(values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _contiguous_values(values)
    torch.cuda.current_stream().wait_stream(side)
    for value in (0.125, -2.0, 7.0):
        packed.fill_(value)
        with torch.cuda.stream(side):
            side.wait_stream(torch.cuda.default_stream())
            graph.replay()
        torch.cuda.current_stream().wait_stream(side)
        assert torch.equal(output, values)


@pytest.mark.gpu
@GPU
@pytest.mark.parametrize("scenario", ["dtype", "stride", "alignment", "alias", "shape"])
def test_ffi_copy_rejects_invalid_storage(scenario):
    from oh_my_vllm.kernels.cuda_backend import compiled

    source = torch.zeros(7, 10240, device="cuda", dtype=torch.bfloat16)[:, 4096:]
    destination = torch.empty(7, 6144, device="cuda", dtype=torch.bfloat16)
    if scenario == "dtype":
        source = source.float()
        message = "BF16 tensors"
    elif scenario == "stride":
        source = source[:, ::2]
        destination = destination[:, ::2]
        message = "dense BF16 rows"
    elif scenario == "alignment":
        source = torch.zeros(7, 10248, device="cuda", dtype=torch.bfloat16)[:, 1:6145]
        message = "aligned eight-value"
    elif scenario == "alias":
        source = destination
        message = "disjoint"
    else:
        destination = destination[:-1]
        message = "dense BF16 rows"
    with pytest.raises(Exception, match=message):
        compiled().contiguous_bf16_rows(source, destination)


@pytest.mark.gpu
@GPU
def test_prefill_output_and_state_equal_the_original_strided_copy(monkeypatch):
    from oh_my_vllm.kernels import gdn

    torch.manual_seed(784)
    packed = torch.randn(132, 10240, device="cuda", dtype=torch.bfloat16)
    q = packed[:, :2048].view(132, 16, 128)
    k = packed[:, 2048:4096].view(132, 16, 128)
    v = packed[:, 4096:].view(132, 48, 128)
    args = (
        q,
        k,
        v,
        -torch.rand(132, 48, device="cuda"),
        torch.rand(132, 48, device="cuda"),
        torch.randn(2, 48, 128, 128, device="cuda") * 0.01,
        torch.tensor([0, 129, 132], device="cuda", dtype=torch.int32),
    )
    output, final = gdn.prefill(*args)
    monkeypatch.setattr(gdn, "_contiguous_values", lambda values: values.contiguous())
    reference, reference_final = gdn.prefill(*args)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    torch.testing.assert_close(final, reference_final, rtol=0, atol=0)
