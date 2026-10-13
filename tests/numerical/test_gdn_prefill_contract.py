"""Independent state, storage, and replay contracts for native GDN prefill."""

import subprocess
import sys

import pytest
import torch
from oh_my_vllm.kernels.cuda_backend import compiled
from oh_my_vllm.kernels.gdn import normalize_qk, prefill

from development.kernels.oracles import prefill as reference
from development.kernels.output_verification import verify

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def inputs(length, dtype=torch.int32):
    torch.manual_seed(length)
    q = torch.randn(length, 17, 128, device="cuda", dtype=torch.bfloat16)[:, :16]
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    v = torch.randn(length, 49, 128, device="cuda", dtype=torch.bfloat16)[:, :48]
    decay = -torch.rand(length, 48, device="cuda") * 0.1
    beta = torch.rand_like(decay)
    state = torch.randn(3, 48, 128, 128, device="cuda") * 0.01
    offsets = torch.tensor([0, 0, length - 1, length], device="cuda", dtype=dtype)
    return q, k, v, decay, beta, state, offsets


@pytest.mark.parametrize("length", [1, 63, 127, 128, 129, 257])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_strided_ragged_empty_sequences_preserve_state_contract(length, dtype):
    args = inputs(length, dtype)
    initial = args[-2].clone()
    expected = reference(*args)
    actual = prefill(*args)
    verify("gdn_prefill", lambda expected=expected: expected, lambda: actual)
    torch.testing.assert_close(args[-2], initial, rtol=0, atol=0)
    torch.testing.assert_close(actual[1][0], initial[0], rtol=0, atol=0)


def test_changed_inputs_and_ragged_offsets_replay_independent_reference():
    args = inputs(257)
    for _ in range(2):
        prefill(*args)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        actual = prefill(*args)
    for boundary in (64, 129):
        args[0].mul_(0.9)
        args[1].add_(0.125)
        args[2].mul_(0.75)
        args[3].mul_(0.8)
        args[4].mul_(0.8)
        args[5].add_(0.01)
        args[6].copy_(torch.tensor([0, 0, boundary, 257], device="cuda"))
        expected = reference(*args)
        graph.replay()
        verify("gdn_prefill", lambda expected=expected: expected, lambda: actual)


@pytest.mark.parametrize("unaligned", [False, True])
def test_chunk_scratch_capacity_alias_and_output_canaries(unaligned):
    q, k, v, decay, beta, state, offsets = inputs(129)
    expected = reference(q, k, v, decay, beta, state, offsets)
    q, k = normalize_qk(q, k)
    source_storages = []
    if unaligned:
        strided = []
        for tensor in (q, k, v):
            token_width = tensor.shape[1] * tensor.shape[2]
            storage = torch.full(
                (tensor.shape[0] * (token_width + 1) + 1,),
                29,
                device="cuda",
                dtype=tensor.dtype,
            )
            view = storage.as_strided(
                tensor.shape, (token_width + 1, tensor.shape[2], 1), storage_offset=1
            )
            view.copy_(tensor)
            source_storages.append((storage, storage.clone()))
            strided.append(view)
        q, k, v = strided
    chunks = 5
    coefficients = torch.empty(chunks, 48, 64, 64, device="cuda", dtype=torch.bfloat16)
    cumulative = torch.empty(chunks, 48, 64, device="cuda")
    transformed_keys = torch.empty(
        chunks, 48, 64, 128, device="cuda", dtype=torch.bfloat16
    )
    transformed_values = torch.empty(chunks, 48, 64, 128, device="cuda")
    query_key_scores = torch.empty(
        chunks, 48, 64, 64, device="cuda", dtype=torch.bfloat16
    )
    carried_keys = torch.empty_like(transformed_keys)
    intermediates = (
        transformed_keys,
        transformed_values,
        query_key_scores,
        carried_keys,
    )
    storage = torch.full((131, 48, 128), 19, device="cuda", dtype=torch.bfloat16)
    final_storage = torch.full(
        (5, 48, 128, 128), 23, device="cuda", dtype=torch.float32
    )
    output, final = storage[1:-1], final_storage[1:-1]
    module = compiled()
    with pytest.raises(Exception, match="Check failed"):
        module.gdn_chunk_prepare(
            k, decay, beta, offsets, coefficients[:2], cumulative[:2]
        )
    module.gdn_chunk_prepare(k, decay, beta, offsets, coefficients, cumulative)
    module.gdn_chunk_project(q, k, v, offsets, coefficients, cumulative, *intermediates)
    with pytest.raises(Exception, match="Check failed"):
        module.gdn_chunk_project(
            q,
            k,
            v,
            offsets,
            coefficients,
            cumulative,
            transformed_keys[:2],
            transformed_values,
            query_key_scores,
            carried_keys,
        )
    with pytest.raises(Exception, match="Check failed"):
        module.gdn_chunk_project(
            q,
            k,
            v,
            offsets,
            coefficients,
            cumulative,
            transformed_keys,
            transformed_values,
            query_key_scores,
            transformed_keys,
        )
    with pytest.raises(Exception, match="Check failed"):
        module.gdn_chunk_scan(
            q,
            k,
            v,
            state,
            offsets,
            coefficients[:2],
            cumulative[:2],
            *intermediates,
            output,
            final,
        )
    with pytest.raises(Exception, match="Check failed"):
        module.gdn_chunk_scan(
            q,
            k,
            v,
            state,
            offsets,
            coefficients,
            cumulative,
            *intermediates,
            output,
            state,
        )
    module.gdn_chunk_scan(
        q, k, v, state, offsets, coefficients, cumulative, *intermediates, output, final
    )
    verify("gdn_prefill", lambda: expected, lambda: (output, final))
    for source_storage, initial in source_storages:
        torch.testing.assert_close(source_storage, initial, rtol=0, atol=0)
    torch.testing.assert_close(
        storage[[0, -1]], torch.full_like(storage[[0, -1]], 19), rtol=0, atol=0
    )
    torch.testing.assert_close(
        final_storage[[0, -1]],
        torch.full_like(final_storage[[0, -1]], 23),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("length", [1, 128])
@pytest.mark.parametrize("bad_offsets", ["origin", "terminal", "decreasing"])
def test_invalid_offsets_fail_in_disposable_cuda_context(length, bad_offsets):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.gpu_slot_guard_case",
            "gdn_prefill",
            str(length),
            bad_offsets,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
