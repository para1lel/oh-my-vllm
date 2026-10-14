"""All-precise scans keep independent state, boundary, and live replay contracts."""

from itertools import pairwise

import pytest
import torch
from oh_my_vllm.kernels.cuda_backend import compiled
from oh_my_vllm.kernels.gdn import normalize_qk

from development.kernels.oracles import prefill as reference
from development.kernels.output_verification import verify

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

CASES = {
    "cancel_4098_0999": (4098, 1, 1, 0.999, False, False),
    "cancel_4098_1p0": (4098, 1, 1, 1.0, False, False),
    "strided_783": (783, 1, 2, 0.9, True, False),
    "strided_785": (785, 1, 2, 0.9, True, False),
    "heads_784_replay": (784, 16, 48, 0.9, True, True),
}


@pytest.mark.parametrize("name", CASES)
def test_all_precise_chunks_preserve_independent_state_and_live_replay(name):
    length, hk, hv, beta_value, padded, replay = CASES[name]
    torch.manual_seed(428)
    offsets = [0, 0, 64, length] if replay else [0, length]
    sequences = len(offsets) - 1
    chunks = (length + 63) // 64 + sequences - 1
    guards = []

    def rows(heads, sentinel, pad=False):
        stride = heads * 128 + (8 if pad else 0)
        storage = torch.full(
            (length * stride + 2,), sentinel, device="cuda", dtype=torch.bfloat16
        )
        tensor = (
            storage[1:-1]
            .view(length, stride)[:, : heads * 128]
            .view(length, heads, 128)
        )
        guards.append((storage, tensor, sentinel))
        return tensor

    queries, keys, values = (
        rows(hk, 43, padded),
        rows(hk, 45, padded),
        rows(hv, 37, padded),
    )
    output = rows(hv, 41)
    initial = torch.empty(sequences, hv, 128, 128, device="cuda")
    final_storage = torch.full(
        (initial.numel() + 8,), 39, device="cuda", dtype=torch.float32
    )
    final = final_storage[4:-4].view_as(initial)
    starts = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    decay = torch.empty(length, hv, device="cuda")
    beta = torch.empty_like(decay)
    coefficients = torch.empty(chunks, hv, 64, 64, device="cuda", dtype=torch.bfloat16)
    cumulative = torch.empty(chunks, hv, 64, device="cuda")
    conditioning = torch.empty(chunks, hv, device="cuda", dtype=torch.int32)
    transformed_keys = torch.empty(
        chunks, hv, 64, 128, device="cuda", dtype=torch.bfloat16
    )
    transformed_values = torch.empty_like(transformed_keys, dtype=torch.float32)
    query_key = torch.empty_like(coefficients)
    carried_keys = torch.empty_like(transformed_keys)
    module = compiled()

    def operation():
        module.gdn_chunk_prepare(
            keys, decay, beta, starts, coefficients, cumulative, conditioning
        )
        module.gdn_chunk_project(
            queries,
            keys,
            values,
            starts,
            coefficients,
            cumulative,
            transformed_keys,
            transformed_values,
            query_key,
            carried_keys,
            conditioning,
        )
        module.gdn_chunk_scan(
            queries,
            keys,
            values,
            initial,
            starts,
            coefficients,
            cumulative,
            transformed_keys,
            transformed_values,
            query_key,
            carried_keys,
            output,
            final,
            conditioning,
            decay,
            beta,
        )

    def change(index):
        raw_q = torch.randn(length, hk, 128, device="cuda", dtype=torch.bfloat16)
        base = torch.randn(1, hk, 128, device="cuda", dtype=torch.bfloat16)
        raw_k = base + torch.randn_like(raw_q) * (
            0.001 if name.startswith("cancel") else 0.01
        )
        nq, nk = normalize_qk(raw_q, raw_k)
        queries.copy_(nq)
        keys.copy_(nk)
        if name.startswith("cancel"):
            values.zero_()
            decay.zero_()
            projection = torch.randn(sequences, hv, 128, device="cuda")[..., None]
            initial.copy_(projection * nk[0].float()[None, :, None, :])
        else:
            values.normal_()
            decay.uniform_(-0.003, -0.0001)
            initial.normal_(std=0.01 + 0.003 * index)
        beta.fill_(beta_value - (0.02 * index if replay else 0))
        changed_offsets = (
            [0, 0, 128 if index % 2 else 64, length] if replay else offsets
        )
        starts.copy_(torch.tensor(changed_offsets, device="cuda", dtype=starts.dtype))
        return raw_q, raw_k, changed_offsets

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        raw_q, raw_k, active_offsets = change(0)
        operation()
        if replay:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                operation()
        for index in range(3 if replay else 1):
            if index:
                raw_q, raw_k, active_offsets = change(index)
            snapshots = [
                tensor.clone()
                for tensor in (queries, keys, values, initial, decay, beta, starts)
            ]
            expected = reference(raw_q, raw_k, values, decay, beta, initial, starts)
            if graph is None:
                operation()
            else:
                graph.replay()
            actual_chunks = sum(
                (right - left + 63) // 64 for left, right in pairwise(active_offsets)
            )
            routes = conditioning[:actual_chunks].clone()
            assert bool((routes == 1).all()), (name, index, routes.cpu().tolist())
            verify(
                "gdn_prefill",
                lambda expected=expected: expected,
                lambda: (output, final),
            )
            for before, tensor in zip(
                snapshots,
                (queries, keys, values, initial, decay, beta, starts),
                strict=True,
            ):
                torch.testing.assert_close(before, tensor, rtol=0, atol=0)
            for sequence, (left, right) in enumerate(pairwise(active_offsets)):
                if left == right:
                    torch.testing.assert_close(
                        final[sequence], initial[sequence], rtol=0, atol=0
                    )
            for storage, tensor, sentinel in guards:
                assert storage[0].item() == sentinel and storage[-1].item() == sentinel
                if tensor.stride(0) > tensor.shape[1] * 128:
                    padding = storage[1:-1].view(length, tensor.stride(0))[
                        :, tensor.shape[1] * 128 :
                    ]
                    assert bool((padding == sentinel).all())
            assert bool(
                (final_storage[:4] == 39).all() & (final_storage[-4:] == 39).all()
            )
    del graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
