"""Subgroup, alignment and long cancellation contracts for the native scan."""

import math

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


@pytest.mark.parametrize("length", [130, 4097])
def test_misaligned_output_and_value_boundaries_replay_changed_routes(length):
    torch.manual_seed(length)
    heads, chunks = 2, (length + 63) // 64 + 2
    raw_queries = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
    raw_keys = torch.empty_like(raw_queries)
    storages = []
    for shape, sentinel in (
        (raw_queries.shape, 43),
        (raw_queries.shape, 45),
        ((length, heads, 128), 37),
        ((length, heads, 128), 41),
    ):
        storage = torch.full(
            (math.prod(shape) + 2,),
            sentinel,
            device="cuda",
            dtype=torch.bfloat16,
        )
        storages.append((storage, sentinel, storage[1:-1].view(shape)))
    queries, keys, values, output = (view for _, _, view in storages)
    values.zero_()
    signed_values = torch.arange(length, device="cuda").remainder(3).sub(1)[:, None]
    for index, column in enumerate((15, 16, 31, 32, 63, 64, 95, 96, 127)):
        values[:, :, column] = signed_values * (1 if index % 2 else -1)
    initial_values = values.clone()
    decay = torch.full((length, heads), -0.001, device="cuda")
    beta = torch.full_like(decay, 0.9)
    starts = torch.tensor([0, 0, length - 1, length], device="cuda", dtype=torch.int64)
    initial = torch.randn(3, heads, 128, 128, device="cuda") * 0.01
    initial_copy = initial.clone()
    final_storage = torch.full(
        (initial.numel() + 8,), 39, device="cuda", dtype=torch.float32
    )
    final = final_storage[4:-4].view_as(initial)
    coefficients = torch.empty(chunks, heads, 64, 64, device="cuda", dtype=values.dtype)
    cumulative = torch.empty(chunks, heads, 64, device="cuda")
    conditioning = torch.empty(chunks, heads, device="cuda", dtype=torch.int32)
    transformed_keys = torch.empty(
        chunks, heads, 64, 128, device="cuda", dtype=values.dtype
    )
    transformed_values = torch.empty_like(transformed_keys, dtype=torch.float32)
    query_key_scores = torch.empty_like(coefficients)
    carried_keys = torch.empty_like(transformed_keys)
    module = compiled()

    def run():
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
            query_key_scores,
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
            query_key_scores,
            carried_keys,
            output,
            final,
            conditioning,
            decay,
            beta,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        raw_keys.normal_()
        normalized_queries, normalized_keys = normalize_qk(raw_queries, raw_keys)
        queries.copy_(normalized_queries)
        keys.copy_(normalized_keys)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for route in (0, 2, 1, 0):
            if route == 0:
                raw_keys.zero_()
                coordinates = torch.arange(length, device="cuda") % 64
                raw_keys.scatter_(2, coordinates.view(-1, 1, 1), 1)
            else:
                raw_keys.copy_(
                    torch.randn(1, 1, 128, device="cuda", dtype=raw_keys.dtype)
                )
                if route == 1:
                    raw_keys.add_(torch.randn_like(raw_keys) * 0.01)
            normalized_queries, normalized_keys = normalize_qk(raw_queries, raw_keys)
            queries.copy_(normalized_queries)
            keys.copy_(normalized_keys)
            expected = reference(
                raw_queries, raw_keys, values, decay, beta, initial, starts
            )
            graph.replay()
            verify(
                "gdn_prefill",
                lambda expected=expected: expected,
                lambda: (output, final),
            )
            assert set(conditioning[: (length - 1) // 64].flatten().tolist()) == {route}
            torch.testing.assert_close(initial, initial_copy, rtol=0, atol=0)
            torch.testing.assert_close(values, initial_values, rtol=0, atol=0)
            torch.testing.assert_close(queries, normalized_queries, rtol=0, atol=0)
            torch.testing.assert_close(keys, normalized_keys, rtol=0, atol=0)
            torch.testing.assert_close(final[0], initial[0], rtol=0, atol=0)
            for storage, sentinel, _ in storages:
                assert storage[0].item() == sentinel
                assert storage[-1].item() == sentinel
            assert (final_storage[:4] == 39).all() and (final_storage[-4:] == 39).all()
    del graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@pytest.mark.parametrize("beta_value", [0.999, 1.0])
def test_long_precise_scan_preserves_nearly_cancelled_state(beta_value):
    torch.manual_seed(428)
    length = 4097
    query = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 128, device="cuda", dtype=query.dtype)
    key = key + torch.randn_like(query) * 0.001
    value = torch.zeros_like(query)
    decay = torch.zeros(length, 1, device="cuda")
    beta = torch.full_like(decay, beta_value)
    _, normalized_key = normalize_qk(query, key)
    initial = torch.randn(1, 1, 128, device="cuda")[..., None]
    initial = initial * normalized_key[0].float()[None, :, None, :]
    starts = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    args = query, key, value, decay, beta, initial, starts
    expected = reference(*args)
    actual = prefill(*args)
    verify("gdn_prefill", lambda: expected, lambda: actual)
