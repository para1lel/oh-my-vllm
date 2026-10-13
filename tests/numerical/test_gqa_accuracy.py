"""Compare the owned paged GQA decode with a CPU FP64 reference."""

from __future__ import annotations

import math

import pytest
import torch
from oh_my_vllm.kernels.backend import NAME
from oh_my_vllm.kernels.decode_attention import decode

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(NAME != "cuda", reason="tests the owned CUDA decode path"),
]


def gqa_reference(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """One GQA decode step using CPU FP64 after BF16 input quantization."""
    _, heads, dim = query.shape
    groups = heads // keys.shape[2]
    expanded_keys = keys.repeat_interleave(groups, dim=2)
    expanded_values = values.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bhd,bthd->bht", query, expanded_keys) / math.sqrt(dim)
    return torch.einsum("bht,bthd->bhd", scores.softmax(-1), expanded_values)


@pytest.mark.parametrize(
    "batch,seq_len",
    [(1, 128), (2, 64), (1, 512), (2, 784), (1, 785)],
)
def test_owned_paged_gqa_matches_fp64(batch: int, seq_len: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(42)
    query = torch.randn(batch, 24, 256, generator=generator).bfloat16()
    keys = torch.randn(batch, seq_len, 4, 256, generator=generator).bfloat16()
    values = torch.randn(batch, seq_len, 4, 256, generator=generator).bfloat16()
    if seq_len == 785:
        # Make the first token of the second page dominate the attention.
        # A missing page must change the result far beyond the BF16 tolerance.
        query.fill_(1)
        keys[:, -1].fill_(1)
        values[:, -1].fill_(16)

    pages_per_row = math.ceil(seq_len / 784)
    cache = torch.zeros(
        1 + batch * pages_per_row,
        2,
        784,
        4,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    tables = torch.empty(batch, pages_per_row, device="cuda", dtype=torch.int32)
    for row in range(batch):
        for page in range(pages_per_row):
            physical = 1 + row * pages_per_row + pages_per_row - 1 - page
            tables[row, page] = physical
            start, stop = page * 784, min((page + 1) * 784, seq_len)
            cache[physical, 0, : stop - start] = keys[row, start:stop].cuda()
            cache[physical, 1, : stop - start] = values[row, start:stop].cuda()
    lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    assert NAME == "cuda"
    actual = decode(
        query.cuda(), cache, tables, lengths, max_tokens=784 * pages_per_row
    )
    expected = gqa_reference(query.double(), keys.double(), values.double())
    if seq_len == 785:
        assert expected.abs().min() > 8
        without_second_page = gqa_reference(
            query.double(), keys[:, :784].double(), values[:, :784].double()
        )
        assert (expected - without_second_page).abs().min() > 8
    torch.testing.assert_close(actual.cpu().double(), expected, atol=0.03, rtol=0.03)
