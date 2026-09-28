"""MTP eager page tables use the current extent and dynamic request mapping."""

from unittest.mock import patch

import pytest
import torch
from oh_my_vllm.kernels.mtp_attention import MTPAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def test_mtp_eager_plan_uses_extent_and_device_row_expansion():
    attention = MTPAttention(262144)
    starts = [0, 2, 5]
    pages = [list(range(1, 43)), list(range(101, 143))]
    positions = [32768, 32769, 32770, 32771, 32772]
    attention.plan(starts, pages, positions)
    assert attention.decode_mode
    assert attention.extent == 36864
    assert attention.tables.shape == (5, 48)
    assert attention.tables.dtype == torch.int32
    assert attention.tables[:2, :42].tolist() == [pages[0]] * 2
    assert attention.tables[2:, :42].tolist() == [pages[1]] * 3
    assert attention.tables[:, 42:].count_nonzero() == 0

    query = torch.empty(5, 24, 256, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    with patch("oh_my_vllm.kernels.mtp_attention.decode") as decode:
        attention(query, cache)
    assert decode.call_args.kwargs["max_tokens"] == 36864

    changed = [table.copy() for table in pages]
    changed[0][41] = 999
    attention.plan(starts, changed, positions)
    assert attention.tables[:2, 41].tolist() == [999, 999]
    assert attention.tables[2:, 41].tolist() == [142, 142, 142]
