"""Paged GQA over the framework's physical FA pages (784 tokens per page)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _append(
    K, V, Cache, Slots, Width: tl.constexpr, Page: tl.constexpr, Block: tl.constexpr
):
    token = tl.program_id(0)
    columns = tl.arange(0, Block)
    slot = tl.load(Slots + token).to(tl.int64)
    page, offset = slot // Page, slot % Page
    destination = page * 2 * Page * Width + offset * Width + columns
    key = tl.load(K + token * Width + columns, columns < Width, 0)
    value = tl.load(V + token * Width + columns, columns < Width, 0)
    tl.store(Cache + destination, key, (columns < Width) & (slot >= 0))
    tl.store(Cache + destination + Page * Width, value, (columns < Width) & (slot >= 0))


def append(
    cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor
) -> None:
    if cache.ndim != 5 or cache.shape[1:3] != (2, 784):
        raise ValueError("FA cache must have [pages,2,784,heads,dim] layout")
    if (
        k.ndim != 3
        or k.numel() == 0
        or v.shape != k.shape
        or k.shape[1:] != cache.shape[3:]
    ):
        raise ValueError("FA keys/values do not match cache head layout")
    if slots.shape != (k.shape[0],) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("FA append needs one integer slot per token")
    if any(t.dtype != torch.bfloat16 for t in (cache, k, v)):
        raise ValueError("FA cache and keys/values must be BF16")
    if any(
        not t.is_contiguous() or not t.is_cuda or t.device != k.device
        for t in (cache, k, v, slots)
    ):
        raise ValueError("FA tensors must be contiguous on the same CUDA device")
    width = k.shape[1] * k.shape[2]
    _append[(len(k),)](k, v, cache, slots, width, 784, triton.next_power_of_2(width))


class PagedAttention:
    """One batch plan shared by all full-attention layers for this execution."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.workspace = torch.empty(128 << 20, device=device, dtype=torch.uint8)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )

    def plan(
        self,
        starts: torch.Tensor,
        page_starts: torch.Tensor,
        pages: torch.Tensor,
        last_page_lengths: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        dim: int,
    ) -> None:
        self.wrapper.plan(
            starts,
            page_starts,
            pages,
            last_page_lengths,
            q_heads,
            kv_heads,
            dim,
            784,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return self.wrapper.run(query, cache)
