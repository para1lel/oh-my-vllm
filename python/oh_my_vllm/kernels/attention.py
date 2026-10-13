"""Paged GQA over the framework's physical FA pages (784 tokens per page)."""

import torch

from .backend import kernel

_append = kernel("attention", "_append")


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
    torch._assert_async(
        torch.all((slots < 0) | (slots < len(cache) * 784)),
        "FA append slot exceeds cache capacity",
    )
    _append(len(cache), *k.shape[1:], str(slots.dtype).removeprefix("torch."))(
        k, v, cache, slots
    )


class PagedAttention:
    """Immutable host sequence offsets and device cache addresses for one step."""

    def __init__(self, device: str | torch.device = "cuda", *, workspace=None) -> None:
        from oh_my_vllm.ir.attention import register_plan

        self.workspace = (
            torch.empty(0, device=device, dtype=torch.uint8)
            if workspace is None
            else workspace
        )
        self._ir_handle = register_plan(self)

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
        starts, page_starts, pages, last_page_lengths = (
            tensor.cpu() for tensor in (starts, page_starts, pages, last_page_lengths)
        )
        if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads or dim != 256:
            raise ValueError("unsupported paged attention head layout")
        self._ir_reference_plan = starts, page_starts, pages, last_page_lengths
        lengths_q = starts[1:] - starts[:-1]
        lengths_kv = (page_starts[1:] - page_starts[:-1] - 1) * 784
        lengths_kv = lengths_kv + last_page_lengths
        if bool((lengths_q <= 0).any()) or bool((lengths_kv < lengths_q).any()):
            raise ValueError("causal attention requires 0 < query length <= KV length")
        self.query_offsets = starts.tolist()
        self.kv_offsets = [0, *lengths_kv.cumsum(0).tolist()]
        offsets = torch.arange(784, dtype=torch.int64)
        slots = []
        for left, right, length in zip(
            page_starts[:-1].tolist(),
            page_starts[1:].tolist(),
            lengths_kv.tolist(),
            strict=True,
        ):
            table = pages[left:right].to(dtype=torch.int64)
            slots.append((table[:, None] * 1568 + offsets).flatten()[:length])
        self.key_slots = torch.cat(slots).to(self.workspace.device)
        self.value_slots = self.key_slots + 784

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        from oh_my_vllm.ir.attention import PAGED_PREFILL, planned_attention

        return planned_attention(query, cache, self._ir_handle, PAGED_PREFILL)

    def _run(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        tables: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .ragged_attention import causal_attention

        flat = cache.view(-1, *cache.shape[-2:])
        return causal_attention(
            query,
            flat.index_select(0, self.key_slots),
            flat.index_select(0, self.value_slots),
            self.query_offsets,
            self.kv_offsets,
        )
