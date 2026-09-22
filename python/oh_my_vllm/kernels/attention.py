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
        starts, page_starts, pages, last_page_lengths = (
            tensor.cpu() for tensor in (starts, page_starts, pages, last_page_lengths)
        )
        lengths_q = starts[1:] - starts[:-1]
        self.ragged = int(lengths_q.max()) >= 1024
        if self.ragged:
            # Compact only the active KV tokens for Blackwell's ragged FMHA.
            # Persistent pages and Rust allocation retain their 784-token layout.
            lengths_kv = (page_starts[1:] - page_starts[:-1] - 1) * 784
            lengths_kv = lengths_kv + last_page_lengths
            if bool((lengths_q <= 0).any()) or bool((lengths_kv < lengths_q).any()):
                raise ValueError(
                    "causal attention requires 0 < query length <= KV length"
                )
            slots = []
            offsets = torch.arange(784, dtype=torch.int64)
            for left, right, length in zip(
                page_starts[:-1].tolist(),
                page_starts[1:].tolist(),
                lengths_kv.tolist(),
                strict=True,
            ):
                table = pages[left:right].to(device="cpu", dtype=torch.int64)
                slots.append((table[:, None] * 1568 + offsets).flatten()[:length])
            device = self.workspace.device
            self.key_slots = torch.cat(slots).to(device)
            self.value_slots = self.key_slots + 784
            self.query_starts = starts.to(device)
            self.kv_starts = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32),
                    lengths_kv.cumsum(0, dtype=torch.int32),
                )
            ).to(device)
            self.kv_lengths = lengths_kv.to(device)
            self.max_q = int(lengths_q.max())
            self.max_kv = int(lengths_kv.max())
            self.scale = dim**-0.5
            return
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
        if self.ragged:
            from flashinfer.prefill import trtllm_ragged_attention_deepseek

            flat = cache.view(-1, *cache.shape[-2:])
            return trtllm_ragged_attention_deepseek(
                query,
                flat.index_select(0, self.key_slots),
                flat.index_select(0, self.value_slots),
                self.workspace,
                seq_lens=self.kv_lengths,
                max_q_len=self.max_q,
                max_kv_len=self.max_kv,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=len(self.kv_lengths),
                window_left=-1,
                cum_seq_lens_q=self.query_starts,
                cum_seq_lens_kv=self.kv_starts,
                enable_pdl=None,
                is_causal=True,
                return_lse=False,
                skip_all_rows_active_check=True,
                use_fp16_softmax=False,
            )
        return self.wrapper.run(query, cache)
