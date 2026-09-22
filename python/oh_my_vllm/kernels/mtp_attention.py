"""MTP attention over token-indexed pages, with cache position zero absent."""

from itertools import pairwise

import torch

from oh_my_vllm.kernels.decode_attention import decode


class MTPAttention:
    def __init__(self, max_tokens: int) -> None:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        self.max_tokens = max_tokens
        self.workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, "NHD", backend="fa2"
        )

    def plan(
        self, starts: list[int], pages: list[list[int]], positions: list[int]
    ) -> None:
        counts = [b - a for a, b in pairwise(starts)]
        if len(counts) != len(pages) or not counts or min(counts) <= 0:
            raise ValueError("MTP attention requires nonempty sequences")
        if starts[0] != 0 or starts[-1] != len(positions):
            raise ValueError("MTP query offsets do not match positions")
        ends = []
        for i, count in enumerate(counts):
            query = positions[starts[i] : starts[i + 1]]
            if query[0] < 1 or query != list(range(query[0], query[0] + count)):
                raise ValueError(
                    "MTP positions must be contiguous and start after zero"
                )
            end = query[-1] + 1
            if end > self.max_tokens or len(pages[i]) * 784 < end:
                raise ValueError("MTP queries exceed allocated pages")
            ends.append(end)
        self.decode_mode = max(counts) <= 5
        if self.decode_mode:
            width = (self.max_tokens + 783) // 784
            tables = [
                table + [0] * (width - len(table))
                for table, count in zip(pages, counts, strict=True)
                for _ in range(count)
            ]
            self.tables = torch.tensor(tables, dtype=torch.int32, device="cuda")
            self.lengths = torch.tensor(
                [p + 1 for p in positions], dtype=torch.int32, device="cuda"
            )
        else:
            slots, offsets = [], [0]
            for table, end in zip(pages, ends, strict=True):
                slots.extend(table[p // 784] * 784 + p % 784 for p in range(1, end))
                offsets.append(len(slots))
            self.slots = torch.tensor(slots, device="cuda")
            self.native = max(counts) >= 128
            if self.native:
                self.query_starts = torch.tensor(
                    starts, dtype=torch.int32, device="cuda"
                )
                self.kv_starts = torch.tensor(offsets, dtype=torch.int32, device="cuda")
                self.lengths = self.kv_starts[1:] - self.kv_starts[:-1]
                self.max_query = max(counts)
                self.max_kv = max(ends) - 1
                return
            self.wrapper.plan(
                torch.tensor(starts, dtype=torch.int32),
                torch.tensor(offsets, dtype=torch.int32),
                24,
                4,
                256,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )

    def __call__(self, query: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        if self.decode_mode:
            return decode(
                query,
                cache,
                self.tables,
                self.lengths,
                first=1,
                max_tokens=self.max_tokens,
            )
        pages, offsets = self.slots // 784, self.slots % 784
        k, v = cache[pages, 0, offsets], cache[pages, 1, offsets]
        if self.native:
            from flashinfer.prefill import trtllm_ragged_attention_deepseek

            return trtllm_ragged_attention_deepseek(
                query,
                k,
                v,
                self.workspace,
                seq_lens=self.lengths,
                max_q_len=self.max_query,
                max_kv_len=self.max_kv,
                bmm1_scale=256**-0.5,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=len(self.lengths),
                window_left=-1,
                cum_seq_lens_q=self.query_starts,
                cum_seq_lens_kv=self.kv_starts,
                enable_pdl=False,
                is_causal=True,
                return_lse=False,
                skip_all_rows_active_check=True,
                use_fp16_softmax=False,
            )
        return self.wrapper.run(query, k, v)
