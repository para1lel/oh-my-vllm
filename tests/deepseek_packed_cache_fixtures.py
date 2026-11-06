from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.packed_cache import (
    PackedHistory,
    TokenHistory,
)
from oh_my_vllm.models.deepseek_v41.quantization import fp4_act_quant


def packed_model():
    args = SimpleNamespace(max_batch_size=2, max_seq_len=16, lazy_cache=False)
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_kv_cache=torch.zeros(2, 4, 32),
                    compress_ratio=2,
                    is_kv_source=True,
                    compress_kv_cache=PackedHistory(
                        args, 32, 2, 16, torch.float8_e4m3fn
                    ),
                    indexer=SimpleNamespace(
                        owns_k=True,
                        k_cache=PackedHistory(args, 32, 2, 32, torch.float8_e8m0fnu),
                    ),
                    compressor=SimpleNamespace(
                        kv_state=torch.zeros(2, 2, 32),
                        score_state=torch.zeros(2, 2, 32),
                    ),
                )
            )
        ],
        mtp=[],
        engram_hash=SimpleNamespace(cache=TokenHistory(args)),
    )


def row(fa, state):
    return {"fa_block_table": fa, "mamba_block_table": state}


class DeepSeekPackedCacheFixture:
    def setUp(self):
        self.model = packed_model()
        self.cache = DeepSeekCache(
            self.model,
            block_tokens=4,
            max_seq_len=16,
            fa_capacity=16,
        )
        self.addCleanup(self.cache.close)

    def produce(self, slot, first, count):
        values = (torch.arange(count * 32).reshape(1, count, 32).float() - 17) / 11
        slots = torch.tensor([slot])
        for history in (
            self.model.layers[0].attn.compress_kv_cache,
            self.model.layers[0].attn.indexer.k_cache,
        ):
            encoded, scales = fp4_act_quant(
                values, history.group_size, scale_dtype=history.scale_dtype
            )
            history.write(slots, first, encoded, scales)
        export = self.cache.exports
        export.bind((slot,))
        attention = self.model.layers[0].attn
        state_values = (
            torch.arange(first * 2, (first + count) * 2)
            .float()
            .view(1, count * 2, 1)
            .expand(1, count * 2, 32)
        )
        export.window(attention.window_kv_cache, state_values, first * 2)
        export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            state_values,
            state_values + 10,
            first * 2,
        )
        return values
