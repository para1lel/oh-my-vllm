from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.packed_cache import PackedHistory, TokenHistory


class FakeModel:
    def __init__(self, *, batch_size=2, context=8):
        args = SimpleNamespace(
            max_batch_size=batch_size, max_seq_len=context, lazy_cache=False
        )
        self.layers = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    is_kv_source=True,
                    window_kv_cache=torch.zeros(batch_size, 4, 3),
                    compress_kv_cache=PackedHistory(
                        args, 32, 2, 16, torch.float8_e4m3fn
                    ),
                    compress_ratio=2,
                    indexer=SimpleNamespace(
                        owns_k=True,
                        k_cache=PackedHistory(args, 32, 2, 32, torch.float8_e8m0fnu),
                    ),
                    compressor=SimpleNamespace(
                        kv_state=torch.zeros(batch_size, 2, 3),
                        score_state=torch.zeros(batch_size, 2, 3),
                    ),
                )
            )
        ]
        self.mtp = [
            SimpleNamespace(
                attn=SimpleNamespace(window_kv_cache=torch.zeros(batch_size, 4, 3))
            )
        ]
        self.engram_hash = SimpleNamespace(cache=TokenHistory(args))


def cache_for(model, **options):
    return DeepSeekCache(model, fa_capacity=128, **options)


def row(fa, mamba):
    return {
        "request_id": 7,
        "fa_block_table": fa,
        "mamba_block_table": mamba,
    }
