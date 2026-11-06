import torch
from oh_my_vllm.models.deepseek_v41.model import (
    ParallelEngramEmbedding,
)


class DeepSeekModelFixture:
    def _engram_staging_table(self):
        table = ParallelEngramEmbedding(8, 32)
        values = torch.arange(8 * 32, dtype=torch.float32, device="cpu")
        table.weight.data.copy_((values.reshape(8, 32) / 16).to(table.weight.dtype))
        table.scale.data.copy_(
            torch.tensor([125, 126, 127, 128, 129, 127, 126, 125], dtype=torch.uint8)
            .reshape(8, 1)
            .view(torch.float8_e8m0fnu)
        )
        return table

    def _engram_lookup_reference(self, table, ids):
        ids = ids.cpu().long()
        mask = (ids < table.vocab_start_idx) | (ids >= table.vocab_end_idx)
        local = (ids - table.vocab_start_idx).masked_fill(mask, 0)
        values = torch.nn.functional.embedding(local, table.weight)
        scales = torch.nn.functional.embedding(local, table.scale)
        values = (
            (values.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1))
            .flatten(-2)
            .to(torch.bfloat16)
        )
        return values.masked_fill(mask.unsqueeze(-1), 0)
