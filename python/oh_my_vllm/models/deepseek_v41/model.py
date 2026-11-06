"""DeepSeek model types and numerical helpers."""

import torch

from .compressor import Compressor as Compressor
from .model_args import ModelArgs as ModelArgs
from .model_attention import Attention as Attention
from .model_block import Block as Block
from .model_block import make_identity_pre_mix as make_identity_pre_mix
from .model_dspark import DSparkAttention as DSparkAttention
from .model_dspark import DSparkBlock as DSparkBlock
from .model_dspark import DSparkConfidenceHead as DSparkConfidenceHead
from .model_dspark import DSparkMarkovHead as DSparkMarkovHead
from .model_engram import Engram as Engram
from .model_engram import ParallelEngramEmbedding as ParallelEngramEmbedding
from .model_expert import Expert as Expert
from .model_expert import Gate as Gate
from .model_head import ParallelHead as ParallelHead
from .model_indexer import Indexer as Indexer
from .model_indexer import select_candidate_blocks as select_candidate_blocks
from .model_linear import ColumnParallelLinear as ColumnParallelLinear
from .model_linear import Linear as Linear
from .model_linear import ParallelEmbedding as ParallelEmbedding
from .model_linear import RMSNorm as RMSNorm
from .model_linear import RowParallelLinear as RowParallelLinear
from .model_linear import linear as linear
from .model_linear import request_einsum as request_einsum
from .model_linear import request_linear as request_linear
from .model_linear import separate_request_math as separate_request_math
from .model_linear import set_dtype as set_dtype
from .model_moe import MoE as MoE
from .model_positions import apply_rotary_emb as apply_rotary_emb
from .model_positions import compressed_context_width as compressed_context_width
from .model_positions import context_token_bucket as context_token_bucket
from .model_positions import get_dspark_topk_idxs as get_dspark_topk_idxs
from .model_positions import (
    get_prefill_window_topk_idxs as get_prefill_window_topk_idxs,
)
from .model_positions import get_window_topk_idxs as get_window_topk_idxs
from .model_positions import position_rows as position_rows
from .model_positions import position_window_topk_idxs as position_window_topk_idxs
from .model_positions import precompute_freqs_cis as precompute_freqs_cis
from .model_sampling import sample as sample
from .model_transformer import Transformer as Transformer

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(dspark_block_size=6, dspark_target_layer_ids=(3, 4))
    x = torch.randint(0, args.vocab_size, (2, 150))
    model = Transformer(args)

    output_ids, logits, main_hidden = model(x[:, :128])
    model.forward_spec(output_ids, main_hidden)
    for i in range(128, 150):
        output_ids, logits, main_hidden = model(x[:, i : i + 1], i)
        result = model.forward_spec(output_ids, main_hidden, i)
        assert result is not None
        output_ids, logits, confidence = result
