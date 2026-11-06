import unittest
import weakref
from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    Transformer,
    get_window_topk_idxs,
    position_window_topk_idxs,
)
from oh_my_vllm.models.deepseek_v41.model_context import shared_attn

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelTransformer(DeepSeekModelFixture, unittest.TestCase):
    def test_transformer_releases_first_layer_input_before_the_next_layer(self):
        first_inputs, checked_layers = [], []
        case = self

        class Layer(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index
                self.ffn = torch.nn.Identity()
                self.engram = None

            def forward(self, hidden, start, pre_mix, image_mask, slots, **options):
                if self.index == 0:
                    first_inputs.append(weakref.ref(hidden))
                else:
                    case.assertIsNone(first_inputs[0]())
                    checked_layers.append(self.index)
                return hidden.clone(), pre_mix

            @staticmethod
            def hc_pre(hidden, pre_mix):
                return hidden[:, :, 0].clone()

        transformer = Transformer.__new__(Transformer)
        torch.nn.Module.__init__(transformer)
        transformer.engram_hash = None
        transformer.embed = torch.nn.Embedding(8, 4, device="cpu")
        transformer.hc_mult = 4
        transformer.target_layer_ids = ()
        transformer.layers = torch.nn.ModuleList([Layer(0), Layer(1)])
        transformer._moe_prefill_workspace = SimpleNamespace(rows=1024)
        transformer(
            torch.tensor([[1, 2]], device="cpu"),
            compute_logits=False,
            check_status=False,
            collect_main_hidden=False,
        )
        self.assertEqual(checked_layers, [1])

    def test_transformer_can_skip_vocab_projection_for_prefill(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=0,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
                dspark_target_layer_ids=(0,),
            )
            model = Transformer(args)
            _, logits, hidden = model(
                torch.randint(0, 128, (1, 2)),
                compute_logits=False,
            )
            self.assertIsNone(logits)
            self.assertEqual(tuple(hidden.shape), (1, 2, 64))
            _, logits, hidden = model(
                torch.randint(0, 128, (1, 2)),
                compute_logits=False,
                collect_main_hidden=False,
            )
            self.assertIsNone(logits)
            self.assertIsNone(hidden)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_transformer_clears_shared_attention_state_before_forward(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        shared_attn.topk_idxs = torch.ones(4, 1, 1, dtype=torch.int32)
        shared_attn.candidates = torch.ones(4, 1, dtype=torch.bool)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(0,),
                kv_source_layers=(),
                index_source_layers=(),
                index_topk=0,
                candidate_source_layer=-1,
                candidate_topk_blocks=0,
                candidate_block_size=0,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            Transformer(args)(torch.randint(0, 128, (1, 1)))
            self.assertIsNone(shared_attn.topk_idxs)
            self.assertIsNone(shared_attn.candidates)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_index_cache_tracks_cache_slots_when_a_group_is_incomplete(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=128,
                dim=64,
                moe_inter_dim=64,
                n_layers=2,
                n_mtp_layers=0,
                n_heads=4,
                n_routed_experts=4,
                n_activated_experts=2,
                q_lora_rank=32,
                head_dim=32,
                rope_head_dim=16,
                o_groups=4,
                o_lora_rank=16,
                window_size=4,
                compress_ratios=(2, 2),
                kv_source_layers=(0,),
                index_source_layers=(0, 1),
                index_topk=4,
                candidate_source_layer=0,
                candidate_topk_blocks=1,
                candidate_block_size=2,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=4,
                max_seq_len=32,
            )
            model = Transformer(args)
            model(torch.randint(0, 128, (4, 4)), cache_slots=torch.arange(4))
            output = model(
                torch.randint(0, 128, (1, 1)),
                start_pos=4,
                cache_slots=torch.tensor([0]),
            )
            self.assertEqual(tuple(output[0].shape), (1,))
            self.assertEqual(
                tuple(shared_attn.index_k.shape), (1, 2, args.index_head_dim)
            )
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_device_window_indices_match_the_integer_path(self):
        for window_size in (1, 4, 8):
            for position in range(1, 2 * window_size + 3):
                eager = get_window_topk_idxs(window_size, 2, 1, position)
                device = position_window_topk_idxs(
                    window_size, 2, torch.tensor(position, dtype=torch.int64)
                )
                self.assertTrue(torch.equal(eager, device))

    def test_device_position_decode_matches_the_integer_path(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = self._device_position_args()
            model = Transformer(args)
            # Equal top-k cutoff scores can select different keys after context padding.
            generator = torch.Generator(device="cpu").manual_seed(0)
            for parameter in model.parameters():
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            initial = {name: value.clone() for name, value in model.named_buffers()}
            tokens = torch.randint(0, 128, (1, 12), generator=generator)

            def decode(with_device_position):
                model(tokens[:, :5], start_pos=0)
                logits, ids = [], []
                for position in range(5, 12):
                    cursor = (
                        torch.tensor(position, dtype=torch.int64)
                        if with_device_position
                        else None
                    )
                    output, step_logits, _ = model(
                        tokens[:, position : position + 1],
                        start_pos=position,
                        position=cursor,
                    )
                    logits.append(step_logits)
                    ids.append(output)
                return logits, ids

            eager_logits, eager_ids = decode(False)
            eager_buffers = {
                name: value.clone() for name, value in model.named_buffers()
            }
            for name, value in model.named_buffers():
                value.copy_(initial[name])
            device_logits, device_ids = decode(True)
            for index in range(len(eager_logits)):
                self.assertTrue(
                    torch.equal(eager_logits[index], device_logits[index]),
                    f"decode step {index} logits differ",
                )
                self.assertTrue(torch.equal(eager_ids[index], device_ids[index]))
            for name, value in model.named_buffers():
                self.assertTrue(
                    torch.equal(eager_buffers[name], value),
                    f"{name} differs after the device position path",
                )
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def _device_position_args(self):
        args = ModelArgs(
            vocab_size=128,
            dim=64,
            moe_inter_dim=64,
            n_layers=3,
            n_mtp_layers=0,
            n_heads=4,
            n_routed_experts=4,
            n_activated_experts=2,
            q_lora_rank=32,
            head_dim=32,
            rope_head_dim=16,
            o_groups=4,
            o_lora_rank=16,
            window_size=4,
            compress_ratios=(0, 2, 1),
            kv_source_layers=(1, 2),
            index_source_layers=(1, 2),
            index_n_heads=4,
            index_head_dim=32,
            index_topk=4,
            candidate_source_layer=-1,
            hc_mult=2,
            dtype="bf16",
            expert_dtype=None,
            temperature=0,
            max_batch_size=1,
            max_seq_len=16,
        )
        return args
