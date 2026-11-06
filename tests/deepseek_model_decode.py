import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.decode import (
    DecodeBindings,
    pause_state_export,
    sampled_token_ids,
)
from oh_my_vllm.models.deepseek_v41.engram import NgramHashState
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    Transformer,
    compressed_context_width,
    context_token_bucket,
)
from oh_my_vllm.models.deepseek_v41.model_context import shared_attn

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelDecode(DeepSeekModelFixture, unittest.TestCase):
    def test_varlen_target_calls_keep_per_request_logits_and_recurrent_state(self):
        previous_device, previous_dtype = (
            torch.get_default_device(),
            torch.get_default_dtype(),
        )
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = self._varlen_target_args()
            transformer = Transformer(args)
            generator = torch.Generator(device="cpu").manual_seed(0)
            for parameter in transformer.parameters():
                torch.nn.init.normal_(
                    parameter, mean=0.0, std=0.02, generator=generator
                )
            starts, slots = (9, 15), torch.tensor([1, 0])
            input_generator = torch.Generator(device="cpu").manual_seed(0)
            for slot, start in zip(slots.tolist(), starts, strict=True):
                transformer(
                    torch.randint(0, 128, (1, start), generator=input_generator),
                    cache_slots=torch.tensor([slot]),
                )
            initial = {
                name: value.clone() for name, value in transformer.named_buffers()
            }
            for length in (1, 6):
                tokens = self._check_varlen_model_rows(
                    transformer, initial, length, input_generator, starts, slots
                )

                self._check_uniform_target_batch(
                    starts, transformer, initial, tokens, slots, length, args
                )
        finally:
            shared_attn.reset_query_state()
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_engram_hash_reads_a_device_position(self):
        state = NgramHashState.__new__(NgramHashState)
        torch.nn.Module.__init__(state)
        state.layout = SimpleNamespace(max_ngram_size=2)
        state.pad_id = 0
        state.cache = torch.zeros(2, 16, dtype=torch.int64)
        state.cache[0, :6] = torch.tensor([3, 4, 5, 6, 7, 1])
        state.token_map = torch.arange(8)
        state.multipliers = torch.tensor([[1, 3]])
        state.primes = torch.tensor([[[11]]])
        state.offsets = torch.tensor([[0]])
        tokens = torch.tensor([[2]])
        slots = torch.tensor([0])
        initial = state.cache.clone()

        eager = state(tokens, start_pos=6, cache_slots=slots)
        eager_cache = state.cache.clone()
        state.cache.copy_(initial)
        device = state(
            tokens,
            start_pos=6,
            cache_slots=slots,
            position=torch.tensor(6, dtype=torch.int64),
        )
        self.assertTrue(torch.equal(eager, device))
        self.assertTrue(torch.equal(eager_cache, state.cache))

    def test_padded_context_widths_never_expose_padding_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
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
                index_topk=16,
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                temperature=0,
                max_batch_size=1,
                max_seq_len=32,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            model(torch.randint(0, 128, (1, 5)), start_pos=0)
            widths = {}
            for position in range(5, 13):
                model(
                    torch.randint(0, 128, (1, 1)),
                    start_pos=position,
                    position=torch.tensor(position, dtype=torch.int64),
                )
                # The last index source (layer 2, ratio 1) owns the published state.
                idxs = shared_attn.topk_idxs
                width = shared_attn.index_k.size(1)
                compress_len = position + 1
                self.assertGreaterEqual(width, compress_len)
                self.assertEqual(
                    width,
                    compressed_context_width(compress_len, 1, args.max_seq_len),
                )
                compressed = idxs[idxs >= args.window_size]
                self.assertTrue(
                    bool((compressed < args.window_size + compress_len).all()),
                    "an emitted index selected a padding row",
                )
                extent = context_token_bucket(compress_len)
                widths.setdefault(extent, set()).add((idxs.size(-1), width))
            for extent, shapes in widths.items():
                self.assertEqual(
                    len(shapes), 1, f"bucket {extent} changed shape: {shapes}"
                )
            self.assertEqual(compressed_context_width(13, 1, args.max_seq_len), 16)
        finally:
            shared_attn.topk_idxs = None
            shared_attn.compress_kv = None
            shared_attn.index_k = None
            shared_attn.candidates = None
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def test_decode_bindings_write_fixed_device_inputs(self):
        bindings = DecodeBindings(2, [3, 1], "cpu")
        bindings.bind(torch.tensor([7, 9]), 12)
        self.assertTrue(torch.equal(bindings.token_ids, torch.tensor([[7], [9]])))
        self.assertEqual(int(bindings.position), 12)
        self.assertTrue(torch.equal(bindings.cache_slots, torch.tensor([3, 1])))
        bindings.bind(torch.tensor([4, 5]), 13)
        self.assertTrue(torch.equal(bindings.token_ids, torch.tensor([[4], [5]])))
        self.assertEqual(int(bindings.position), 13)
        with self.assertRaises(ValueError):
            DecodeBindings(2, [1, 1], "cpu")
        with self.assertRaises(ValueError):
            DecodeBindings(2, [1], "cpu")
        single = DecodeBindings(1, [0], "cpu")
        single.bind(6, 4)
        self.assertTrue(torch.equal(single.token_ids, torch.tensor([[6]])))
        self.assertEqual(sampled_token_ids(torch.tensor([[1.0, 3.0, 2.0]])), [1])

    def test_decode_graph_validates_every_request_before_capture_and_replay(self):
        from oh_my_vllm.models.deepseek_v41.decode import DecodeGraph

        bindings = DecodeBindings(2, (3, 1), "cpu", length=6, varlen=True)
        graph = DecodeGraph.__new__(DecodeGraph)
        graph.bindings = bindings
        graph.start_pos = 14
        graph.context_bucket = 32
        graph.cycle = 2
        graph.residue = 0
        graph.window_prefixes = ((8, 8),)
        graph.routed_bindings = ()
        graph.closed = False
        graph.cache_lease = None
        graph.output = None
        graph.model = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "bind"):
            graph._validate_positions()
        for positions in ((12, 14), (14, 18), (18, 26)):
            bindings.bind([[1] * 6] * 2, positions)
            graph._validate_positions()
        for positions in ((8, 14), (13, 14), (6, 14), (14, 28)):
            bindings.bind([[1] * 6] * 2, positions)
            with (
                self.subTest(positions=positions),
                self.assertRaisesRegex(ValueError, "outside"),
            ):
                graph.replay()
        bindings.bind([[1] * 6] * 2, (8, 14))
        with (
            patch.object(graph, "close") as close,
            self.assertRaisesRegex(ValueError, "outside"),
        ):
            graph.capture()
        close.assert_called_once_with()
        graph.context_bucket = 16
        graph.start_pos = 6
        graph.window_prefixes = ((8, 6),)
        bindings.bind([[1] * 6] * 2, (6, 6))
        graph._validate_positions()
        bindings.bind([[1] * 6] * 2, (4, 6))
        with self.assertRaisesRegex(ValueError, "outside"):
            graph.replay()
        bindings.close()

    def test_capture_pauses_host_state_export(self):
        export = SimpleNamespace(steps={0: "step"}, batch=(0,))
        model = SimpleNamespace(_state_export=export)
        with pause_state_export(model):
            self.assertEqual(export.steps, {})
        self.assertEqual(export.steps, {0: "step"})
        self.assertEqual(export.batch, (0,))
        plain = SimpleNamespace()
        with pause_state_export(plain):
            pass
        self.assertFalse(hasattr(plain, "_state_export"))

    def test_captured_forward_defers_the_sparse_status_check(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.bfloat16)
        try:
            args = ModelArgs(
                vocab_size=64,
                dim=64,
                moe_inter_dim=64,
                n_layers=1,
                n_mtp_layers=0,
                n_heads=2,
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
                candidate_source_layer=-1,
                hc_mult=2,
                dtype="bf16",
                expert_dtype=None,
                max_batch_size=1,
                max_seq_len=8,
            )
            model = Transformer(args)
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            ids = torch.randint(0, 64, (1, 2))
            with self.assertRaises(ValueError):
                model(
                    ids,
                    start_pos=0,
                    position=torch.tensor([[0, 1]], dtype=torch.int64),
                )
            with patch(
                "oh_my_vllm.models.deepseek_v41.model_transformer.require_sparse_status"
            ) as status:
                model(ids, start_pos=0, check_status=False)
                status.assert_not_called()
                model(ids, start_pos=0)
                status.assert_called_once()
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def _check_uniform_target_batch(
        self, starts, transformer, initial, tokens, slots, length, args
    ):
        from oh_my_vllm.models.deepseek_v41.batch import target_decode_batch

        uniform_starts = (max(starts),) * len(starts)
        for name, value in transformer.named_buffers():
            value.copy_(initial[name])
        outputs = [
            transformer(
                tokens[index : index + 1],
                start_pos=start,
                cache_slots=slots[index : index + 1],
                full_logits=length > 1,
                sample_output=False,
                collect_main_hidden=False,
            )[1]
            for index, start in enumerate(uniform_starts)
        ]
        expected = {name: value.clone() for name, value in transformer.named_buffers()}
        for name, value in transformer.named_buffers():
            value.copy_(initial[name])
        from tests.test_deepseek_speculation import _worker

        worker = _worker()
        worker.model = transformer
        worker.args = args
        worker.decode_cycle = 2
        worker.decode_graph_limit = 0
        result = target_decode_batch(
            worker, tokens.tolist(), uniform_starts, slots.tolist()
        )
        torch.testing.assert_close(result.logits, torch.cat(outputs), rtol=0, atol=0)
        for name, value in transformer.named_buffers():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
        self.assertEqual(shared_attn.index_k.size(1), max(starts) + length)
        self.assertEqual(shared_attn.compress_kv.size(1), max(starts) + length)
        for binding in worker.decode_bindings.values():
            binding.close()

    def _check_varlen_model_rows(
        self, transformer, initial, length, input_generator, starts, slots
    ):
        tokens = torch.randint(0, 128, (2, length), generator=input_generator)
        for name, value in transformer.named_buffers():
            value.copy_(initial[name])
        outputs = []
        for index, start in enumerate(starts):
            outputs.append(
                transformer(
                    tokens[index : index + 1],
                    start_pos=start,
                    cache_slots=slots[index : index + 1],
                    full_logits=length > 1,
                    sample_output=False,
                    collect_main_hidden=False,
                )[1]
            )
        expected = {name: value.clone() for name, value in transformer.named_buffers()}
        for name, value in transformer.named_buffers():
            value.copy_(initial[name])
        plan = object()
        valid_rows = torch.ones(tokens.numel(), dtype=torch.bool)
        calls = []

        def dispatch(*, layer, rows, routing, valid_rows, plan, calls=calls):
            calls.append((layer.layer_id, valid_rows, plan))
            return layer._forward_rows(rows, routing=routing)

        actual = transformer(
            tokens,
            start_pos=max(starts),
            position=torch.tensor(starts).view(2, 1),
            cache_slots=slots,
            cache_slots_cpu=slots.tolist(),
            cache_positions_cpu=starts,
            full_logits=length > 1,
            sample_output=False,
            collect_main_hidden=False,
            logical_rows=1,
            execution_plan=plan,
            valid_rows=valid_rows,
            dispatch_combine=dispatch,
        )[1]
        self.assertEqual([call[0] for call in calls], [0, 1, 2])
        self.assertTrue(
            all(call[1] is valid_rows and call[2] is plan for call in calls)
        )
        with self.assertRaisesRegex(ValueError, "one device boolean"):
            transformer(tokens, valid_rows=valid_rows[:-1])
        torch.testing.assert_close(actual, torch.cat(outputs), rtol=0, atol=0)
        for name, value in transformer.named_buffers():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
        return tokens

    def _varlen_target_args(self):
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
            max_batch_size=2,
            max_seq_len=32,
        )
        return args
