import copy
import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from .reference_dspark import DSpark
from .reference_model import TorchWeights
from .test_reference_model import ArrayWeights


def fixture(*, active=False, rope=0):
    config = dict(
        hidden_size=2,
        vocab_size=6,
        num_hidden_layers=1,
        num_nextn_predict_layers=3,
        dspark_block_size=5,
        dspark_noise_token_id=5,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=1,
        dspark_num_experts_per_tok=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=0.0,
        rms_norm_eps=1e-20,
        max_position_embeddings=64,
        head_dim=2,
        num_attention_heads=1,
        o_groups=1,
        o_lora_rank=2,
        qk_rope_head_dim=rope,
        rope_theta=10000,
        sliding_window=3,
        compress_ratios=[0, 0, 0, 0],
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )
    arrays = {
        "embed.weight": np.array(
            [[3.0, 4.0], [1.0, -1.0], [-2.0, 1.0], [2.0, 3.0], [-1.0, -2.0], [0.0, 2.0]]
        ),
        "head.weight": np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0]]
        ),
        "mtp.0.main_proj.weight": np.array(
            [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]
        ),
        "mtp.0.main_norm.weight": np.ones(2),
        "mtp.2.norm.weight": np.array([2.0, 1.0]),
        "mtp.2.markov_head.embed.weight": np.arange(1, 7, dtype=np.float64)[:, None],
        "mtp.2.markov_head.head.weight": np.array(
            [[1.0], [2.0], [-1.0], [0.0], [0.5], [-0.5]]
        ),
        "mtp.2.confidence_head.proj.weight": np.array([[2.0, -1.0, 3.0]]),
    }
    for stage in range(3):
        prefix = f"mtp.{stage}"
        arrays[prefix + ".attn.attn_sink"] = np.zeros(1)
        arrays[prefix + ".ffn.gate.weight"] = np.zeros((2, 2))
        arrays[prefix + ".ffn.gate.bias"] = np.array([1.0, 0.0])
        for branch in ("attn", "ffn"):
            arrays[f"{prefix}.{branch}_norm.weight"] = np.ones(2)
            arrays[f"{prefix}.hc_{branch}_fn"] = np.zeros((8, 4))
            arrays[f"{prefix}.hc_{branch}_base"] = np.zeros(8)
            arrays[f"{prefix}.hc_{branch}_scale"] = np.ones(3)
        for norm in ("q_norm", "kv_norm"):
            arrays[f"{prefix}.attn.{norm}.weight"] = np.ones(2)
        for projection in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            arrays[f"{prefix}.attn.{projection}.weight"] = (
                np.eye(2)
                if projection in ("wkv", "wo_a") or (active and projection == "wo_b")
                else np.zeros((2, 2))
            )
        for expert in ("experts.0", "shared_experts"):
            for projection in ("w1", "w2", "w3"):
                arrays[f"{prefix}.ffn.{expert}.{projection}.weight"] = (
                    np.eye(2) * 0.2 if active else np.zeros((2, 2))
                )
    return config, arrays


def reference(config, arrays):
    source = ArrayWeights(arrays)
    return DSpark(
        TorchWeights(source, device="cpu", row_limit=2), {"text_config": config}
    ), source


def fp64(value):
    return torch.tensor(value, dtype=torch.float64)


class DSparkReference(unittest.TestCase):
    def test_seed_projects_main_features_and_keeps_only_ring_kv_across_chunks(self):
        config, arrays = fixture(rope=2)
        model, source = reference(config, arrays)
        target = fp64([[3, 4, 9, 8, 7, 6], [0, 2, 5, 4, 3, 2]])
        projected = model.project_features(target)
        torch.testing.assert_close(
            projected,
            fp64([[3 / math.sqrt(12.5), 4 / math.sqrt(12.5)], [0, math.sqrt(2)]]),
        )
        history = fp64([[1, 0], [0, 1], [1, 1], [-1, 0], [0, -1], [1, -1], [2, 1]])
        whole = model.seed(history)
        chunk = model.seed(history[:2])
        saved = tuple(window.clone() for window in chunk.windows)
        chunked = model.seed(history[2:5], start_pos=2, previous=chunk)
        chunked = model.seed(history[5:], start_pos=5, previous=chunked)
        self.assertEqual(whole.length, 7)
        for stage in range(3):
            torch.testing.assert_close(
                whole.windows[stage], chunked.windows[stage], rtol=0, atol=0
            )
            torch.testing.assert_close(
                chunk.windows[stage], saved[stage], rtol=0, atol=0
            )
            self.assertEqual(whole.windows[stage].shape, (3, 2))
        # Slots 0, 1, 2 hold positions 6, 4, 5 after the wrap.
        angle = 6
        expected = fp64(
            [
                [
                    math.sqrt(2 / 5) * (2 * math.cos(angle) - math.sin(angle)),
                    math.sqrt(2 / 5) * (2 * math.sin(angle) + math.cos(angle)),
                ]
            ]
        )
        torch.testing.assert_close(whole.windows[0][:1], expected)
        target_history = torch.cat((history, history, history), dim=-1)
        appended = model.append_main(target_history, start_pos=0, previous=None)
        projected_seed = model.seed(model.project_features(target_history))
        prefix = model.append_main(target_history[:2], start_pos=0, previous=None)
        saved_prefix = tuple(window.clone() for window in prefix.windows)
        continued = model.append_main(target_history[2:5], start_pos=2, previous=prefix)
        continued = model.append_main(
            target_history[5:], start_pos=5, previous=continued
        )
        for stage in range(3):
            torch.testing.assert_close(
                appended.windows[stage], projected_seed.windows[stage], rtol=0, atol=0
            )
            torch.testing.assert_close(
                appended.windows[stage], continued.windows[stage], rtol=0, atol=0
            )
            torch.testing.assert_close(
                prefix.windows[stage], saved_prefix[stage], rtol=0, atol=0
            )
        self.assertEqual(continued.length, 7)
        self.assertFalse(
            any(".ffn." in name or ".hc_" in name for name, _ in source.reads)
        )
        self.assertFalse(any(".wq_" in name for name, _ in source.reads))

    def test_full_head_has_hand_known_logits_and_raw_unnormalized_confidence(self):
        config, arrays = fixture()
        model, _ = reference(config, arrays)
        state = model.seed(fp64([[1, 0], [0, 1]]))
        output = model.forward(
            fp64([[1, 1]]),
            start_pos=2,
            input_token=0,
            previous=state,
            proposal_tokens=[1, 2, 3, 4, 0],
        )
        self.assertEqual(output.main_feature_position, 2)
        self.assertEqual(output.draft_row_positions, (3, 4, 5, 6, 7))
        self.assertEqual(output.draft_input_tokens, (0, 5, 5, 5, 5))
        self.assertEqual(output.input_token, 0)
        self.assertEqual(output.proposal_tokens, (1, 2, 3, 4, 0))
        self.assertEqual(output.previous_tokens, (0, 1, 2, 3, 4))
        expected_base = fp64(
            [
                [6, 4, -6, -4, 10, -10],
                [0, 5, 0, -5, 5, -5],
            ]
        ) / math.sqrt(12.5)
        expected_base = torch.cat((expected_base[:1], expected_base[1:].repeat(4, 1)))
        torch.testing.assert_close(
            output.base_logits, expected_base, rtol=1e-14, atol=1e-14
        )
        embeddings = fp64([[1], [2], [3], [4], [5]])
        torch.testing.assert_close(output.markov_embeddings, embeddings, rtol=0, atol=0)
        bias = embeddings * fp64([[1, 2, -1, 0, 0.5, -0.5]])
        torch.testing.assert_close(
            output.logits, expected_base + bias, rtol=1e-14, atol=1e-14
        )
        torch.testing.assert_close(
            output.collapsed_hidden, fp64([[3, 4]] + [[0, 2]] * 4)
        )
        torch.testing.assert_close(output.raw_confidence, fp64([5, 4, 7, 10, 13]))

    def test_attention_uses_all_five_bidirectional_rows_and_sink(self):
        config, arrays = fixture(active=True)
        model, _ = reference(config, arrays)
        window = fp64([[1, 0], [0, 0], [0, 0]])
        rows = fp64([[1, 0]] * 5)
        output = model._attention(rows, window, start_pos=0, stage=0)
        # Zero queries give each of six visible KV rows and the sink mass 1/7.
        expected = fp64([[(1 + 5 * math.sqrt(2)) / 7, 0]] * 5)
        torch.testing.assert_close(output, expected, rtol=1e-14, atol=1e-14)
        changed = rows.clone()
        changed[-1] = fp64([0, 1])
        difference = model._attention(changed, window, start_pos=0, stage=0) - output
        torch.testing.assert_close(
            difference,
            fp64([[-math.sqrt(2) / 7, math.sqrt(2) / 7]] * 5),
            rtol=1e-14,
            atol=1e-14,
        )
        state = model.seed(fp64([[1, 0]]))
        first = model.forward(
            fp64([[0, 1]]), start_pos=1, input_token=0, previous=state
        )
        other_arrays = copy.deepcopy(arrays)
        other_arrays["embed.weight"][5] = [2, -1]
        other, _ = reference(config, other_arrays)
        second = other.forward(
            fp64([[0, 1]]), start_pos=1, input_token=0, previous=state
        )
        self.assertFalse(torch.allclose(first.base_logits[0], second.base_logits[0]))
        config, arrays = fixture(active=True, rope=2)
        config.update(num_attention_heads=2, o_groups=2, o_lora_rank=1)
        arrays["mtp.0.attn.attn_sink"] = np.zeros(2)
        arrays["mtp.0.attn.wq_a.weight"] = np.eye(2)
        arrays["mtp.0.attn.wq_b.weight"] = np.array(
            [[1.0, 0], [0, 1.0], [0, 1.0], [-1.0, 0]]
        )
        arrays["mtp.0.attn.wo_a.weight"] = np.diag([2.0, 3.0])
        arrays["mtp.0.attn.wo_b.weight"] = np.array([[1.0, 2.0], [3.0, 4.0]])
        model, _ = reference(config, arrays)
        output = model._attention(rows, window * math.sqrt(2), start_pos=1, stage=0)
        # At row 0 (position 2), the main key at 0 has relative angle -2.
        # Draft relative angles are 0 through 4. One zero key and the sink add 2.
        angles = [-2, 0, 1, 2, 3, 4]
        first_mass = [math.exp(math.sqrt(2) * math.cos(i)) for i in angles]
        second_mass = [math.exp(-math.sqrt(2) * math.sin(i)) for i in angles]
        first_group = (
            2
            * math.sqrt(2)
            * sum(
                mass * math.cos(i) for i, mass in zip(angles, first_mass, strict=True)
            )
            / (2 + sum(first_mass))
        )
        second_group = (
            3
            * math.sqrt(2)
            * sum(
                mass * math.sin(i) for i, mass in zip(angles, second_mass, strict=True)
            )
            / (2 + sum(second_mass))
        )
        torch.testing.assert_close(
            output[0],
            fp64([first_group + 2 * second_group, 3 * first_group + 4 * second_group]),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_three_stages_use_delayed_hc_mixes(self):
        config, arrays = fixture(active=True)
        for stage in range(3):
            arrays[f"mtp.{stage}.hc_attn_base"][2:4] = [-1.0, 1.0]
            arrays[f"mtp.{stage}.hc_ffn_base"][2:4] = [-0.5, 0.5]
        model, _ = reference(config, arrays)
        state = model.seed(fp64([[1, 0]]))

        def run(changed):
            candidate, _ = reference(config, changed)
            return candidate.forward(
                fp64([[0, 1]]), start_pos=1, input_token=0, previous=state
            )

        baseline = run(arrays)
        for stage in range(3):
            for branch in ("attn.wo_b", "ffn.experts.0.w2", "ffn.shared_experts.w2"):
                with self.subTest(stage=stage, branch=branch):
                    changed = copy.deepcopy(arrays)
                    changed[f"mtp.{stage}.{branch}.weight"] *= 0
                    self.assertFalse(
                        torch.allclose(
                            baseline.collapsed_hidden,
                            run(changed).collapsed_hidden,
                            rtol=1e-10,
                            atol=1e-10,
                        )
                    )
        # The last attention's pre-mix is used by its FFN, not by its attention.
        zero_ffn = copy.deepcopy(arrays)
        for expert in ("experts.0", "shared_experts"):
            zero_ffn[f"mtp.2.ffn.{expert}.w2.weight"] *= 0
        before = run(zero_ffn)
        changed = copy.deepcopy(zero_ffn)
        changed["mtp.2.hc_attn_base"][:2] = [-4.0, 3.0]
        torch.testing.assert_close(
            before.collapsed_hidden, run(changed).collapsed_hidden, rtol=0, atol=0
        )
        changed = copy.deepcopy(zero_ffn)
        changed["mtp.1.hc_ffn_base"][:2] = [-4.0, 3.0]
        self.assertFalse(
            torch.allclose(before.collapsed_hidden, run(changed).collapsed_hidden)
        )
        changed = copy.deepcopy(arrays)
        changed["mtp.2.hc_attn_base"][:2] = [-4.0, 3.0]
        self.assertFalse(
            torch.allclose(baseline.collapsed_hidden, run(changed).collapsed_hidden)
        )

    def test_markov_uses_preceding_selected_token_and_greedy_is_repeatable(self):
        config, arrays = fixture()
        model, _ = reference(config, arrays)
        state = model.seed(fp64([[1, 0]]))
        arguments = dict(start_pos=1, input_token=0, previous=state)
        first = model.forward(
            fp64([[0, 1]]), proposal_tokens=[1, 2, 3, 4, 0], **arguments
        )
        changed = model.forward(
            fp64([[0, 1]]), proposal_tokens=[5, 2, 3, 4, 0], **arguments
        )
        torch.testing.assert_close(
            first.base_logits, changed.base_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.logits[[0, 2, 3, 4]], changed.logits[[0, 2, 3, 4]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            changed.logits[1] - first.logits[1], fp64([4, 8, -4, 0, 2, -2])
        )
        self.assertEqual(
            float(changed.raw_confidence[1] - first.raw_confidence[1]), 12.0
        )
        greedy = model.forward(fp64([[0, 1]]), **arguments)
        self.assertEqual(greedy.proposal_tokens, (4, 1, 1, 1, 1))
        forced = model.forward(
            fp64([[0, 1]]), proposal_tokens=greedy.proposal_tokens, **arguments
        )
        torch.testing.assert_close(greedy.logits, forced.logits, rtol=0, atol=0)

    def test_continuation_is_bounded_and_does_not_mutate_or_store_draft_rows(self):
        config, arrays = fixture(active=True, rope=2)
        model, _ = reference(config, arrays)
        history = fp64([[1, 0], [0, 1]])
        state = model.seed(history)
        original = tuple(window.clone() for window in state.windows)
        for position in range(2, 8):
            main = fp64([[position, -1]])
            saved = tuple(window.clone() for window in state.windows)
            output = model.forward(
                main,
                start_pos=position,
                input_token=position % 6,
                previous=state,
                proposal_tokens=[0, 1, 2, 3, 4],
            )
            history = torch.cat((history, main))
            direct = model.seed(history)
            self.assertEqual(output.state.length, position + 1)
            for stage in range(3):
                torch.testing.assert_close(
                    state.windows[stage], saved[stage], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    output.state.windows[stage], direct.windows[stage], rtol=0, atol=0
                )
                self.assertEqual(output.state.windows[stage].numel(), 6)
            state = output.state
        fresh = model.seed(history[:2])
        for stage in range(3):
            torch.testing.assert_close(
                fresh.windows[stage], original[stage], rtol=0, atol=0
            )
        saved = tuple(window.clone() for window in state.windows)
        invalid = [
            dict(start_pos=9),
            dict(start_pos=0),
            dict(input_token=True),
            dict(proposal_tokens=[0]),
            dict(proposal_tokens=[0, 1, 2, 3, 6]),
            dict(previous=replace(state, windows=state.windows[:2])),
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                arguments = dict(start_pos=8, input_token=0, previous=state)
                arguments.update(changes)
                model.forward(fp64([[1, 0]]), **arguments)
        for value in (
            torch.ones((1, 2)),
            fp64([[1, 0], [0, 1]]),
            fp64([[float("nan"), 0]]),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.forward(value, start_pos=8, input_token=0, previous=state)
        with self.assertRaises(ValueError):
            model.seed(fp64([]).reshape(0, 2))
        with self.assertRaises(ValueError):
            model.seed(fp64([[1, 0]]), start_pos=1)
        with self.assertRaises(ValueError):
            model.forward(
                fp64([[1, 0]]),
                start_pos=60,
                input_token=0,
                previous=replace(state, length=60),
            )
        for stage in range(3):
            torch.testing.assert_close(
                state.windows[stage], saved[stage], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
