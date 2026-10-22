import copy
import json
import unittest
from unittest.mock import patch

from . import layout, memory
from .test_layout import checkpoint_config


def planner_config():
    return {
        "model": checkpoint_config(),
        "device_capacity_bytes": [192 * 2**30] * 4,
        "device_unavailable_bytes": [8 * 2**30] * 4,
        "host_pinned_capacity_bytes": 256 * 2**30,
        "host_reservations": {"kv_offload": 0, "uploads": 0, "transfer_leases": 0},
        "reservations": {
            name: [2**30] * 4
            for name in ("graphs", "staging", "communication", "chunks", "media")
        },
    }


class MemoryPlanTests(unittest.TestCase):
    def test_placements_count_bytes_not_blanket_quarters(self):
        plans = {
            strategy: memory.plan_memory(planner_config(), strategy)
            for strategy in ("TP4", "TEP4", "DEP4")
        }
        for strategy, plan in plans.items():
            with self.subTest(strategy=strategy):
                self.assertEqual(plan["tensor_count"], 96085)
                # Class sums include promotions and rank-0 vision (970536960 bytes).
                expected_weights = (
                    [88392951624, 87422414664, 87422414664, 87422414664]
                    if strategy == "DEP4"
                    else [79217294664, 78246757704, 78246757704, 78246757704]
                )
                self.assertEqual(
                    [r["weight_bytes"] for r in plan["ranks"]], expected_weights
                )
                self.assertEqual(plan["host"]["engram_table_bytes"], 196613849600)
                self.assertEqual(plan["host"]["engram_scale_bytes"], 6144182800)
                self.assertEqual(plan["host"]["engram_authoritative_copies"], 1)
                tensors = plan["tensors"]
                table = tensors["layers.1.engram.embed.weight"]
                self.assertEqual(table["placement"], "shared_pinned_host")
                self.assertEqual(table["rank_resident_bytes"], [0] * 4)
                self.assertEqual(
                    tensors["layers.0.hc_attn_fn"]["rank_resident_bytes"],
                    [1966080] * 4,
                )
                self.assertEqual(
                    tensors["layers.0.attn.q_norm.weight"]["rank_resident_bytes"],
                    [2560] * 4,
                )
                self.assertEqual(
                    tensors["layers.1.engram.wkv.weight"]["rank_resident_bytes"],
                    [157286400] * 4,
                )
                self.assertEqual(
                    tensors["vision.patch_embed.proj.weight"]["rank_resident_bytes"],
                    [1204224, 0, 0, 0],
                )
                self.assertEqual(
                    tensors["mtp.2.confidence_head.proj.weight"]["rank_resident_bytes"],
                    [21504] * 4,
                )
                dense_parts = 1 if strategy == "DEP4" else 4
                self.assertEqual(
                    tensors["head.weight"]["rank_resident_bytes"],
                    [2647654400 // dense_parts] * 4,
                )
                self.assertEqual(
                    tensors["layers.0.attn.wo_a.weight"]["rank_resident_bytes"],
                    [67108864 // dense_parts] * 4,
                )
                self.assertEqual(
                    tensors["layers.0.attn.wo_a.scale"]["rank_resident_bytes"],
                    [0] * 4,
                )
                expert = tensors["layers.0.ffn.experts.0.w2.weight"]
                scale = tensors["layers.0.ffn.experts.0.w2.scale"]
                if strategy == "TP4":
                    self.assertEqual(expert["rank_resident_bytes"], [1474560] * 4)
                    self.assertEqual(scale["rank_resident_bytes"], [92160] * 4)
                    self.assertEqual(expert["shard_axis"], 1)
                else:
                    self.assertEqual(expert["rank_resident_bytes"], [5898240, 0, 0, 0])
                    self.assertEqual(scale["rank_resident_bytes"], [368640, 0, 0, 0])
                    self.assertEqual(expert["owner_rank"], 0)
                    self.assertEqual(
                        tensors["mtp.1.ffn.experts.127.w1.weight"]["owner_rank"], 3
                    )
                    for expert, owner in (
                        (95, 0),
                        (96, 1),
                        (191, 1),
                        (192, 2),
                        (287, 2),
                        (288, 3),
                        (383, 3),
                    ):
                        self.assertEqual(
                            tensors[f"layers.0.ffn.experts.{expert}.w1.weight"][
                                "owner_rank"
                            ],
                            owner,
                        )
                for rank in plan["ranks"]:
                    self.assertEqual(
                        rank["weight_bytes"],
                        sum(
                            t["rank_resident_bytes"][rank["rank"]]
                            for t in tensors.values()
                        ),
                    )

    def test_combined_capacity_keeps_all_layers_and_speculative_state(self):
        for strategy, requests, global_bytes in (
            ("TP4", 64, 59726888960),
            ("TEP4", 64, 59726888960),
            ("DEP4", 16, 14931722240),
        ):
            with self.subTest(strategy=strategy):
                config = planner_config()
                original = copy.deepcopy(config)
                plan = memory.plan_memory(config, strategy)
                self.assertEqual(config, original)
                self.assertEqual(plan["limits"]["requests"], 64)
                self.assertEqual(plan["limits"]["context_tokens"], 1048576)
                self.assertEqual(plan["limits"]["verification_rows"], 6)
                self.assertFalse(plan["runtime_feasibility_verified"])
                self.assertTrue(plan["unknown_workspaces"])
                self.assertEqual(json.loads(json.dumps(plan)), plan)
                rank = plan["ranks"][0]
                self.assertEqual(rank["requests"], requests)
                cache = rank["cache"]
                self.assertEqual(cache["global_sources"]["bytes"], global_bytes)
                self.assertEqual(cache["target_windows"]["rows"], requests * 5120)
                self.assertEqual(cache["target_windows"]["bytes"], requests * 2703360)
                self.assertEqual(cache["draft_windows"]["rows"], requests * 384)
                self.assertEqual(cache["draft_windows"]["bytes"], requests * 202752)
                self.assertEqual(
                    cache["partial_compressors"]["bytes"], requests * 24576
                )
                self.assertEqual(
                    cache["target_window_shadow"]["bytes"], requests * 126720
                )
                self.assertEqual(cache["global_shadow"]["bytes"], requests * 5340)
                self.assertEqual(
                    cache["proposal_probabilities"]["bytes"], requests * 2585600
                )
                self.assertEqual(cache["target_features"]["bytes"], requests * 30720)
                self.assertEqual(
                    rank["accounted_bytes"],
                    rank["weight_bytes"]
                    + rank["cache_bytes"]
                    + rank["reservation_bytes"],
                )
                self.assertEqual(rank["available_bytes"], 184 * 2**30)
                self.assertEqual(
                    rank["headroom_bytes"],
                    rank["available_bytes"] - rank["accounted_bytes"],
                )
                expected_comm = 47259648 if strategy == "DEP4" else 248217600
                self.assertEqual(
                    rank["reservations"]["communication"]["minimum_bytes"],
                    expected_comm,
                )

    def test_budget_shortfalls_and_uneven_request_ownership(self):
        config = planner_config()
        config["limits"] = {"requests": 3, "context_tokens": 129, "draft_tokens": 0}
        plan = memory.plan_memory(config, "DEP4")
        self.assertEqual([r["requests"] for r in plan["ranks"]], [1, 1, 1, 0])
        self.assertEqual(
            [r["request_slots"] for r in plan["ranks"]], [[0], [1], [2], []]
        )
        self.assertEqual(plan["ranks"][0]["cache"]["global_sources"]["bytes"], 114276)
        self.assertEqual(plan["ranks"][3]["cache"]["global_sources"]["bytes"], 0)
        self.assertEqual(
            plan["ranks"][0]["cache"]["proposal_probabilities"]["bytes"], 0
        )
        config["device_capacity_bytes"] = [1] * 4
        config["device_unavailable_bytes"] = [0] * 4
        config["host_pinned_capacity_bytes"] = 1
        plan = memory.plan_memory(config, "TP4")
        self.assertFalse(plan["fits_supplied_budgets"])
        self.assertLess(plan["host"]["headroom_bytes"], 0)
        self.assertTrue(all(not r["fits_budget"] for r in plan["ranks"]))
        for values in config["reservations"].values():
            values[:] = [0] * 4
        plan = memory.plan_memory(config, "TP4")
        self.assertGreater(
            plan["ranks"][0]["reservations"]["staging"]["minimum_bytes"], 0
        )
        self.assertGreater(
            plan["ranks"][0]["reservations"]["chunks"]["minimum_bytes"], 0
        )

    def test_rejects_invalid_topology_overflow_and_unmapped_storage(self):
        for field, value in (
            ("device_capacity_bytes", [1] * 3),
            ("device_unavailable_bytes", [-1] * 4),
            ("device_capacity_bytes", [True] * 4),
            ("limits", {"requests": 65}),
            ("limits", {"context_tokens": 1048577}),
            ("limits", {"draft_tokens": 6}),
            ("limits", {"draft_tokens": 4}),
            ("limits", {"requests_per_rank": [64, 0, 0, 0]}),
            ("reservations", {"graphs": [0] * 4}),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                config = planner_config()
                config[field] = value
                memory.plan_memory(config, "DEP4")
        with self.assertRaises(ValueError):
            memory.plan_memory(planner_config(), "TP8")
        config = planner_config()
        config["reservations"]["graphs"] = [2**64 - 1] * 4
        with self.assertRaises(OverflowError):
            memory.plan_memory(config, "TP4")
        tensors = layout.tensor_layout(checkpoint_config())
        for name, spec in (
            ("layers.0.unmapped.weight", {"dtype": "BF16", "shape": [32, 32]}),
            (
                "layers.0.ffn.experts.0.w2.weight",
                {"dtype": "I8", "shape": [5120, 1154]},
            ),
            (
                "layers.0.ffn.experts.384.w1.weight",
                {"dtype": "I8", "shape": [2304, 2560]},
            ),
        ):
            changed = dict(tensors)
            changed[name] = spec
            with (
                self.subTest(tensor=name),
                patch.object(memory, "tensor_layout", return_value=changed),
                self.assertRaises(ValueError),
            ):
                memory.plan_memory(planner_config(), "TP4")


if __name__ == "__main__":
    unittest.main()
