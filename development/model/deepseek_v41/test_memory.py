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
    def allocation_budgets(self):
        return {
            "device_reserve_bytes": [2**30] * 4,
            "host_capacity_bytes": 2**41,
            "host_unavailable_bytes": 128 * 2**30,
            "host_reserve_bytes": 128 * 2**30,
            "request_state_max_bytes": 16 * 2**30,
            "response_max_bytes": 8 * 2**30,
            "pageable_reservations": {
                "control": 4 * 2**30,
                "request_payloads": 24 * 2**30,
                "media": 2 * 2**30,
                "processes": 16 * 2**30,
                "registration": 2 * 2**30,
                "kernel_io": 2 * 2**30,
            },
        }

    def test_allocation_caps_keep_full_dspark_and_separate_startup_storage(self):
        from .allocation import plan_allocation

        for strategy, layout_bytes in (
            ("TP4", 3127455680),
            ("TEP4", 64303040),
            ("DEP4", 180385280),
        ):
            with self.subTest(strategy=strategy):
                config = planner_config()
                config["host_pinned_capacity_bytes"] = 512 * 2**30
                plan = memory.plan_memory(config, strategy)
                result = plan_allocation(plan, **self.allocation_budgets())
                self.assertEqual(result["rows"], {"target": 384, "predictor": 320})
                self.assertEqual(result["routed_edges"], {"target": 2304, "draft": 960})
                self.assertFalse(result["runtime_feasibility_verified"])
                self.assertTrue(result["unresolved_bounds"])
                self.assertEqual(json.loads(json.dumps(result)), result)
                self.assertNotIn("tensors", result)
                for rank, source in zip(result["ranks"], plan["ranks"], strict=True):
                    native_layout = rank["weight_layout"]
                    self.assertEqual(native_layout["payload_bytes"], layout_bytes)
                    self.assertEqual(native_layout["cap_bytes"], 4 * 2**30)
                    self.assertEqual(
                        native_layout["spare_bytes"], 4 * 2**30 - layout_bytes
                    )
                    self.assertEqual(native_layout["lifetime"], "process")
                    self.assertEqual(
                        sum(b["bytes"] for b in native_layout["buffers"].values()),
                        layout_bytes,
                    )
                    self.assertEqual(
                        rank["arena_caps_bytes"]["weight_layout"], 4 * 2**30
                    )
                    self.assertEqual(
                        rank["persistent_bytes"],
                        source["weight_bytes"] + source["cache_bytes"],
                    )
                    self.assertEqual(rank["available_bytes"], 183 * 2**30)
                    self.assertEqual(
                        rank["startup_bytes"] - rank["runtime_bytes"], 128 * 2**20
                    )
                    self.assertEqual(
                        rank["runtime_bytes"],
                        rank["persistent_bytes"]
                        + sum(rank["arena_caps_bytes"].values()),
                    )
                    compute = result["device_arenas"]["compute"]
                    self.assertLessEqual(compute["payload_bytes"], 4 * 2**30)
                    self.assertEqual(
                        compute["buffers"]["expert_intermediates"]["shape"],
                        [2304, 2304],
                    )
                    self.assertEqual(
                        compute["buffers"]["draft_base_hidden"]["shape"], [320, 5120]
                    )
                    self.assertEqual(
                        result["device_arenas"]["staging"]["payload_bytes"], 14516224
                    )
                    self.assertEqual(rank["arena_caps_bytes"]["graphs"], 2 * 2**30)
                    self.assertEqual(rank["arena_caps_bytes"]["staging"], 2**30)
                host = result["host"]
                self.assertEqual(host["pinned_runtime_bytes"], 479247466496)
                self.assertEqual(host["pageable_runtime_bytes"], 50 * 2**30)
                self.assertEqual(host["runtime_bytes"], 532934557696)
                self.assertEqual(host["startup_bytes"], 534008299520)
                self.assertEqual(
                    host["pinned_startup_bytes"] - host["pinned_runtime_bytes"],
                    512 * 2**20,
                )

    def test_allocation_caps_reject_startup_shortage_and_invalid_children(self):
        from .allocation import _arena, plan_allocation

        config = planner_config()
        config["host_pinned_capacity_bytes"] = 512 * 2**30
        plan = memory.plan_memory(config, "TP4")
        budgets = self.allocation_budgets()
        result = plan_allocation(plan, **budgets)
        larger = dict(plan, ranks=[dict(rank) for rank in plan["ranks"]])
        larger["ranks"][0]["reservations"] = dict(
            plan["ranks"][0]["reservations"], staging={"charged_bytes": 2 * 2**30}
        )
        larger["host"] = dict(
            plan["host"],
            reservations={
                "kv_offload": 257 * 2**30,
                "uploads": 2**30,
                "transfer_leases": 2 * 2**30,
            },
        )
        increased = plan_allocation(larger, **budgets)
        self.assertEqual(
            increased["ranks"][0]["runtime_bytes"]
            - result["ranks"][0]["runtime_bytes"],
            2**30,
        )
        self.assertEqual(
            increased["ranks"][0]["arena_caps_bytes"]["staging"], 2 * 2**30
        )
        self.assertEqual(
            increased["host"]["runtime_bytes"] - result["host"]["runtime_bytes"],
            5 * 2**29,
        )
        self.assertEqual(
            plan["ranks"][0]["reservations"]["staging"]["charged_bytes"], 2**30
        )
        tiny = plan_allocation(
            plan, **dict(budgets, request_state_max_bytes=1, response_max_bytes=257)
        )
        self.assertEqual(
            tiny["host"]["pageable"]["request_payloads"]["payload_bytes"], 8192
        )
        short = dict(plan, ranks=[dict(rank) for rank in plan["ranks"]])
        short["ranks"][0]["available_bytes"] = (
            result["ranks"][0]["startup_bytes"] + 2**30 - 1
        )
        with self.assertRaisesRegex(ValueError, "startup"):
            plan_allocation(short, **budgets)
        weight = "layers.0.ffn.experts.0.w2.weight"
        scale = weight.removesuffix(".weight") + ".scale"
        for invalid in (
            {scale: None},
            {weight: None},
            {weight: dict(plan["tensors"][weight], placement="shared_pinned_host")},
            {scale: dict(plan["tensors"][scale], stored_shape=[5120, 73])},
            {
                scale: dict(
                    plan["tensors"][scale],
                    rank_resident_bytes=[92161, 92160, 92160, 92160],
                )
            },
        ):
            tensors = dict(plan["tensors"])
            for name, tensor in invalid.items():
                if tensor is None:
                    tensors.pop(name)
                else:
                    tensors[name] = tensor
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_allocation(dict(plan, tensors=tensors), **budgets)
        for key, value in (
            ("host_capacity_bytes", result["host"]["startup_bytes"] + 256 * 2**30 - 1),
            ("device_reserve_bytes", [True] * 4),
            ("host_reserve_bytes", -1),
            ("request_state_max_bytes", True),
            (
                "pageable_reservations",
                dict(budgets["pageable_reservations"], control=0),
            ),
            ("pageable_reservations", dict(budgets["pageable_reservations"], other=1)),
            (
                "pageable_reservations",
                dict(budgets["pageable_reservations"], processes=2**64 - 1),
            ),
        ):
            with (
                self.subTest(key=key, value=value),
                self.assertRaises((ValueError, OverflowError)),
            ):
                plan_allocation(plan, **dict(budgets, **{key: value}))
        pinned_short = dict(
            plan,
            host=dict(
                plan["host"], available_bytes=result["host"]["pinned_startup_bytes"] - 1
            ),
        )
        with self.assertRaisesRegex(ValueError, "pinned"):
            plan_allocation(pinned_short, **budgets)
        child = ("buffer", (257,), 1, 1)
        with self.assertRaises(ValueError):
            _arena(511, [child])
        with self.assertRaises(OverflowError):
            _arena(
                2**64 - 1,
                [("buffer", (2**64 - 1, 2), 1, 1)],
            )
        with self.assertRaises(ValueError):
            _arena(
                1024,
                [child, child],
            )

    def test_placements_count_bytes_not_blanket_quarters(self):
        from oh_my_vllm.models.deepseek_v41.placement import rank_plan

        model = checkpoint_config()
        stored = layout.tensor_layout(model)
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
                for rank in (0, 3):
                    selected = rank_plan(model, stored, strategy, rank)
                    self.assertEqual(
                        (selected.strategy, selected.rank), (strategy, rank)
                    )
                    self.assertEqual(selected.resident_bytes, expected_weights[rank])
                    self.assertEqual(
                        selected.source_bytes,
                        (85530033480 if rank == 0 else 84559496520)
                        if strategy == "DEP4"
                        else (78477964104 if rank == 0 else 77507427144),
                    )
                    slices = selected.selections
                    self.assertNotIn("layers.1.engram.embed.weight", slices)
                    self.assertNotIn("layers.1.engram.embed.scale", slices)
                    self.assertEqual("vision.norm.weight" in slices, rank == 0)
                    expert = f"layers.0.ffn.experts.{0 if rank == 0 else 383}.w2"
                    self.assertIn(f"{expert}.weight", slices)
                    other = f"layers.0.ffn.experts.{383 if rank == 0 else 0}.w2.weight"
                    self.assertEqual(other in slices, strategy == "TP4")
                    for suffix, width in (("weight", 1152), ("scale", 72)):
                        raw = slices[f"{expert}.{suffix}"]
                        self.assertEqual(
                            raw.dtype, "I8" if suffix == "weight" else "F8_E8M0"
                        )
                        start = rank * width // 4 if strategy == "TP4" else 0
                        stop = (rank + 1) * width // 4 if strategy == "TP4" else width
                        self.assertEqual(raw.shape, (5120, width))
                        self.assertEqual(raw.start, (0, start))
                        self.assertEqual(raw.stop, (5120, stop))
                    for suffix, rows in (("weight", 8192), ("scale", 256)):
                        raw = slices[f"layers.0.attn.wo_a.{suffix}"]
                        parts = 1 if strategy == "DEP4" else 4
                        start = 0 if parts == 1 else rank * rows // 4
                        self.assertEqual(raw.start, (start, 0))
                        self.assertEqual(
                            raw.stop,
                            (
                                start + rows // parts,
                                4096 if suffix == "weight" else 128,
                            ),
                        )
                    for suffix, width in (("weight", 8192), ("scale", 256)):
                        raw = slices[f"layers.0.attn.wo_b.{suffix}"]
                        start = rank * width // 4 if strategy != "DEP4" else 0
                        stop = (rank + 1) * width // 4 if strategy != "DEP4" else width
                        self.assertEqual(raw.start, (0, start))
                        self.assertEqual(
                            raw.stop, (5120 if suffix == "weight" else 160, stop)
                        )
                    draft = f"mtp.1.ffn.experts.{0 if rank == 0 else 127}.w1.weight"
                    self.assertIn(draft, slices)
                    self.assertEqual(
                        selected.promotions["layers.0.attn.wo_a.weight"], "BF16"
                    )
                    self.assertEqual(selected.promotions["head.weight"], "F32")
                    self.assertNotIn("layers.0.attn.wo_a.scale", selected.promotions)
                    with self.assertRaises(TypeError):
                        selected.selections["other"] = slices["norm.weight"]

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
        from oh_my_vllm.models.deepseek_v41.placement import (
            rank_plan,
            weight_placements,
        )

        model = checkpoint_config()
        stored = layout.tensor_layout(model)
        for strategy, rank in (
            ("TP8", 0),
            ("TP4", -1),
            ("TP4", 4),
            ("TP4", True),
            ("TP4", 1.0),
        ):
            with (
                self.subTest(strategy=strategy, rank=rank),
                self.assertRaises(ValueError),
            ):
                rank_plan(model, stored, strategy, rank)
        changed_model = copy.deepcopy(model)
        changed_model["text_config"]["hidden_size"] = 5121
        changed_layout = dict(stored)
        changed_layout["norm.weight"] = {"dtype": "F32", "shape": [5120]}
        for config, tensors in ((changed_model, stored), (model, changed_layout)):
            with self.assertRaises(ValueError):
                weight_placements(config, tensors, "TP4")
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
