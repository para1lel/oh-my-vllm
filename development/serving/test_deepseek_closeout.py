import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.models.deepseek_v41.trace import execution_modes, runtime_identity

from tests.test_deepseek_accounting_floor import routed_trace, split_decode

from . import deepseek_acceptance as gate
from . import deepseek_accounting as source
from . import deepseek_accounting_floor as floor
from . import deepseek_accounting_math as arithmetic
from . import deepseek_accounting_packed as packed
from . import deepseek_accounting_roofline as roofline
from .deepseek_benchmark import reference_configuration, runtime_reference
from .deepseek_runtime_trace import assemble_trace, response_manifest
from .test_deepseek_benchmark import run_fixture, runtime_fixture

MODEL = replace(
    arithmetic.DEEPSEEK_V41_MODEL,
    num_layers=1,
    compress_ratios=(0,),
    kv_source_layers=(),
    index_source_layers=(),
    engram_layers=(),
)


def measured_rows(version=roofline.VERSION):
    configuration = reference_configuration(version)
    modes = {
        strategy: {
            "batch_decode": True,
            "decode_graph": strategy == "tp4",
            "decode_graph_limit": 32,
        }
        for strategy in ("tp4", "tep4", "dep4")
    }
    identity = {
        "source": "source-digest",
        "source_commit": "commit",
        "binary": "binary-digest",
        "native": {
            "projection": {"source_id": "native-source", "binary_id": "native-binary"}
        },
        "checkpoint": {"config": "config", "index": "index"},
        "tokenizer": {"tokenizer": "digest"},
        "topology": [
            {"uuid": f"GPU-{rank}", "name": "NVIDIA B200"} for rank in range(4)
        ],
        "harness_numeric_runtime": {"python": "test"},
        "configuration": {"execution_modes": modes},
        "reference": configuration,
    }
    rows = []
    for identifier, case in gate.LONG_CASES.items():
        binding = {
            "source": {"commit": "commit", "digest": "source-digest"},
            "native_modules": {
                "projection": {"source": "native-source", "binary": "native-binary"}
            },
            "checkpoint_config": "config",
            "checkpoint_index": "index",
            "tokenizer": identity["tokenizer"],
            "gpu_uuids": [f"GPU-{rank}" for rank in range(4)],
            "runtime_modes": modes[case["strategy"]],
            "case_id": identifier,
            "strategy": case["strategy"],
            "effective_strategy": case["strategy"],
            "sampling": "greedy",
        }
        responses = [
            {
                "id": f"chatcmpl_gate_{ordinal + 1}",
                "usage": {
                    "prompt_tokens": case["input_tokens"],
                    "completion_tokens": case["output_tokens"],
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
            for ordinal in range(case["batch"])
        ]
        manifest = response_manifest(responses, case)
        if case["strategy"] == "dep4":
            for request in manifest["requests"]:
                request["owner_rank"] = (request["ordinal"] + 2) % 4
        calls = [
            {
                "rank": rank,
                "request_rows": [
                    {"id": request["id"]}
                    for request in manifest["requests"]
                    if request["owner_rank"] == rank
                ],
            }
            for rank in range(4)
        ]
        rows.append(
            {
                **case,
                **configuration,
                "status": "completed",
                "effective_strategy": case["strategy"],
                "request_count": case["batch"],
                "measurement_identity": copy.deepcopy(identity),
                "seconds": 100,
                "reference_seconds": 33,
                "efficiency_percent": 33,
                "reference_acceptance": {
                    packed.VERSION: "trace-project-estimate",
                    floor.VERSION: "trace-resource-floor",
                    roofline.VERSION: "trace-roofline",
                }[version],
                "responses": responses,
                "trace": {
                    "binding": binding,
                    "manifest": manifest,
                    "calls": calls,
                },
            }
        )
    return rows


class PackedHistory(unittest.TestCase):
    def test_native_rows_and_shared_storage_are_distinct_from_consumer_reads(self):
        model = replace(
            MODEL,
            num_layers=2,
            compress_ratios=(2, 2),
            kv_source_layers=(0,),
            index_source_layers=(0, 1),
        )
        layers = arithmetic.layer_table(model)
        rows = [{"start": 4, "count": 2}]
        seen = {}
        first, first_union = packed.history_work(rows, layers[0], model, seen)
        second, second_union = packed.history_work(rows, layers[1], model, seen)
        self.assertEqual(packed.packed_row_bytes(512, 16), 288)
        self.assertEqual(packed.packed_row_bytes(128, 32), 68)
        self.assertEqual((first.hbm_bytes, first_union), (5832, 4808))
        self.assertEqual((second.hbm_bytes, second_union), (5184, 4096))
        window, union = packed.history_work(
            rows, arithmetic.layer_table(MODEL)[0], MODEL, {}
        )
        self.assertEqual((window.hbm_bytes, union), (4096, 4096))
        with self.assertRaises(ValueError):
            packed.packed_row_bytes(127, 32)

    def test_direction_and_dependency_order_match_parallel_strategies(self):
        matrix = [[0] * 4 for _ in range(4)]
        matrix[0][1], matrix[1][0], matrix[0][0] = 2, 1, 100
        self.assertEqual(packed.directed_bytes(matrix, 0, 1024), (2048, 1024))
        self.assertEqual(
            packed.directed_bytes(matrix, 0, 2048, reverse=True), (2048, 4096)
        )
        self.assertEqual(packed.collective_bytes("all_gather", 4096), 3072)
        self.assertEqual(packed.collective_bytes("all_reduce", 4096), 6144)
        for strategy in ("tp4", "tep4", "dep4"):
            case = dict(
                mode="dspark5",
                strategy=strategy,
                batch=1,
                input_tokens=2,
                output_tokens=2,
            )
            trace = source.ideal_trace(case, model=MODEL)
            records = source.validate_trace(
                trace, strategy=strategy, model=MODEL
            ).calls[0]
            groups, _directions, _unions = packed.call_groups(
                records, strategy=strategy, model=MODEL
            )
            names = [group.name for group in groups[0]]
            self.assertLess(names.index("0.attention_output"), names.index("0.router"))
            if strategy == "tp4":
                self.assertLess(
                    names.index("0.expert.up_gate"),
                    names.index("0.expert.activation_gather"),
                )
                self.assertLess(
                    names.index("0.expert.activation_gather"),
                    names.index("0.expert.down"),
                )
            elif strategy == "dep4":
                self.assertLess(
                    names.index("0.dispatch.payload"), names.index("0.expert.up_gate")
                )
                self.assertLess(
                    names.index("0.expert.down"), names.index("0.expert.combine")
                )
            else:
                self.assertLess(
                    names.index("0.expert.down"), names.index("0.expert.reduce")
                )

    def test_protocol_metadata_does_not_inflate_the_project_reference(self):
        case = dict(
            mode="dspark5", strategy="dep4", batch=1, input_tokens=2, output_tokens=2
        )
        trace = source.ideal_trace(case, model=MODEL)
        first = packed.reference_trace(trace, strategy="dep4", model=MODEL)
        original = packed.call_groups

        def inflated(*args, **kwargs):
            ranks, directions, unions = original(*args, **kwargs)
            ranks = [
                [
                    replace(group, work=arithmetic.Work(nvlink_bytes=10**20))
                    if group.name.endswith(".dispatch.counts")
                    else group
                    for group in groups
                ]
                for groups in ranks
            ]
            return ranks, directions, unions

        with patch.object(packed, "call_groups", side_effect=inflated):
            changed = packed.reference_trace(trace, strategy="dep4", model=MODEL)
        self.assertEqual(changed.seconds, first.seconds)
        self.assertGreater(
            changed.protocol_serial_seconds, first.protocol_serial_seconds
        )
        self.assertEqual(
            first.conditional_partial_floor_seconds,
            floor.reference_trace(trace, strategy="dep4", model=MODEL).seconds,
        )


class WholeHttpRoofline(unittest.TestCase):
    def test_call_splitting_does_not_increase_necessary_resources(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                trace = routed_trace(strategy, outputs=7)
                split = split_decode(trace, strategy)
                first = roofline.reference_trace(trace, strategy=strategy, model=MODEL)
                second = roofline.reference_trace(split, strategy=strategy, model=MODEL)
                self.assertEqual(first.seconds, second.seconds)
                self.assertEqual(first.ranks, second.ranks)
                self.assertEqual(first.protocol_stages, second.protocol_stages)
                self.assertEqual(first.dependency_seconds, 0)
                self.assertGreater(
                    packed.reference_trace(
                        split, strategy=strategy, model=MODEL
                    ).seconds,
                    packed.reference_trace(
                        trace, strategy=strategy, model=MODEL
                    ).seconds,
                )

    def test_weight_witnesses_take_maximum_and_native_payload_has_one_source(self):
        case = dict(
            mode="dspark5",
            strategy="tp4",
            batch=4,
            input_tokens=400000,
            output_tokens=8192,
        )
        trace = source.ideal_trace(case, model=MODEL)
        result = roofline.reference_trace(trace, strategy="tp4", model=MODEL)
        rank = result.ranks[0]
        self.assertEqual(rank.mandatory_target_rounds, 1367)
        self.assertEqual(
            rank.weight_read_lower_bytes,
            max(
                0,
                rank.weight_union_lower_bytes - (1 << 30),
                1367 * max(0, rank.dense_weight_bytes - (1 << 30)),
            ),
        )
        model = replace(
            MODEL,
            num_layers=2,
            compress_ratios=(2, 2),
            kv_source_layers=(0,),
            index_source_layers=(0, 1),
        )
        payload, writes = roofline._cache_writes(
            [{"input_tokens": 20000000, "output_tokens": 1}], model
        )
        self.assertEqual(payload, 10000000 * (288 + 68))
        self.assertEqual(writes, payload - (1 << 30))

    def test_communication_directions_order_and_empty_owner_match_literal_routes(self):
        for strategy in ("tp4", "tep4", "dep4"):
            trace = routed_trace(strategy, batch=3)
            result = roofline.reference_trace(trace, strategy=strategy, model=MODEL)
            stages = {row.name: row for row in result.protocol_stages}
            rows = 3 * 21
            if strategy == "tp4":
                activation = stages["0.expert.activation_gather"]
                self.assertEqual(activation.sent_bytes, (rows * 20736,) * 4)
                self.assertEqual(
                    (activation.after, activation.before),
                    ("0.expert.up_gate", "0.expert.down"),
                )
                self.assertEqual(
                    stages["0.expert.output_gather"].sent_bytes, (rows * 15360,) * 4
                )
            elif strategy == "tep4":
                self.assertEqual(
                    stages["0.expert.reduce"].sent_bytes, (rows * 30720,) * 4
                )
            else:
                empty = result.ranks[3]
                self.assertGreater(empty.fp8_flops, 0)
                self.assertEqual(empty.mandatory_target_rounds, 0)
                self.assertEqual(empty.host_h2d_bytes, 0)
                dispatch, combine = stages["0.dispatch"], stages["0.combine"]
                self.assertEqual(sum(dispatch.sent_bytes), sum(dispatch.received_bytes))
                self.assertEqual(
                    combine.sent_bytes,
                    tuple(2 * value for value in dispatch.received_bytes),
                )
                self.assertEqual(
                    combine.received_bytes,
                    tuple(2 * value for value in dispatch.sent_bytes),
                )
                for rank, resource in enumerate(result.ranks):
                    self.assertEqual(
                        resource.wire_sent_bytes,
                        dispatch.sent_bytes[rank] + combine.sent_bytes[rank],
                    )
                    self.assertEqual(
                        resource.wire_received_bytes,
                        dispatch.received_bytes[rank] + combine.received_bytes[rank],
                    )
        model = replace(MODEL, engram_layers=(0,))
        trace = routed_trace("tep4", model=model)
        result = roofline.reference_trace(trace, strategy="tep4", model=model)
        stages = list(result.protocol_stages)
        names = [stage.name for stage in stages]
        self.assertLess(
            names.index("0.engram.reduce"), names.index("0.attention.reduce")
        )
        self.assertEqual(result.pooled_host_h2d_bytes, 24 * 264)

    def test_identity_and_measured_latency_cannot_change_without_detection(self):
        trace = routed_trace("tp4")
        first = roofline.reference_trace(trace, strategy="tp4", model=MODEL)
        original = roofline.rate_profile_digest(MODEL)
        with patch.object(roofline, "FAST_MEMORY_BYTES", 2 << 30):
            self.assertNotEqual(roofline.rate_profile_digest(MODEL), original)
        with patch.object(
            arithmetic, "collective_latency_seconds", side_effect=AssertionError
        ):
            second = roofline.reference_trace(trace, strategy="tp4", model=MODEL)
        self.assertEqual(first, second)
        changed = copy.deepcopy(trace)
        changed["binding"]["source"] = "changed"
        with self.assertRaises(source.BindingError):
            roofline.reference_trace(
                changed, strategy="tp4", model=MODEL, expected_binding=trace["binding"]
            )

    def test_v6_runtime_binding_and_harness_use_the_same_calculator(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                case = dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=1,
                    input_tokens=20,
                    output_tokens=4,
                )
                fixture = runtime_fixture(case, reference_version=roofline.VERSION)
                harness, events, responses = fixture
                trace = assemble_trace(events, responses, harness=harness, case=case)
                result, csv = run_fixture(case, *fixture, version=roofline.VERSION)
                self.assertEqual(result["status"], "completed")
                reference = roofline.reference_trace(
                    trace, strategy=strategy, expected_binding=trace["binding"]
                )
                self.assertEqual(result["reference_seconds"], reference.seconds)
                self.assertEqual(result["reference_acceptance"], "trace-roofline")
                self.assertEqual(result["accounting_version"], roofline.VERSION)
                self.assertEqual(result["source_trace_version"], source.VERSION)
                self.assertEqual(
                    result["reference_scope"], "fixed-algorithm resource roofline"
                )
                self.assertEqual(csv["accounting_version"], roofline.VERSION)


class LongHttpGate(unittest.TestCase):
    def assess(self, rows):
        acceptance = (
            rows[0]["reference_acceptance"] if rows else "trace-project-estimate"
        )
        with patch.object(
            gate,
            "runtime_reference",
            return_value=(SimpleNamespace(seconds=33, acceptance=acceptance), {}),
        ):
            return gate.evaluate(rows)

    def test_responses_are_required_and_bound_to_each_trace_request(self):
        rows = measured_rows()
        for mutation in (
            "missing",
            "empty",
            "partial",
            "wrong_prompt",
            "wrong_completion",
            "wrong_id",
            "duplicate_id",
            "cached",
            "wrong_dep_owner",
        ):
            broken = copy.deepcopy(rows)
            result = broken[0]
            if mutation == "missing":
                del result["responses"]
            elif mutation == "empty":
                result["responses"] = []
            elif mutation == "partial":
                result["responses"].pop()
            elif mutation == "wrong_prompt":
                result["responses"][0]["usage"]["prompt_tokens"] = 1
            elif mutation == "wrong_completion":
                result["responses"][0]["usage"]["completion_tokens"] = 1
            elif mutation == "wrong_id":
                result["responses"][0]["id"] = "chatcmpl_other_999999"
            elif mutation == "duplicate_id":
                result["responses"][0]["id"] = result["responses"][1]["id"]
            elif mutation == "cached":
                result["responses"][0]["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ] = 1
            else:
                result = next(row for row in broken if row["strategy"] == "dep4")
                result["trace"]["manifest"]["requests"][0]["owner_rank"] = 0
            with self.subTest(mutation=mutation):
                report = self.assess(broken)
                self.assertFalse(report["target_passed"])
                self.assertEqual(report["status"], "invalid")

    def test_real_calculator_binds_responses_and_reviewed_b200_hardware(self):
        for strategy in ("tp4", "tep4", "dep4"):
            case = dict(
                mode="dspark5",
                strategy=strategy,
                batch=4,
                input_tokens=400000,
                output_tokens=64,
            )
            harness, events, responses = runtime_fixture(
                case, reference_version=roofline.VERSION
            )
            modes = {
                name: {
                    "batch_decode": True,
                    "decode_graph": name == "tp4",
                    "decode_graph_limit": 32,
                }
                for name in ("tp4", "tep4", "dep4")
            }
            harness.update(
                binary="cpu-fixture-binary",
                harness_numeric_runtime={"python": "cpu-fixture"},
                configuration={"execution_modes": modes},
            )
            for device in harness["topology"]:
                device["name"] = "NVIDIA B200"
            for event_rows in events:
                event_rows[0]["execution"]["runtime"]["execution_modes"] = modes[
                    strategy
                ].copy()
            trace = assemble_trace(events, responses, harness=harness, case=case)
            reference, metadata = runtime_reference(
                trace,
                strategy=strategy,
                expected_binding=trace["binding"],
                version=roofline.VERSION,
            )
            seconds = reference.seconds * 2
            measured = {
                **case,
                **metadata,
                "status": "completed",
                "effective_strategy": strategy,
                "request_count": case["batch"],
                "measurement_identity": harness,
                "seconds": seconds,
                "reference_seconds": reference.seconds,
                "reference_acceptance": reference.acceptance,
                "efficiency_percent": source.efficiency_percent(
                    reference.seconds, seconds
                ),
                "trace": trace,
                "responses": responses,
            }
            gate.validate_measurement(measured)
            for mutation in ("b300", "one_b300", "missing_name"):
                broken = copy.deepcopy(measured)
                devices = broken["measurement_identity"]["topology"]
                for device in devices if mutation != "one_b300" else devices[:1]:
                    if mutation == "missing_name":
                        del device["name"]
                    else:
                        device["name"] = "NVIDIA B300"
                with (
                    self.subTest(strategy=strategy, mutation=mutation),
                    self.assertRaisesRegex(ValueError, "four reviewed B200"),
                ):
                    gate.validate_measurement(broken)
            for mutation in ("missing", "empty", "usage", "id"):
                broken = copy.deepcopy(measured)
                if mutation == "missing":
                    del broken["responses"]
                elif mutation == "empty":
                    broken["responses"] = []
                elif mutation == "usage":
                    broken["responses"][0]["usage"]["completion_tokens"] = 1
                else:
                    broken["responses"][0]["id"] = "chatcmpl_other_999999"
                with (
                    self.subTest(strategy=strategy, mutation=mutation),
                    self.assertRaises(ValueError),
                ):
                    gate.validate_measurement(broken)

    def test_complete_exact_threshold_and_one_row_below(self):
        rows = measured_rows()
        report = self.assess(rows)
        self.assertTrue(report["target_passed"])
        self.assertEqual(report["observed_long_cases"], 19)
        rows[0]["seconds"] = 101
        rows[0]["efficiency_percent"] = 100 * 33 / 101
        report = self.assess(rows)
        self.assertFalse(report["target_passed"])
        self.assertEqual(len(report["failed_threshold_cases"]), 1)

    def test_missing_duplicate_failed_and_mixed_identity_cannot_pass(self):
        rows = measured_rows()
        variants = [rows[:-1], [*rows, rows[0]]]
        failed = copy.deepcopy(rows)
        failed[0]["status"] = "failed"
        variants.append(failed)
        mixed = copy.deepcopy(rows)
        mixed[0]["measurement_identity"]["binary"] = "different"
        variants.append(mixed)
        wrong_modes = copy.deepcopy(rows)
        wrong_modes[0]["trace"]["binding"]["runtime_modes"] = {
            "batch_decode": False,
            "decode_graph": False,
        }
        variants.append(wrong_modes)
        for variant in variants:
            with self.subTest(size=len(variant)):
                self.assertFalse(self.assess(variant)["target_passed"])
        for version in (floor.VERSION, packed.VERSION):
            historical = measured_rows(version)
            for row in historical:
                for device in row["measurement_identity"]["topology"]:
                    del device["name"]
            partial = self.assess(historical)
            self.assertEqual(partial["status"], "ineligible")
            self.assertTrue(partial["reference_threshold_met"])
            self.assertFalse(partial["target_passed"])

    def test_qualification_cannot_be_supplied_by_a_result_label(self):
        rows = measured_rows()
        rows[0]["reference_scope"] = "complete physical roofline"
        self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        del rows[0]["measurement_identity"]["configuration"]["execution_modes"]
        self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        rows[0]["rate_profile"] = "old-calculator-digest"
        self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        rows[0]["assumptions_sha256"] = "changed-assumptions"
        self.assertFalse(self.assess(rows)["target_passed"])


class RuntimeModes(unittest.TestCase):
    def test_launcher_defaults_case_and_graph_limits_match_runtime_settings(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                return_value=True,
            ),
        ):
            self.assertEqual(
                execution_modes(),
                {
                    "batch_decode": False,
                    "decode_graph": False,
                    "decode_graph_limit": 32,
                },
            )
            for strategy in ("TP4", "TEP4", "DEP4"):
                for limit in (0, 32):
                    with patch.dict(
                        "os.environ",
                        {
                            "OH_MY_VLLM_DEEPSEEK_STRATEGY": strategy,
                            "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "1",
                            "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": str(limit),
                        },
                    ):
                        self.assertEqual(
                            execution_modes(),
                            {
                                "batch_decode": True,
                                "decode_graph": strategy == "TP4",
                                "decode_graph_limit": limit,
                            },
                        )
            with (
                patch.dict("os.environ", {"OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": "-1"}),
                self.assertRaises(ValueError),
            ):
                execution_modes()

    def test_runtime_identity_uses_parsed_worker_flags_after_environment_changes(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "OH_MY_VLLM_SOURCE_COMMIT": "cpu-fixture-commit",
                    "OH_MY_VLLM_DEEPSEEK_STRATEGY": "tep4",
                    "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "0",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH": "0",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": "0",
                },
                clear=True,
            ),
            patch("oh_my_vllm.models.deepseek_v41.trace.NATIVE_MODULES", ()),
            patch(
                "oh_my_vllm.models.deepseek_v41.trace.source_identity",
                return_value="cpu-fixture-source",
            ),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(uuid="GPU-cpu-fixture"),
            ),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                side_effect=AssertionError("must use parsed Worker graph flag"),
            ),
        ):
            identities = [
                runtime_identity(
                    2,
                    strategy="tp4",
                    batch_decode=True,
                    decode_graph=True,
                    decode_graph_limit=limit,
                )
                for limit in (0, 32)
            ]
        self.assertNotEqual(identities[0], identities[1])
        for identity, limit in zip(identities, (0, 32), strict=True):
            self.assertEqual(
                identity["execution_modes"],
                {
                    "batch_decode": True,
                    "decode_graph": True,
                    "decode_graph_limit": limit,
                },
            )

    def test_observed_modes_reach_trace_binding_and_changed_rank_is_rejected(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "1",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH": "1",
                },
            ),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                return_value=True,
            ),
        ):
            modes = {
                strategy: execution_modes(strategy)
                for strategy in ("tp4", "tep4", "dep4")
            }
        self.assertEqual(
            modes["tp4"],
            {"batch_decode": True, "decode_graph": True, "decode_graph_limit": 32},
        )
        self.assertEqual(
            modes["tep4"],
            {"batch_decode": True, "decode_graph": False, "decode_graph_limit": 32},
        )
        case = dict(
            mode="dspark5", strategy="tp4", batch=1, input_tokens=20, output_tokens=4
        )
        harness, events, responses = runtime_fixture(
            case, reference_version=packed.VERSION
        )
        harness["configuration"] = {"execution_modes": modes}
        for rows in events:
            rows[0]["execution"]["runtime"]["execution_modes"] = modes["tp4"].copy()
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(trace["binding"]["runtime_modes"], modes["tp4"])
        result, _metadata = runtime_reference(
            trace,
            strategy="tp4",
            expected_binding=trace["binding"],
            version=packed.VERSION,
        )
        self.assertEqual(result.version, packed.VERSION)
        events[2][0]["execution"]["runtime"]["execution_modes"]["batch_decode"] = False
        with self.assertRaisesRegex(source.BindingError, "modes"):
            assemble_trace(events, responses, harness=harness, case=case)
        events[2][0]["execution"]["runtime"]["execution_modes"] = modes["tp4"].copy()
        measured, _csv = run_fixture(
            case, harness, events, responses, version=packed.VERSION
        )
        self.assertEqual(measured["status"], "completed")
        self.assertEqual(measured["measurement_identity"], harness)


if __name__ == "__main__":
    unittest.main()
