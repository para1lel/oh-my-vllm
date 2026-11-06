import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.test_deepseek_accounting_floor import routed_trace, split_decode

from . import deepseek_acceptance as gate
from . import deepseek_accounting as source
from . import deepseek_accounting_floor as floor
from . import deepseek_accounting_math as arithmetic
from . import deepseek_accounting_packed as packed
from . import deepseek_accounting_roofline as roofline
from .deepseek_benchmark import reference_configuration
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


def _measurement_identity(version):
    configuration = reference_configuration(version)
    modes = {
        strategy: {
            "batch_decode": True,
            "decode_graph": strategy == "tp4",
            "decode_graph_limit": 32,
            "proposal_graph": False,
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
    return configuration, modes, identity


def measured_rows(version=roofline.VERSION):
    configuration, modes, identity = _measurement_identity(version)
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


def _reject_unreviewed_hardware(self, measured, strategy):
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


if __name__ == "__main__":
    unittest.main()
