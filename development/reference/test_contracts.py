import hashlib
import json
import math
import struct
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from oh_my_vllm.kernels.native import source_digest
from oh_my_vllm.kernels.specification import MODEL
from oh_my_vllm.worker.checkpoint import Checkpoint

from .checks import CHECKS, FLAGS, verify_case
from .reference import numerical_error


def numerical_evidence(operator, variant="default", *, rows=1, mtp=False):
    metric = {
        "passed": True,
        "nrmse": 0,
        "max_abs": 0,
        "reference_peak": 1,
        "max_tolerance_ratio": 0,
        "tolerance_name": "bf16",
        "required_atol": 0,
    }
    evidence = {key: deepcopy(metric) for key in CHECKS[operator]}
    evidence.update({key: True for key in FLAGS[operator]})
    evidence.update(passed=True, acceptance_state_selection=True)
    if operator == "sample":
        flags = {
            key: True
            for key in (
                "passed",
                "probabilities_exact",
                "tokens_exact",
                "selected_probability_exact",
                "rng_unchanged",
            )
        }
        evidence["greedy"] = {
            "passed": True,
            "cases": {
                case: {mode: deepcopy(flags) for mode in ("eager", "graph")}
                for case in (
                    "penalties_and_mask",
                    "tied_logits",
                    "penalized_tie",
                    "empty_mask",
                )
            },
        }
    for key in {"cache_writes", "state_writes", "convolution_writes"} & evidence.keys():
        evidence[key] = [deepcopy(metric)]
        evidence[key][0]["tolerance_name"] = "state"
        evidence[key][0]["state_ftz_atol"] = 2.0**-126
    if operator == "fa_attention" and variant == "query_only":
        evidence["cache_writes"] = []
    if operator == "fa_attention":
        if rows > 128 and variant in {"default", "full"} and not mtp:
            evidence["output"]["tolerance_name"] = "mxfp8_fa_output"
            evidence["cache_writes"][0]["tolerance_name"] = "mxfp8_fa_state"
        elif mtp:
            evidence["output"]["tolerance_name"] = "mtp_attention"
    if operator == "gdn_attention":
        evidence["activation_quantization"].update(codes_exact=True, scales_exact=True)
        for key in ("normalization", "input_projection", "ba_projection"):
            evidence[key]["worst_chunk_nrmse"] = evidence[key].pop("nrmse")
            evidence[key]["rows_checked"] = 1
        if rows > 128:
            evidence["input_projection"]["tolerance_name"] = "mxfp8_gdn_input"
            evidence["output"]["tolerance_name"] = "mxfp8_gdn_output"
            for key in ("state_writes", "convolution_writes"):
                evidence[key][0]["tolerance_name"] = "mxfp8_gdn_state"
    if operator == "vision_block":
        evidence["output"]["tolerance_name"] = "vision"
    if operator == "ffn" and rows > 128:
        evidence["output"]["tolerance_name"] = "mxfp8_ffn"
    if operator in {"mtp_context", "mtp_draft"}:
        evidence["attention"] = numerical_evidence(
            "fa_attention",
            "kv_only" if operator == "mtp_context" else variant,
            rows=rows,
            mtp=operator == "mtp_draft",
        )
        if operator == "mtp_context":
            evidence["endpoints"] = {"passed": True, "reference": deepcopy(metric)}
        else:
            evidence["sample"] = numerical_evidence("sample", rows=rows)
            evidence["attention"]["output"]["tolerance_name"] = "mtp_attention"
            evidence["ffn"]["tolerance_name"] = "mtp_hidden"
            evidence["ffn_residual"]["tolerance_name"] = "mtp_residual"
            evidence["sample"]["logits"]["tolerance_name"] = "mtp_logits"
    return evidence


def checkpoint_fixture(root):
    config = b'{"fixture":true}'
    (root / "config.json").write_bytes(config)
    weights = {name: f"{name}.safetensors" for name in ("weight", "embedding")}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    for name, filename in weights.items():
        header = json.dumps(
            {name: {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
        ).encode()
        header += b" " * (-len(header) % 8)
        (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")
    return hashlib.sha256(config).hexdigest()


class Contracts(unittest.TestCase):
    def test_numerical_contracts_reject_incorrect_tolerances(self):
        for rows in (1, 65, 128, 129):
            check = numerical_evidence("ffn", rows=rows)
            verify_case(check, "ffn", rows=rows)
            check["output"]["tolerance_name"] = "bf16" if rows > 128 else "mxfp8_ffn"
            check["rows"] = 129 if rows <= 128 else 1
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                verify_case(check, "ffn", rows=rows)
        for operator, field in (
            ("ffn", "residual"),
            ("fa_attention", "output"),
            ("mtp_draft", "ffn"),
        ):
            check = numerical_evidence(operator, rows=129)
            check[field]["tolerance_name"] = "mxfp8_ffn"
            with (
                self.subTest(operator=operator, field=field),
                self.assertRaises(ValueError),
            ):
                verify_case(check, operator, rows=129)
        for rows in (1, 65, 128, 129, 144):
            check = numerical_evidence("gdn_attention", rows=rows)
            verify_case(check, "gdn_attention", rows=rows)
            for field, name in (
                ("input_projection", "mxfp8_gdn_input"),
                ("output", "mxfp8_gdn_output"),
                ("state_writes", "mxfp8_gdn_state"),
                ("convolution_writes", "mxfp8_gdn_state"),
            ):
                changed = deepcopy(check)
                metric = (
                    changed[field][0]
                    if isinstance(changed[field], list)
                    else changed[field]
                )
                metric["tolerance_name"] = name if rows <= 128 else "bf16"
                changed["rows"] = 129 if rows <= 128 else 1
                with (
                    self.subTest(rows=rows, field=field),
                    self.assertRaises(ValueError),
                ):
                    verify_case(changed, "gdn_attention", rows=rows)
        for operator, field in (
            ("gdn_attention", "residual"),
            ("gdn_attention", "normalization"),
            ("gdn_attention", "ba_projection"),
            ("ffn", "output"),
            ("fa_attention", "output"),
        ):
            check = numerical_evidence(operator, rows=129)
            check[field]["tolerance_name"] = "mxfp8_gdn_output"
            with (
                self.subTest(operator=operator, field=field),
                self.assertRaises(ValueError),
            ):
                verify_case(check, operator, rows=129)

        for rows in (1, 65, 128, 129, 144):
            for variant in ("default", "full", "kv_only", "query_only"):
                check = numerical_evidence("fa_attention", variant, rows=rows)
                verify_case(check, "fa_attention", rows=rows, variant=variant)
                mx = rows > 128 and variant in {"default", "full"}
                for field, name in (
                    ("output", "bf16" if mx else "mxfp8_fa_output"),
                    ("cache_writes", "state" if mx else "mxfp8_fa_state"),
                    ("residual", "mxfp8_fa_output"),
                ):
                    changed = deepcopy(check)
                    if field == "cache_writes":
                        changed[field] = [deepcopy(check["residual"])]
                        changed[field][0]["tolerance_name"] = name
                        changed[field][0]["state_ftz_atol"] = 2.0**-126
                    else:
                        changed[field]["tolerance_name"] = name
                    changed["rows"] = 129 if rows <= 128 else 1
                    with (
                        self.subTest(rows=rows, variant=variant, field=field),
                        self.assertRaises(ValueError),
                    ):
                        verify_case(changed, "fa_attention", rows=rows, variant=variant)
        for rows in (129, 144):
            check = numerical_evidence("fa_attention", rows=rows, mtp=True)
            verify_case(check, "fa_attention", rows=rows, mtp=True)
            for field, name in (
                ("output", "mxfp8_fa_output"),
                ("cache_writes", "mxfp8_fa_state"),
            ):
                changed = deepcopy(check)
                metric = (
                    changed[field][0] if field == "cache_writes" else changed[field]
                )
                metric["tolerance_name"] = name
                with self.assertRaises(ValueError):
                    verify_case(changed, "fa_attention", rows=rows, mtp=True)
            for operator in ("mtp_context", "mtp_draft"):
                check = numerical_evidence(operator, rows=rows)
                verify_case(check, operator, rows=rows)
                check["attention"]["cache_writes"][0]["tolerance_name"] = (
                    "mxfp8_fa_state"
                )
                with self.assertRaises(ValueError):
                    verify_case(check, operator, rows=rows)
        for operator, field in (
            ("ffn", "output"),
            ("gdn_attention", "output"),
            ("gdn_attention", "input_projection"),
        ):
            check = numerical_evidence(operator, rows=129)
            check[field]["tolerance_name"] = "mxfp8_fa_output"
            with self.assertRaises(ValueError):
                verify_case(check, operator, rows=129)
        check = numerical_evidence("ffn")
        check["output"]["nrmse"] = "infinity"
        with self.assertRaises(ValueError):
            verify_case(check, "ffn", rows=1)

        floor = 2.0**-126
        for operator, rows, name, write_field, guard in (
            ("gdn_attention", 1, "state", "state_writes", "unchanged_state"),
            (
                "gdn_attention",
                129,
                "mxfp8_gdn_state",
                "state_writes",
                "unchanged_state",
            ),
            ("fa_attention", 129, "mxfp8_fa_state", "cache_writes", "unchanged_cache"),
        ):
            for actual, expected in ((0.0, floor), (floor / 2, 0.0)):
                check = numerical_evidence(operator, rows=rows)
                metric = numerical_error(
                    [actual], [expected], state=True, state_tolerance=name
                )
                check[write_field] = [metric]
                self.assertTrue(metric["passed"])
                self.assertEqual(metric["nrmse"], 1 if expected else "infinity")
                check = json.loads(json.dumps(check, allow_nan=False))
                verify_case(check, operator, rows=rows)
                for field, value in (
                    ("state_ftz_atol", floor * 2),
                    ("state_ftz_atol", None),
                    ("nrmse", math.nan),
                    ("nrmse", math.inf),
                    ("nrmse", "Infinity"),
                    ("nrmse", "inf"),
                    ("nrmse", "NaN"),
                    ("max_abs", math.inf),
                ):
                    changed = deepcopy(check)
                    changed[write_field][0][field] = value
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        verify_case(changed, operator, rows=rows)
                if expected:
                    changed = deepcopy(check)
                    changed[write_field][0]["nrmse"] = "infinity"
                    with self.assertRaises(ValueError):
                        verify_case(changed, operator, rows=rows)
                check[guard] = False
                with self.assertRaises(ValueError):
                    verify_case(check, operator, rows=rows)
            check = numerical_evidence(operator, rows=rows)
            metric = numerical_error(
                [0.0],
                [math.nextafter(floor, math.inf)],
                state=True,
                state_tolerance=name,
            )
            self.assertFalse(metric["passed"])
            check[write_field] = [metric]
            with self.assertRaises(ValueError):
                verify_case(check, operator, rows=rows)
            metric.update(passed=True, max_tolerance_ratio=0)
            with self.assertRaises(ValueError):
                verify_case(check, operator, rows=rows)
            metric["nrmse"] = "infinity"
            metric["reference_peak"] = 0
            with self.assertRaises(ValueError):
                verify_case(check, operator, rows=rows)
        for actual, expected in (
            ([math.nan], [0.0]),
            ([math.inf], [0.0]),
            ([0.0], [math.inf]),
        ):
            with self.assertRaises(ValueError):
                numerical_error(actual, expected, state=True)

    def test_checkpoint_identity_tracks_loaded_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint_fixture(root)
            with patch.dict(MODEL, {"config_sha256": config}):
                loaded = Checkpoint(root)
                loaded.raw_tensor("weight")
                frozen = loaded.verify_identity()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "weight.safetensors",
                ):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    self.assertEqual(loaded.identity, frozen)
                    with (
                        self.subTest(checkpoint=name),
                        self.assertRaisesRegex(ValueError, "file bytes changed"),
                    ):
                        loaded.verify_identity(frozen)
                    path.write_bytes(original)
                    self.assertEqual(loaded.verify_identity(frozen), frozen)
                loaded.raw_tensor("embedding")
                with self.assertRaisesRegex(ValueError, "loaded checkpoint set"):
                    loaded.verify_identity(frozen)

    def test_native_identity_tracks_only_transitive_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, header, leaf, unrelated = (
                root / name for name in ("op.cu", "shared.cuh", "leaf.cuh", "other.cuh")
            )
            source.write_text('#include "shared.cuh"\n')
            header.write_text('#include "leaf.cuh"\n')
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 128\n')
            unrelated.write_text("unrelated\n")
            identity = source_digest(source)
            unrelated.write_text("changed\n")
            self.assertEqual(source_digest(source), identity)
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 256\n')
            self.assertNotEqual(source_digest(source), identity)
            leaf.unlink()
            with self.assertRaises(FileNotFoundError):
                source_digest(source)
