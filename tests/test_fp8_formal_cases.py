"""The extended matrix keeps old gates and covers each owned FP8 chain."""

import json
import os
import subprocess
import sys
from pathlib import Path

from development.kernels.cases import cases


def test_all_existing_230_case_ids_are_kept():
    path = (
        Path(__file__).resolve().parents[1]
        / "bench/evidence/2026-10-09-dspark-operators.json"
    )
    previous = json.loads(path.read_text())["data"]["rows"]
    ids = {case["id"] for case in cases()}
    assert len(previous) == 230
    assert {case["id"] for case in previous} <= ids


def test_owned_fp8_matrix_covers_all_distinct_projection_chains():
    shapes = [624, 1248, 2496, 32144, 32288, 32290, 32768]
    expected = {
        (rows, columns, width, silu)
        for rows in shapes
        for columns, width, silu in (
            (2048, 5120, False),
            (16384, 5120, False),
            (14336, 5120, False),
            (5120, 6144, False),
            (5120, 17408, True),
        )
    }
    expected.add((32143, 2048, 5120, False))
    observed = {
        (config["tokens"], config["columns"], config["width"], config["silu"])
        for case in cases()
        if (config := case["configuration"])["operation"] == "fp8_linear"
    }
    assert observed == expected
    fused = [
        case["configuration"]
        for case in cases()
        if case["configuration"]["operation"] == "add_norm_fp8_linear"
    ]
    assert len(fused) == 16
    assert {c["tokens"] for c in fused if c["tokens"] > 32} == set(shapes)


def test_full_gdn_prefill_covers_only_prefill_rows():
    configurations = {
        tuple(case["configuration"]["counts"])
        for case in cases()
        if case["configuration"]["operation"] == "gdn_prefill"
    }
    assert configurations == {
        (624,),
        (624, 624),
        (624, 624, 624, 624),
        (32144,),
        (144, 32144),
        (624, 32144),
    }
    assert len(cases()) == 317


def test_gated_projection_covers_all_model_rows_and_keeps_previous_case_count():
    expected = {
        1,
        2,
        4,
        5,
        8,
        10,
        16,
        20,
        32,
        624,
        1248,
        2496,
        32144,
        32288,
        32290,
        32768,
    }
    observed = {
        case["configuration"]["tokens"]
        for case in cases()
        if case["configuration"]["operation"] == "gated_norm_fp8_linear"
    }
    assert observed == expected
    assert len(cases()) - len(observed) == 301


def test_partial_preparation_includes_cold_mtp_and_live_verification_maxima():
    context = {
        case["configuration"]["tokens"]
        for case in cases()
        if case["configuration"]["operation"] == "prepare_context"
    }
    query = {
        case["configuration"]["tokens"]
        for case in cases()
        if case["configuration"]["operation"] == "prepare_query"
    }
    assert context == {624, 1248, 2496, 32143, 32144, 32288, 32290, 32768}
    assert query == {1, 2, 4, 9, 25}


def test_case_filter_selects_exact_ids_and_rejects_typos():
    root = Path(__file__).resolve().parents[1]
    chosen = next(c for c in cases() if c["configuration"]["operation"] == "fp8_linear")
    command = [
        sys.executable,
        str(root / "benchmarks/kernels.py"),
        "--list",
        "--case-id",
    ]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [*command, chosen["id"]],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["cases"] == [chosen]
    rejected = subprocess.run(
        [*command, "unknown-case"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "unknown case IDs" in rejected.stderr
