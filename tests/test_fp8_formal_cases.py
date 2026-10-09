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
            (16384, 5120, False),
            (14336, 5120, False),
            (5120, 6144, False),
            (5120, 17408, True),
        )
    }
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
