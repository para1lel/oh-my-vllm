"""Supplemental formal maxima preserve all original operator acceptance gates."""

import hashlib
import json
from pathlib import Path

from development.kernels.cases import cases
from development.kernels.reference import source_hashes, verify_reference

ROOT = Path(__file__).resolve().parents[1]


def test_original_147_cases_survive_in_extended_matrix():
    manifest = verify_reference()["dspark_supplemental"]
    ids = manifest["preserved_original_case_ids"]
    assert len(ids) == 147
    assert hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest() == (
        "2f2d818f81380383832af55ce0f2d6124310727b7fc030afbb8728e9ebf30d9b"
    )
    assert set(ids) <= {case["id"] for case in cases()}


def test_new_attention_maxima_cover_all_dispatches_and_context_ceiling():
    configs = [case["configuration"] for case in cases()]
    block = [config for config in configs if config["operation"] == "dspark_attention"]
    assert {(case["batch"], case["length"]) for case in block} == {
        (batch, extent) for batch in (1, 2, 3, 4) for extent in (36864, 262144)
    }
    target = [config for config in configs if config["operation"] == "attention"]
    assert all(
        config.get("first", 1) == 0 for config in target if config["queries"] == 8
    )
    assert all(
        any(
            config["queries"] == 8
            and config["grouped"]
            and config["length"] == 262144
            and config["batch"] == batch
            for config in target
        )
        for batch in (1, 2, 3, 4)
    )
    norm = [config for config in configs if config["operation"] == "dspark_norm_rope"]
    assert any(config["tokens"] == 32768 and config["heads"] == 8 for config in norm)
    assert any(
        config["tokens"] == 28
        and config["heads"] == 32
        and config["draft"]
        and config["position_base"] + 6 == 262149
        for config in norm
    )


def test_formal_provenance_includes_new_semantics_and_pin():
    hashes = source_hashes()
    assert {
        "python/oh_my_vllm/ir/core.py",
        "python/oh_my_vllm/ir/dspark.py",
        "python/oh_my_vllm/models/dspark.py",
        "python/oh_my_vllm/kernels/dspark_tilelang_reference.py",
        "development/kernels/dspark-reference.json",
    } <= hashes.keys()
