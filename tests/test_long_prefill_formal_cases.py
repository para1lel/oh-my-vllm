"""Preserved gates and static maxima for balanced long-prefill workloads."""

import hashlib
import json

from development.kernels.cases import cases, original_cases


def test_all_317_original_cases_are_unchanged_and_preserved():
    original = original_cases()
    encoded = json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    assert len(original) == 317
    assert hashlib.sha256(encoded).hexdigest() == (
        "cdf00d89c4e62e00ecb070e0b2661317f86c5cd485919dda5063dd32ce8aee6c"
    )
    observed = {row["id"]: row for row in cases()}
    for previous in original:
        row = observed[previous["id"]]
        assert row["configuration"] == previous["configuration"]
        assert set(previous["workloads"]) <= set(row["workloads"])
    assert len(observed) == 517


def test_multiple_prefill_maxima_cover_the_new_projection_and_state_paths():
    configurations = [row["configuration"] for row in cases()]
    for batch in (2, 4):
        quota = 32768 // batch // 784 * 784
        shape = (quota,) * batch
        for operation in ("convolution", "gdn_prefill"):
            assert {"operation": operation, "counts": shape} in configurations
    expected = {
        (31360, columns, width, silu)
        for columns, width, silu in (
            (2048, 5120, False),
            (16384, 5120, False),
            (14336, 5120, False),
            (5120, 6144, False),
            (5120, 17408, True),
        )
    }
    projections = {
        (c["tokens"], c["columns"], c["width"], c["silu"])
        for c in configurations
        if c["operation"] == "fp8_linear"
    }
    assert expected <= projections


def test_shifted_cold_mtp_context_keeps_the_complete_owned_chain():
    configurations = [row["configuration"] for row in cases()]
    for rows in (32143, 31358, 31356):
        for config in (
            dict(operation="norm", tokens=rows, width=5120),
            dict(operation="quant", tokens=rows, width=5120, column=False),
            dict(
                operation="fp8_linear",
                tokens=rows,
                columns=2048,
                width=5120,
                silu=False,
            ),
            dict(operation="prepare_context", tokens=rows, index_dtype="int64"),
        ):
            assert config in configurations
    assert all(
        c.get("width") != 10240
        for c in configurations
        if c["operation"] == "fp8_linear"
    )


def test_long_attention_uses_each_legal_context_and_metadata_path():
    attention = [
        row["configuration"]
        for row in cases()
        if row["configuration"]["operation"] == "attention"
        and row["configuration"].get("max_tokens") == 135168
    ]
    assert len(attention) == 20
    for batch in (1, 2, 3, 4):
        paths = {
            (
                c["queries"],
                c["grouped"],
                c["table_dtype"],
                c["length_dtype"],
                c["length"],
            )
            for c in attention
            if c["batch"] == batch
        }
        assert {
            (1, False, "int64", "int64", 135166),
            (1, False, "int32", "int64", 135168),
            (5, True, "int32", "int64", 135168),
            (5, False, "int32", "int32", 135168),
        } <= paths
        if batch in (1, 2, 4):
            assert (1, False, "int32", "int32", 131073) in paths
        if batch == 4:
            assert (1, False, "int32", "int32", 135168) in paths
    assert all(c["first"] == 1 for c in attention)


def test_last_prefill_and_clipped_decode_keep_their_distinct_implementation_paths():
    configurations = [row["configuration"] for row in cases()]
    for batch in (1, 2, 4):
        tail = 131072 % 784
        assert (
            dict(operation="norm", tokens=(tail + 1) * batch, width=5120)
            in configurations
        )
        assert dict(operation="gdn_prefill", counts=(tail,) * batch) in configurations
        assert dict(operation="qk", tokens=tail * batch) in configurations
        assert (
            dict(
                operation="add_norm_fp8_linear",
                tokens=tail * batch,
                width=5120,
                columns=34816,
            )
            in configurations
        )
        assert (
            dict(
                operation="fp8_linear",
                tokens=tail * batch,
                columns=5120,
                width=17408,
                silu=True,
            )
            in configurations
        )
    for operation in ("convolution", "gdn_prefill"):
        assert dict(operation=operation, counts=(2352,)) in configurations
    assert dict(operation="qk", tokens=2352) in configurations
    assert (
        dict(operation="add_norm_fp8_linear", tokens=2352, width=5120, columns=34816)
        in configurations
    )
    for width in (5120, 6144):
        assert (
            dict(operation="quant", tokens=3, width=width, column=True)
            in configurations
        )
    assert (
        dict(operation="silu_quant", tokens=3, width=17408, column=True)
        in configurations
    )


def test_context_restore_quantizes_six_kv_rows_without_six_attention_queries():
    matrix = cases()
    for batch in (1, 2, 4):
        config = dict(operation="quant", tokens=6 * batch, width=5120, column=True)
        row = next(r for r in matrix if r["configuration"] == config)
        assert {
            f"mtp-32768-{batch}",
            f"prefix-32768-{batch}",
            f"mtp-131072-{batch}",
        } <= set(row["workloads"])
    assert all(
        r["configuration"].get("queries", 0) <= 5
        for r in matrix
        if r["configuration"]["operation"] == "attention"
        and r["configuration"].get("first", 1) == 1
    )
