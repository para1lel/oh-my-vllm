"""Mixed FIFO steps and boundary-restored context keep their static gates."""

from development.kernels.cases import cases


def test_mixed_target_maxima_keep_decode_and_prefill_rows_together():
    configs = [r["configuration"] for r in cases()]
    for decode in (1, 5, 8):
        rows = 624 + 31360 + decode
        assert (
            dict(
                operation="add_norm_fp8_linear", tokens=rows, width=5120, columns=34816
            )
            in configs
        )
        assert (
            dict(operation="gated_norm_fp8_linear", tokens=rows, columns=5120)
            in configs
        )
        assert (
            dict(operation="convolution", counts=(624, 784, decode, decode)) in configs
        )
    assert dict(operation="gdn_prefill", counts=(624, 784)) in configs
    assert dict(operation="qk", tokens=1408) in configs


def test_seeded_prefix_context_keeps_its_boundary_row_and_complete_chain():
    configs = [r["configuration"] for r in cases()]
    for batch in (1, 2, 4):
        rows = 625 * batch
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
            assert config in configs
    for rows in (630, 1419):
        assert dict(operation="norm", tokens=rows, width=5120) in configs


def test_dspark_context_preserves_its_nonstreaming_hidden_norm_path():
    configs = [r["configuration"] for r in cases()]
    for rows in (624, 632, 1424):
        assert dict(operation="dspark_rms_norm", tokens=rows) in configs
