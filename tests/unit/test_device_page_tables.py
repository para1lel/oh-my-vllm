"""Compact page-table uploads preserve per-token rows and dynamic graph inputs."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from oh_my_vllm.worker import tensors
from oh_my_vllm.worker.mtp import MTP


def test_page_tables_upload_one_row_per_request_then_expand():
    tables = [[11, 12, 13, 99], [21, 22], [31]]
    with patch.object(tensors, "device_tensor", wraps=tensors.device_tensor) as upload:
        actual = tensors.device_page_tables(
            tables, 3, counts=[2, 1, 0], device="cpu", dtype=torch.int64
        )
    expected = torch.tensor([[11, 12, 13], [11, 12, 13], [21, 22, 0]])
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.int64
    assert upload.call_count == 2
    assert upload.call_args_list[0].args[0] == [
        [11, 12, 13],
        [21, 22, 0],
        [31, 0, 0],
    ]
    assert upload.call_args_list[1].args[0] == [0, 0, 1]


def test_page_tables_without_repeats_upload_only_compact_rows():
    for counts in (None, [1, 1]):
        with patch.object(
            tensors, "device_tensor", wraps=tensors.device_tensor
        ) as upload:
            actual = tensors.device_page_tables(
                [[4], [8, 9]], 2, counts=counts, device="cpu"
            )
        torch.testing.assert_close(
            actual, torch.tensor([[4, 0], [8, 9]], dtype=torch.int32)
        )
        assert upload.call_count == 1


def test_page_tables_reject_invalid_dimensions():
    for tables, width, counts in (
        ([], 2, None),
        ([[1]], 0, None),
        ([[1]], 2, [1, 2]),
        ([[1]], 2, [-1]),
        ([[1]], 2, [0]),
    ):
        try:
            tensors.device_page_tables(tables, width, counts=counts, device="cpu")
        except ValueError:
            continue
        raise AssertionError((tables, width, counts))


def test_all_zero_counts_rejected_before_device_upload():
    with (
        patch.object(tensors, "device_tensor") as upload,
        pytest.raises(ValueError, match="at least one token row"),
    ):
        tensors.device_page_tables([[1], [2]], 2, counts=[0, 0])
    upload.assert_not_called()


def make_mtp():
    model = SimpleNamespace(mtp=object(), embedding=torch.zeros(1))
    with patch("oh_my_vllm.worker.mtp.MTPAttention"):
        return MTP(model, 8, 262144)


def test_mtp_draft_graph_uses_extent_and_dynamic_expanded_tables():
    mtp = make_mtp()
    tables = [list(range(1, 43)), list(range(101, 143))]
    positions = [32768, 32769, 32770, 32771, 32772]
    hidden = torch.zeros(5, 5120, dtype=torch.bfloat16)
    with (
        patch.dict(os.environ, {"OH_MY_VLLM_ENFORCE_EAGER": "0"}),
        patch("oh_my_vllm.worker.decode_graph.DraftGraph") as graph_type,
    ):
        graph_type.return_value.replay.return_value = hidden
        mtp._run([7] * 5, hidden, [0, 2, 5], tables, positions)
        uploaded = graph_type.call_args.args[5]
        assert graph_type.call_args.args[6] == 36864
        assert uploaded.shape == (5, 48)
        torch.testing.assert_close(
            uploaded[:2, :42], torch.tensor(tables[0], dtype=torch.int32).expand(2, -1)
        )
        torch.testing.assert_close(
            uploaded[2:, :42], torch.tensor(tables[1], dtype=torch.int32).expand(3, -1)
        )
        assert uploaded[:, 42:].count_nonzero() == 0

        changed = [table.copy() for table in tables]
        changed[0][41] = 999
        mtp._run([7] * 5, hidden, [0, 2, 5], changed, positions)
        assert graph_type.call_count == 1
        replay_tables = graph_type.return_value.replay.call_args.args[3]
        assert replay_tables.shape == uploaded.shape
        assert replay_tables[:2, 41].tolist() == [999, 999]
        assert replay_tables[2:, 41].tolist() == [142, 142, 142]


@pytest.mark.parametrize(
    ("position", "expected_extent", "expected_width"),
    [(783, 4096, 6), (784, 4096, 6), (262143, 262144, 335)],
)
def test_mtp_draft_graph_page_boundaries_and_context_cap(
    position, expected_extent, expected_width
):
    mtp = make_mtp()
    table = list(range(1, position // 784 + 2))
    hidden = torch.zeros(1, 5120, dtype=torch.bfloat16)
    with (
        patch.dict(os.environ, {"OH_MY_VLLM_ENFORCE_EAGER": "0"}),
        patch("oh_my_vllm.worker.decode_graph.DraftGraph") as graph_type,
    ):
        graph_type.return_value.replay.return_value = hidden
        mtp._run([7], hidden, [0, 1], [table], [position])
    assert graph_type.call_args.args[6] == expected_extent
    uploaded = graph_type.call_args.args[5]
    assert uploaded.shape == (1, expected_width)
    assert uploaded[0, : len(table)].tolist() == table
    assert uploaded[0, len(table) :].count_nonzero() == 0


def test_mtp_proposal_graph_uses_extent_without_token_row_repetition():
    mtp = make_mtp()
    mtp.device = SimpleNamespace(type="cuda")
    tables = [list(range(1, 43)), list(range(101, 143))]

    def cpu_tensor(values, *, device, dtype=None):
        return torch.tensor(values, dtype=dtype)

    def cpu_tables(pages, width, *, dtype, **_):
        return tensors.device_page_tables(pages, width, device="cpu", dtype=dtype)

    with (
        patch.dict(os.environ, {"OH_MY_VLLM_ENFORCE_EAGER": "0"}),
        patch("oh_my_vllm.worker.mtp.device_tensor", side_effect=cpu_tensor),
        patch("oh_my_vllm.worker.mtp.device_page_tables", side_effect=cpu_tables),
        patch("oh_my_vllm.worker.decode_graph.ProposalGraph") as graph_type,
    ):
        graph_type.return_value.replay.return_value = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]]
        )
        result = mtp._proposal_graph(
            torch.zeros(2, 5120),
            [(1, tables[0], 32768, 262144), (2, tables[1], 32769, 262144)],
        )
    assert result == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert graph_type.call_args.args[5] == 36864
    uploaded = graph_type.call_args.args[4]
    assert uploaded.shape == (2, 48)
    assert uploaded.dtype == torch.int64
    assert uploaded[0, :42].tolist() == tables[0]
    assert uploaded[1, :42].tolist() == tables[1]
    assert uploaded[:, 42:].count_nonzero() == 0
