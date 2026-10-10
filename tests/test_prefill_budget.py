"""Prefill storage admission follows touched rows and the attention route."""

from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.models.qwen import Batch
from oh_my_vllm.worker.prefill_budget import prefill_capture_headroom


def storage(rows, sequences=1, *, ragged=False, context=32768, features=5):
    def tensor(*shape, dtype=torch.int64):
        return torch.empty(shape, dtype=dtype, device="meta")

    model = SimpleNamespace(feature_layer_ids=tuple(range(features)))
    attention = SimpleNamespace(
        ragged=ragged,
        query_starts=tensor(sequences + 1, dtype=torch.int32),
        kv_starts=tensor(sequences + 1, dtype=torch.int32),
        kv_lengths=tensor(sequences, dtype=torch.int32),
    )
    if ragged:
        attention.key_slots = tensor(sequences * context)
        attention.value_slots = tensor(sequences * context)
    else:
        attention.subpages = tensor(
            sequences, (context + 783) // 784 * 49, dtype=torch.int32
        )
    tokens = tensor(rows)
    batch = Batch(
        positions=tensor(rows),
        fa_slots=tensor(rows),
        attention=attention,
        starts=tensor(sequences + 1),
        sequence_ids=tensor(rows),
        state_reads=tensor(sequences),
        state_writes=tensor(rows),
        final_state_writes=tensor(sequences),
        prefill_sequences=sequences,
        prefill_tokens=rows,
    )
    caches = [tensor(1, 2, 784, 4, 256, dtype=torch.bfloat16) for _ in range(16)]
    caches += [
        (
            tensor(128, 10240, 3, dtype=torch.bfloat16),
            tensor(128, 48, 128, 128, dtype=torch.bfloat16),
        )
        for _ in range(48)
    ]
    return model, caches, tokens, batch


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("rows_per_request", [624, 1023])
def test_native_queries_have_complete_storage_allowance_below_ragged_floor(
    batch_size, rows_per_request
):
    args = storage(rows_per_request * batch_size, batch_size, context=262144)
    budget = prefill_capture_headroom(*args, list(range(1, batch_size + 1)))
    assert 4 << 30 < budget < 32 << 30
    assert (
        prefill_capture_headroom(*storage(32144, batch_size, ragged=True), [1])
        >= 32 << 30
    )


def test_snapshot_budget_deduplicates_state_slots_but_includes_null_if_written():
    args = storage(624)
    initial = prefill_capture_headroom(*args, [1, 1, -1, -1])
    assert prefill_capture_headroom(*args, [1, 1, 1, -1]) == initial
    cache = args[1][16]
    one_slot = 48 * sum(state[0].numel() * state.element_size() for state in cache)
    assert prefill_capture_headroom(*args, [1, 3, -1, -1]) == initial + one_slot
    assert prefill_capture_headroom(*args, [1, 0, -1, -1]) == initial + one_slot


def test_feature_taps_and_concatenation_have_two_phase_storage():
    rows = 1414
    native = storage(rows, 4, features=0)
    dspark = storage(rows, 4, features=5)
    delta = prefill_capture_headroom(*dspark, [1]) - prefill_capture_headroom(
        *native, [1]
    )
    assert delta == 2 * (2 * rows * 5120 * 2 * 5)


def test_ragged_context_and_maximal_state_snapshots_can_increase_safety_floor():
    args = storage(32768, 4, ragged=True, context=262144)
    no_writes = prefill_capture_headroom(*args, [])
    all_writes = prefill_capture_headroom(*args, range(128))
    assert no_writes >= 32 << 30
    assert all_writes > no_writes


def test_selected_attention_metadata_and_all_fa_rows_are_counted():
    args = storage(624)
    before = prefill_capture_headroom(*args, [1])
    args[3].output_attention = SimpleNamespace(
        tables=torch.empty(4, 335, dtype=torch.int32, device="meta"),
        lengths=torch.empty(4, dtype=torch.int32, device="meta"),
    )
    after = prefill_capture_headroom(*args, [1])
    assert after - before == (4 * 335 + 4) * 4
    # FA capture saves every row, even if logical slots were duplicated.
    args[1].append(torch.empty(1, 2, 784, 4, 256, dtype=torch.bfloat16, device="meta"))
    assert prefill_capture_headroom(*args, [1]) - after == 624 * 2 * 4 * 256 * 2
