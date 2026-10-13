"""CPU behavior checks for weak diagnostics and actual ownership intersections."""

import gc
import json
import threading
import weakref
from types import SimpleNamespace

from oh_my_vllm.ir.executable_diagnostics import ExecutableDiagnostics, enabled


def owner():
    return lambda: None


def test_observation_does_not_keep_executables_alive():
    diagnostics = ExecutableDiagnostics()
    executable = owner()
    reference = weakref.ref(executable)
    diagnostics.register(executable, "target")
    snapshot = diagnostics.snapshot({}, {})
    assert snapshot["weak_live_not_enumerated"] == 1
    assert snapshot["created"] == 1
    assert snapshot["dead"] == 0
    # Retaining serialized diagnostics must not retain the observed function.
    json.dumps(snapshot)
    del executable
    gc.collect()
    assert reference() is None
    closed = diagnostics.snapshot({}, {})
    assert closed["dead"] == 1
    assert closed["weak_live_records"] == 0
    assert not closed["owners"]


def test_actual_tuple_lengths_deduplication_and_persistent_overlap():
    diagnostics = ExecutableDiagnostics()
    shared, local, other = owner(), owner(), owner()
    diagnostics.register(shared, "shared")
    diagnostics.register(local, "local")
    diagnostics.register(other, "other")
    graphs = {
        "target": [SimpleNamespace(execution_owners=(local,))],
        "dspark": [
            SimpleNamespace(execution_owners=(shared, shared)),
            SimpleNamespace(execution_owners=(shared,)),
        ],
    }
    units = {"backbone": SimpleNamespace(execution_owners=(shared,))}
    snapshot = diagnostics.snapshot(graphs, units)
    assert snapshot["graph_owner_union"] == 2
    assert snapshot["persistent_owner_union"] == 1
    assert snapshot["graph_persistent_overlap"] == 1
    assert snapshot["weak_live_not_enumerated"] == 1
    assert snapshot["graph_families"]["dspark"]["owner_tuple_lengths"] == {2: 1, 1: 1}
    assert snapshot["persistent_units"]["backbone"]["owner_deque_length"] == 1
    rows = {row["registrations"][0]["unit"]: row for row in snapshot["owners"]}
    assert rows["shared"]["graph_families"] == ["dspark"]
    assert rows["shared"]["persistent_units"] == ["backbone"]
    assert rows["other"]["graph_families"] == []
    assert snapshot["measurement_seconds"] >= 0
    assert snapshot["byte_scope"] == "shallow owner and closure container only"


def test_generations_and_missing_weak_coverage_stay_explicit():
    diagnostics = ExecutableDiagnostics()
    executable = owner()
    diagnostics.register(executable, "first")
    diagnostics.register(executable, "second")
    unobserved = owner()
    unsupported = object()
    diagnostics.register(unsupported, "unsupported")
    snapshot = diagnostics.snapshot(
        {"target": [SimpleNamespace(execution_owners=(executable, unobserved))]}, {}
    )
    assert snapshot["created"] == 3
    assert snapshot["weak_coverage_unavailable"] == 1
    assert snapshot["weak_live_records"] == 2
    assert snapshot["weak_live_unique"] == 1
    assert snapshot["enumerated_weak_coverage_missing"] == 1
    row = next(row for row in snapshot["owners"] if row["registrations"])
    assert row["registrations"] == [
        {"generation": 1, "unit": "first"},
        {"generation": 2, "unit": "second"},
    ]
    del executable, row
    gc.collect()
    # The serialized snapshot contains generation values, never owner roots.
    assert diagnostics.snapshot({}, {})["dead"] == 2
    replacement = owner()
    diagnostics.register(replacement, "replacement")
    assert diagnostics.snapshot({}, {})["owners"][0]["registrations"] == [
        {"generation": 4, "unit": "replacement"}
    ]


def test_diagnostics_require_explicit_option(monkeypatch):
    monkeypatch.delenv("OH_MY_VLLM_EXECUTABLE_DIAGNOSTICS", raising=False)
    assert not enabled()
    monkeypatch.setenv("OH_MY_VLLM_EXECUTABLE_DIAGNOSTICS", "1")
    assert enabled()


def test_pending_callback_is_not_counted_as_a_live_executable():
    diagnostics = ExecutableDiagnostics()
    executable = owner()
    diagnostics.register(executable, "target")
    died = threading.Event()
    reference = weakref.ref(executable, lambda _: died.set())
    holder = [executable]
    del executable
    thread = threading.Thread(target=holder.clear)
    try:
        with diagnostics.lock:
            thread.start()
            assert died.wait(timeout=5)
            assert reference() is None
            snapshot = diagnostics.snapshot({}, {})
            assert snapshot["weak_live_records"] == 0
            assert snapshot["registry_records"] == 1
            assert snapshot["pending_dead_records"] == 1
            assert snapshot["dead"] == 0
    finally:
        # The callback can finish only after this observer releases its lock.
        thread.join(timeout=5)
    assert not thread.is_alive()
    closed = diagnostics.snapshot({}, {})
    assert closed["dead"] == 1
    assert closed["pending_dead_records"] == 0


def test_observation_overflow_is_bounded_and_coverage_is_not_claimed():
    diagnostics = ExecutableDiagnostics(capacity=1)
    first, second = owner(), owner()
    diagnostics.register(first, "first")
    diagnostics.register(second, "second")
    snapshot = diagnostics.snapshot(
        {"target": [SimpleNamespace(execution_owners=(first, second))]}, {}
    )
    assert snapshot["registry_records"] == snapshot["registry_capacity"] == 1
    assert snapshot["weak_coverage_overflow"] == 1
    assert snapshot["enumerated_owner_union"] == 2
    assert snapshot["weak_live_unique"] == 1
    assert not snapshot["weak_coverage_complete"]
    del first
    gc.collect()
    diagnostics.register(second, "second")
    assert diagnostics.snapshot({}, {})["registry_records"] == 1
