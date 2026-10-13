"""Optional weak executable identities; no change to compiler ownership."""

import os
import sys
import threading
import time
import weakref
from collections import Counter
from types import FunctionType


def enabled():
    return os.environ.get("OH_MY_VLLM_EXECUTABLE_DIAGNOSTICS") == "1"


class ExecutableDiagnostics:
    """Keep generation identities only while a weakly observed owner lives."""

    def __init__(self, capacity=8192):
        if capacity <= 0:
            raise ValueError("executable observation capacity must be positive")
        self.lock = threading.RLock()
        self.capacity = capacity
        self.records = {}
        self.created = 0
        self.dead = 0
        self.unavailable = 0
        self.overflow = 0

    def register(self, owner, unit):
        with self.lock:
            self.created += 1
            generation = self.created
            if len(self.records) >= self.capacity:
                self.overflow += 1
                return

            def released(reference):
                with self.lock:
                    self.records.pop(generation, None)
                    self.dead += 1

            try:
                reference = weakref.ref(owner, released)
            except TypeError:
                self.unavailable += 1
                return
            self.records[generation] = reference, unit

    def snapshot(self, graph_families, units):
        """Return scalar summaries at the existing synchronized diagnostics boundary.

        Temporary references end with this call. Shallow sizes exclude closure
        contents, GraphModule, ShapeEnv, native payloads and allocator retention.
        """
        started = time.perf_counter()
        owners = {}
        families = {}
        persistent = {}

        def collect(values):
            identities = set()
            for owner in values:
                identity = id(owner)
                owners[identity] = owner
                identities.add(identity)
            return identities

        for name, graphs in graph_families.items():
            lengths = Counter()
            identities = set()
            for graph in graphs:
                values = getattr(graph, "execution_owners", ())
                lengths[len(values)] += 1
                identities.update(collect(values))
            families[name] = lengths, identities
        for name, unit in units.items():
            values = getattr(unit, "execution_owners", ())
            persistent[name] = len(values), collect(values)
        graph_ids = set().union(*(ids for _, ids in families.values()))
        unit_ids = set().union(*(ids for _, ids in persistent.values()))
        enumerated_ids = graph_ids | unit_ids
        live = {}
        with self.lock:
            pending_dead = 0
            for generation, (reference, unit) in self.records.items():
                owner = reference()
                if owner is not None:
                    identity = id(owner)
                    owners[identity] = owner
                    live.setdefault(identity, []).append((generation, unit))
                else:
                    pending_dead += 1
            counts = dict(
                created=self.created,
                dead=self.dead,
                registry_records=len(self.records),
                registry_capacity=self.capacity,
                weak_live_records=sum(len(values) for values in live.values()),
                pending_dead_records=pending_dead,
                weak_coverage_unavailable=self.unavailable,
                weak_coverage_overflow=self.overflow,
            )
        rows = []
        for identity, owner in owners.items():
            registrations = live.get(identity, [])
            closure = owner.__closure__ if isinstance(owner, FunctionType) else None
            rows.append(
                dict(
                    registrations=[
                        {"generation": generation, "unit": unit}
                        for generation, unit in registrations
                    ],
                    type=f"{type(owner).__module__}.{type(owner).__qualname__}",
                    shallow_owner_bytes=sys.getsizeof(owner),
                    shallow_closure_container_bytes=sys.getsizeof(closure)
                    if closure is not None
                    else 0,
                    graph_families=[
                        name for name, (_, ids) in families.items() if identity in ids
                    ],
                    persistent_units=[
                        name for name, (_, ids) in persistent.items() if identity in ids
                    ],
                )
            )
        result = {
            **counts,
            "weak_live_unique": len(live),
            "live_records_by_unit": dict(
                Counter(unit for records in live.values() for _, unit in records)
            ),
            "graph_owner_union": len(graph_ids),
            "persistent_owner_union": len(unit_ids),
            "graph_persistent_overlap": len(graph_ids & unit_ids),
            "enumerated_owner_union": len(enumerated_ids),
            "enumerated_weak_coverage_missing": len(enumerated_ids - live.keys()),
            "weak_coverage_complete": not (
                counts["weak_coverage_unavailable"]
                or counts["weak_coverage_overflow"]
                or enumerated_ids - live.keys()
            ),
            "weak_live_not_enumerated": len(live.keys() - enumerated_ids),
            "graph_families": {
                name: {
                    "graphs": sum(lengths.values()),
                    "owner_tuple_lengths": dict(lengths),
                    "owner_union": len(identities),
                }
                for name, (lengths, identities) in families.items()
            },
            "persistent_units": {
                name: {"owner_deque_length": length, "owner_union": len(identities)}
                for name, (length, identities) in persistent.items()
            },
            "owners": rows,
            "byte_scope": "shallow owner and closure container only",
            "measurement_seconds": time.perf_counter() - started,
        }
        return result


EXECUTABLE_DIAGNOSTICS = ExecutableDiagnostics()
