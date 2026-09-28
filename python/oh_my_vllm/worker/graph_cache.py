"""Bounded CUDA graph admission and eviction for long-lived workers."""

from collections import Counter, OrderedDict
from collections.abc import Callable

import torch


class GraphCache:
    MIN_CAPTURE_FREE_BYTES = 4 << 30

    def __init__(
        self,
        capacity: int = 32,
        *,
        family_floors: dict[str, int] | None = None,
        probation_capacity: int = 256,
        recent_capacity: int = 4096,
        admission_hits: int = 4,
        admission_factor: int = 2,
        failure_cooldown: int = 64,
        churn_window_decisions: int = 4096,
        churn_cooldown_decisions: int = 0,
        synchronize: Callable[[], None] | None = None,
        free_bytes: Callable[[], int] | None = None,
        reserved_bytes: Callable[[], int] | None = None,
    ) -> None:
        floors = dict(family_floors or {})
        if (
            capacity <= 0
            or probation_capacity <= 0
            or recent_capacity <= 0
            or admission_hits <= 0
            or admission_factor < 0
            or failure_cooldown < 0
            or churn_window_decisions <= 0
            or churn_cooldown_decisions < 0
            or any(value < 0 for value in floors.values())
            or sum(floors.values()) > capacity
        ):
            raise ValueError("invalid graph cache configuration")
        self.capacity = capacity
        self.family_floors = floors
        self.probation_capacity = probation_capacity
        self.recent_capacity = recent_capacity
        self.admission_hits = admission_hits
        self.admission_factor = admission_factor
        self.failure_cooldown = failure_cooldown
        self.churn_window_decisions = churn_window_decisions
        self.churn_cooldown_decisions = churn_cooldown_decisions
        self.synchronize = synchronize or torch.cuda.synchronize
        self.free_bytes = free_bytes
        self.reserved_bytes = reserved_bytes
        self.graphs: OrderedDict[tuple[str, tuple], object] = OrderedDict()
        self.probation: OrderedDict[tuple[str, tuple], int] = OrderedDict()
        self.recent_captures: OrderedDict[tuple[str, tuple], int] = OrderedDict()
        self.frequencies: Counter[tuple[str, tuple]] = Counter()
        self.retry_after: OrderedDict[tuple[str, tuple], int] = OrderedDict()
        self.counters: Counter[str] = Counter()
        self.clock = 0
        self.churn_cooldown_until = -1

    def contains(self, family: str, key: tuple) -> bool:
        return (family, key) in self.graphs

    def has_family(self, family: str) -> bool:
        return any(name == family for name, _ in self.graphs)

    def is_full(self) -> bool:
        return len(self.graphs) >= self.capacity

    def _victim(self) -> tuple[str, tuple] | None:
        counts = Counter(name for name, _ in self.graphs)
        return min(
            (
                resident
                for resident in self.graphs
                if counts[resident[0]] > self.family_floors.get(resident[0], 0)
            ),
            key=lambda resident: self.frequencies[resident],
            default=None,
        )

    def should_use(self, family: str, key: tuple) -> bool:
        """Admit only sustained misses hotter than an evictable resident."""
        combined = family, key
        self.clock += 1
        if self.clock % 512 == 0:
            for resident in list(self.frequencies):
                self.frequencies[resident] //= 2
                if self.frequencies[resident] == 0 and resident not in self.graphs:
                    self.frequencies.pop(resident)
        if combined in self.graphs:
            self.frequencies[combined] += 1
            self.graphs.move_to_end(combined)
            self.counters[f"{family}_hit"] += 1
            return True
        if self.clock <= self.churn_cooldown_until:
            self.counters[f"{family}_churn_eager"] += 1
            return False
        self.frequencies[combined] += 1
        if self.clock < self.retry_after.get(combined, 0):
            self.counters[f"{family}_capture_failure_eager"] += 1
            return False
        self.retry_after.pop(combined, None)
        if (
            self.free_bytes is not None
            and self.free_bytes() < self.MIN_CAPTURE_FREE_BYTES
        ):
            self.counters[f"{family}_headroom_eager"] += 1
            return False
        if not self.is_full():
            return True
        seen = self.probation.get(combined, 0) + 1
        self.probation[combined] = seen
        self.probation.move_to_end(combined)
        if len(self.probation) > self.probation_capacity:
            forgotten, _ = self.probation.popitem(last=False)
            self.frequencies.pop(forgotten, None)
            self.retry_after.pop(forgotten, None)
            self.counters["probation_eviction"] += 1
        victim = self._victim()
        if (
            seen < self.admission_hits
            or victim is None
            or self.frequencies[combined]
            <= self.frequencies[victim] * self.admission_factor
        ):
            self.counters[f"{family}_budget_eager"] += 1
            return False
        return True

    def get_or_create(self, family: str, key: tuple, create: Callable[[], object]):
        """Create a probed graph, or defer after a headroom or churn change."""
        combined = family, key
        if combined in self.graphs:
            return self.graphs[combined]
        if self.clock <= self.churn_cooldown_until:
            self.counters[f"{family}_churn_eager"] += 1
            return None
        if (
            self.free_bytes is not None
            and self.free_bytes() < self.MIN_CAPTURE_FREE_BYTES
        ):
            self.counters[f"{family}_headroom_eager"] += 1
            return None
        victim = None
        if self.is_full():
            victim = self._victim()
            if victim is None:
                raise RuntimeError("graph cache has no evictable family")
            self.synchronize()
        if self.free_bytes is not None:
            free = self.free_bytes()
            if free < self.MIN_CAPTURE_FREE_BYTES:
                self.counters[f"{family}_headroom_eager"] += 1
                return None
            metric = f"{family}_min_capture_free_bytes"
            current = self.counters.get(metric)
            self.counters[metric] = free if current is None else min(current, free)
        # The old graph remains resident during capture so its pool handle
        # stays valid. If capture fails, its key, LRU order, and probation state
        # survive; the next request may retry the same capture.
        before_reserved = self.reserved_bytes() if self.reserved_bytes else None
        try:
            graph = create()
        except Exception:
            self.retry_after[combined] = self.clock + self.failure_cooldown + 1
            self.retry_after.move_to_end(combined)
            if len(self.retry_after) > self.probation_capacity:
                self.retry_after.popitem(last=False)
            self.counters[f"{family}_capture_failure"] += 1
            raise
        if before_reserved is not None:
            delta = max(0, self.reserved_bytes() - before_reserved)
            self.counters[f"{family}_max_capture_reserved_delta"] = max(
                self.counters[f"{family}_max_capture_reserved_delta"], delta
            )
        self.graphs[combined] = graph
        if victim is not None:
            # Publish the new owner before releasing the old one. The cache is
            # briefly over capacity only inside this synchronous transaction.
            self.graphs.pop(victim)
            self.counters[f"{victim[0]}_eviction"] += 1
        self.probation.pop(combined, None)
        self.retry_after.pop(combined, None)
        if victim is not None:
            self.frequencies.pop(victim, None)
        previous_capture = self.recent_captures.get(combined)
        if previous_capture is not None:
            self.counters[f"{family}_recent_recapture"] += 1
            self.recent_captures.move_to_end(combined)
        else:
            if len(self.recent_captures) >= self.recent_capacity:
                self.recent_captures.popitem(last=False)
                self.counters["capture_history_overflow"] += 1
        self.recent_captures[combined] = self.clock
        if (
            previous_capture is not None
            and self.clock - previous_capture <= self.churn_window_decisions
            and self.churn_cooldown_decisions
        ):
            self.churn_cooldown_until = self.clock + self.churn_cooldown_decisions
            self.counters["churn_cooldown_started"] += 1
        self.counters[f"{family}_capture"] += 1
        return graph

    def snapshot(self) -> dict[str, int]:
        counts = Counter(name for name, _ in self.graphs)
        result = {
            "resident": len(self.graphs),
            "probation": len(self.probation),
            "capture_history_overflow": self.counters["capture_history_overflow"],
            "probation_eviction": self.counters["probation_eviction"],
            "retry_cooldown": len(self.retry_after),
            "churn_cooldown_remaining": max(0, self.churn_cooldown_until - self.clock),
            "churn_cooldown_started": self.counters["churn_cooldown_started"],
        }
        for family in ("target", "draft", "proposal"):
            result[f"{family}_resident"] = counts[family]
            for metric in (
                "hit",
                "capture",
                "eviction",
                "budget_eager",
                "headroom_eager",
                "capture_failure_eager",
                "capture_failure",
                "recent_recapture",
                "churn_eager",
                "max_capture_reserved_delta",
                "min_capture_free_bytes",
            ):
                result[f"{family}_{metric}"] = self.counters[f"{family}_{metric}"]
        return result

    def clear(self) -> None:
        self.graphs.clear()
        self.probation.clear()
        self.recent_captures.clear()
        self.frequencies.clear()
        self.retry_after.clear()
        self.churn_cooldown_until = -1
