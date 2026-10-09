"""Resource lower bounds on a semantic DAG, independent of kernel boundaries.

A multiply-add counts as two floating-point operations. Independent resource
classes can overlap. Intermediate tensors can remain on chip. Measured kernel
latencies and implementation padding cannot change a semantic node's cost.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType

B200_SM_COUNT = 148
B200_MAX_SM_CLOCK_HZ = 1965e6
B200_L2_BYTES = 132644864


@dataclass(frozen=True)
class Hardware:
    """Single-device dense peaks and cache capacity, in SI units."""

    rates: dict[str, float]
    hbm_bytes_per_s: float
    cache_bytes: int = 0

    def __post_init__(self):
        object.__setattr__(self, "rates", MappingProxyType(dict(self.rates)))
        if (
            not self.rates
            or any(not math.isfinite(v) or v <= 0 for v in self.rates.values())
            or not math.isfinite(self.hbm_bytes_per_s)
            or self.hbm_bytes_per_s <= 0
            or type(self.cache_bytes) is not int
            or self.cache_bytes < 0
        ):
            raise ValueError("invalid hardware limits")


def b200(sm_count: int, max_clock_hz: float, cache_bytes: int) -> Hardware:
    """Match the 1000 W B200 datasheet, with native instruction-table rates.

    Tensor and floating-point peaks use the single-GPU non-sparse datasheet.
    SFU and integer rates use SM 10.0 native instruction results per cycle.
    General math functions must be expanded to their native operations.
    Unpublished on-chip bandwidth is relaxed to infinity.
    """
    if (
        type(sm_count) is not int
        or sm_count != B200_SM_COUNT
        or max_clock_hz != B200_MAX_SM_CLOCK_HZ
    ):
        raise ValueError("unsupported full B200 SM count or theoretical clock ceiling")
    return Hardware(
        rates={
            "tensor_fp8": 4.5e15,
            "tensor_bf16": 2.25e15,
            "simt_fp32": 75e12,
            "simt_fp64": 37e12,
            "sfu": sm_count * max_clock_hz * 16,
            "int_add": sm_count * max_clock_hz * 128,
            "int_compare": sm_count * max_clock_hz * 64,
            "int_shift": sm_count * max_clock_hz * 64,
            "convert_bf16_fp32": sm_count * max_clock_hz * 32,
            "convert_fp32_bf16": sm_count * max_clock_hz * 16,
            "shuffle": sm_count * max_clock_hz * 32,
        },
        hbm_bytes_per_s=7.7e12,
        cache_bytes=cache_bytes,
    )


def retention_bytes(device):
    """Give all physical on-chip storage ideal cross-operation persistence.

    SM100 has 256 KiB registers, 256 KiB unified L1/shared/texture, 256 KiB
    TMEM, and an 8 KiB constant-cache working set per SM. Shared-memory
    carveouts do not add capacity to the unified pool. Even TMEM is allowed
    to survive kernel boundaries here, an optimistic capacity relaxation.
    """
    if device.get("compute_capability") != [10, 0] or "B200" not in str(
        device.get("name", "")
    ):
        raise ValueError("unsupported theoretical hardware profile")
    if device.get("register_bytes_per_sm") != 256 * 1024:
        raise ValueError("register capacity differs from SM100 specification")
    if (
        device.get("sm_count") != B200_SM_COUNT
        or device.get("l2_bytes") != B200_L2_BYTES
    ):
        raise ValueError("cache or SM capacity differs from reviewed full B200 profile")
    return device["l2_bytes"] + device["sm_count"] * (3 * 256 + 8) * 1024


@dataclass(frozen=True)
class Node:
    """One semantic operation with explicit data and persistent-state inputs."""

    name: str
    parents: tuple[int, ...] = ()
    work: dict[str, float] = field(default_factory=dict)
    external_bytes: int = 0

    def __post_init__(self):
        object.__setattr__(self, "work", MappingProxyType(dict(self.work)))
        object.__setattr__(self, "parents", tuple(self.parents))
        if (
            type(self.external_bytes) is not int
            or self.external_bytes < 0
            or any(not math.isfinite(v) or v < 0 for v in self.work.values())
        ):
            raise ValueError("invalid operation work")


class Dag:
    """Topologically ordered nodes; CUDA launch order does not supply edges."""

    def __init__(self, hardware: Hardware):
        self.hardware = hardware
        self.nodes: list[Node] = []
        self.ends: list[float] = []
        self.compute_ends: list[float] = []

    def add(self, node: Node) -> int:
        index = len(self.nodes)
        if len(set(node.parents)) != len(node.parents) or any(
            parent < 0 or parent >= index for parent in node.parents
        ):
            raise ValueError("invalid or cyclic data dependency")
        unknown = node.work.keys() - self.hardware.rates.keys()
        if unknown:
            raise ValueError(f"unrecognized execution resource: {sorted(unknown)}")
        compute = max(
            (
                count / self.hardware.rates[resource]
                for resource, count in node.work.items()
            ),
            default=0.0,
        )
        memory = node.external_bytes / self.hardware.hbm_bytes_per_s
        self.nodes.append(node)
        self.ends.append(
            max((self.ends[p] for p in node.parents), default=0.0)
            + max(compute, memory)
        )
        self.compute_ends.append(
            max((self.compute_ends[p] for p in node.parents), default=0.0) + compute
        )
        return index

    def bound(self, endpoint: int, *, cache_credit_bytes: int = 0) -> float:
        """Relax at most one initial cache's bytes on the selected ancestor path.

        Removing C bytes can decrease the uncached critical path by at most
        C/BW. The compute-only path remains a bound. This permits ideal cache
        placement without assigning all of L2 independently to every node.
        Apply one credit per step, with persistent data entering that step.
        """
        if not 0 <= endpoint < len(self.nodes) or cache_credit_bytes < 0:
            raise ValueError("invalid endpoint or cache credit")
        if cache_credit_bytes > self.hardware.cache_bytes:
            raise ValueError("cache credit exceeds device capacity")
        return max(
            self.compute_ends[endpoint],
            self.ends[endpoint] - cache_credit_bytes / self.hardware.hbm_bytes_per_s,
        )

    def resource_bound(self, endpoint: int, *, read_credit_bytes: int = 0) -> float:
        """Shared-resource limit over the endpoint's complete ancestor set.

        FP8 and BF16 operations share Tensor Cores. Read bytes must be unique
        compulsory incoming buffer ranges. This API does not include final
        dirty writes; those require their own independent capacity credit.
        """
        if not 0 <= endpoint < len(self.nodes):
            raise ValueError("invalid endpoint")
        if not 0 <= read_credit_bytes <= self.hardware.cache_bytes:
            raise ValueError("cache credit exceeds device capacity")
        seen, pending = set(), [endpoint]
        counts = {}
        read_bytes = 0
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            node = self.nodes[index]
            pending.extend(node.parents)
            read_bytes += node.external_bytes
            for resource, work in node.work.items():
                counts[resource] = counts.get(resource, 0) + work
        tensor_time = sum(
            counts.get(kind, 0) / self.hardware.rates[kind]
            for kind in ("tensor_fp8", "tensor_bf16")
            if kind in self.hardware.rates
        )
        return max(
            tensor_time,
            max(
                (
                    count / self.hardware.rates[resource]
                    for resource, count in counts.items()
                    if not resource.startswith("tensor_")
                ),
                default=0,
            ),
            max(0, read_bytes - read_credit_bytes) / self.hardware.hbm_bytes_per_s,
        )

    def relaxed_path(self, endpoint: int) -> float:
        """Allow streaming within recognized vector operations on every branch.

        Whole-matrix service after all inputs arrive would invent a barrier.
        Here compute-event service is zero; full arithmetic is accounted by
        resource totals. Each incoming read gets a whole cache independently,
        an optimistic path relaxation. Shared HBM accounting has one credit
        per genuine feedback interval. Conditional reads retain token parents.
        """
        if not 0 <= endpoint < len(self.nodes):
            raise ValueError("invalid endpoint")
        ends = []
        for node in self.nodes[: endpoint + 1]:
            service = max(0, node.external_bytes - self.hardware.cache_bytes)
            ends.append(
                max((ends[p] for p in node.parents), default=0.0)
                + service / self.hardware.hbm_bytes_per_s
            )
        return ends[endpoint]
