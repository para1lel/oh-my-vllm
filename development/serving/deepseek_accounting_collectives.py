import math
from itertools import pairwise

from .deepseek_accounting_spec import (
    COLLECTIVE_LATENCY_PROFILE_US,
    RATES,
    WORLD_SIZE,
    Work,
    _checked_nonnegative_int,
    _checked_positive_int,
)


def collective_latency_seconds(kind, dtype, profile_input_bytes):
    """Return an idle, unlocked latency from the measured profile."""
    profile_input_bytes = _checked_nonnegative_int(
        profile_input_bytes, "collective profile input"
    )
    points_us = COLLECTIVE_LATENCY_PROFILE_US.get((kind, dtype))
    if points_us is None:
        raise ValueError(f"no measured latency profile for {kind}/{dtype}")
    if profile_input_bytes == 0:
        return 0.0
    points = tuple((size, latency_us * 1e-6) for size, latency_us in points_us)
    monotonic_points = []
    highest_latency = 0.0
    for size, latency in points:
        highest_latency = max(highest_latency, latency)
        monotonic_points.append((size, highest_latency))
    if profile_input_bytes <= monotonic_points[0][0]:
        return monotonic_points[0][1]
    if profile_input_bytes >= monotonic_points[-1][0]:
        return monotonic_points[-1][1]
    for left, right in pairwise(monotonic_points):
        if left[0] <= profile_input_bytes <= right[0]:
            fraction = (profile_input_bytes - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    raise ValueError(f"no latency interpolation interval for {kind}/{dtype}")


def all_reduce_wire_bytes(world_size, bytes_per_rank):
    """Ring all-reduce bytes sent per rank for N input bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    bytes_per_rank = _checked_nonnegative_int(bytes_per_rank, "bytes_per_rank")
    return 2 * (world_size - 1) * bytes_per_rank / world_size


def all_gather_wire_bytes(world_size, output_bytes_per_rank):
    """Ring all-gather bytes sent per rank for N output bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    output_bytes_per_rank = _checked_nonnegative_int(
        output_bytes_per_rank, "output_bytes_per_rank"
    )
    return (world_size - 1) * output_bytes_per_rank / world_size


def reduce_scatter_wire_bytes(world_size, output_bytes_per_rank):
    """Ring reduce-scatter bytes sent per rank for M output bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    output_bytes_per_rank = _checked_nonnegative_int(
        output_bytes_per_rank, "output_bytes_per_rank"
    )
    return (world_size - 1) * output_bytes_per_rank


def route_wire_pressure_bytes(route_rows, rank, element_bytes):
    """Return max(sent, received) wire bytes for one rank.

    Self routes use no wire. Each NVLink direction has its own 900 GB/s,
    so a sender and a receiver load different directions. The larger count
    sets the time for this rank.
    """
    if not route_rows:
        raise ValueError("route matrix must not be empty")
    world_size = len(route_rows)
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise ValueError("rank is outside the route matrix")
    element_bytes = _checked_positive_int(element_bytes, "element_bytes")
    for row in route_rows:
        if len(row) != world_size:
            raise ValueError("route matrix must be square")
        for count in row:
            _checked_nonnegative_int(count, "route count")
    sent = sum(
        route_rows[rank][destination]
        for destination in range(world_size)
        if destination != rank
    )
    received = sum(
        route_rows[source][rank] for source in range(world_size) if source != rank
    )
    return max(sent, received) * element_bytes


def dep4_all_to_all_seconds(wire_bytes, source_dtype):
    if source_dtype not in ("bf16", "fp32"):
        raise ValueError("DEP4 proxy supports BF16 and FP32 route values")
    profile_input_bytes = math.ceil(wire_bytes * WORLD_SIZE / (WORLD_SIZE - 1))
    latency = collective_latency_seconds("all_to_all", "bf16", profile_input_bytes)
    return max(latency, wire_bytes / RATES["nvlink_bytes"])


def collective_work(kind, dtype, profile_input_bytes, output_bytes_per_rank):
    if kind == "all_gather":
        wire = all_gather_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    elif kind == "all_reduce":
        wire = all_reduce_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    else:
        raise ValueError(f"unknown collective kind: {kind}")
    latency = collective_latency_seconds(kind, dtype, profile_input_bytes)
    return Work(
        nvlink_bytes=wire,
        collective_seconds=max(latency, wire / RATES["nvlink_bytes"]),
    )
