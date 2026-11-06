from dataclasses import replace
from functools import lru_cache

from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_contract import HISTORY_CACHE_BYTES
from .deepseek_accounting_history import HistoryGroup, PooledHistoryGroup, _gather
from .deepseek_accounting_math import (
    RATES,
    WORLD_SIZE,
    Group,
    Work,
    critical_seconds,
    scale_accumulation_work,
)


def _resource_times(groups, *, pooled=False):
    groups = [g for g in groups if isinstance(g, PooledHistoryGroup) == pooled]
    work = arithmetic.add_work(
        *(arithmetic.scale_work(group.work, group.repetitions) for group in groups)
    )
    work = scale_accumulation_work(work)
    return (
        work.hbm_bytes / RATES["hbm_bytes"],
        work.fp8_flops / RATES["fp8_flops"] + work.bf16_flops / RATES["bf16_flops"],
        work.fp32_flops / RATES["fp32_flops"],
        work.nvlink_bytes / RATES["nvlink_bytes"],
        work.host_bytes / RATES["host_bytes"],
        work.collective_seconds,
    )


def _resource_vector(ranks):
    fixed = tuple(value for groups in ranks for value in _resource_times(groups))
    pooled = [0.0] * 6
    for groups in ranks:
        for index, value in enumerate(_resource_times(groups, pooled=True)):
            pooled[index] += value / WORLD_SIZE
    return (*fixed, *pooled)


def _resource_price(vector):
    fixed, pooled = vector[: WORLD_SIZE * 6], vector[WORLD_SIZE * 6 :]
    global_floor = max(
        sum(fixed[resource::6]) / WORLD_SIZE + pooled[resource] for resource in range(6)
    )
    return max(max(fixed), global_floor)


def candidate_handoff(records, model):
    # A query owner keeps the selected block IDs until a head-partitioned consumer.
    block_size = 8
    blocks = model.candidate_positions // block_size
    rows = records[0]["request_rows"]
    size = (
        sum(
            arithmetic.compressed_pairs(
                row["start"] + block_size - 1, row["count"], block_size, blocks
            )
            for row in rows
        )
        * 4
    )
    return [
        Group(
            "index.candidate_handoff",
            arithmetic.add_work(_gather("fp32", size), Work(hbm_bytes=size)),
        )
    ]


@lru_cache(maxsize=2048)
def _credit_index(groups, previous):
    credited = []
    seen = previous
    for group in groups:
        if isinstance(group, HistoryGroup):
            size = group.reusable_hbm_bytes
            credit = min(size, previous, HISTORY_CACHE_BYTES)
            work = replace(group.work, hbm_bytes=group.work.hbm_bytes - credit)
            credited.append(
                PooledHistoryGroup(group.name, work)
                if isinstance(group, PooledHistoryGroup)
                else Group(group.name, work)
            )
            seen = max(previous, size)
        else:
            credited.append(group)
    return tuple(credited), seen


def _index_step(options, previous):
    return _cached_index_step(tuple(tuple(groups) for groups in options), previous)


@lru_cache(maxsize=2048)
def _cached_index_step(options, previous):
    credited = [
        _credit_index(tuple(groups), old)
        for groups, old in zip(options, previous, strict=True)
    ]
    groups = [row[0] for row in credited]
    seconds = critical_seconds(groups).seconds
    resources = _resource_vector(groups)
    return groups, tuple(row[1] for row in credited), seconds, resources


def _index_states(layers, options, model, base_serial, base_resources):
    states = [((), base_serial, base_resources, (0,) * WORLD_SIZE)]
    for layer in layers:
        next_states = []
        for choices, serial, total, seen in states:
            if layer in model.kv_source_layers:
                seen = (0,) * WORLD_SIZE
            for choice in range(2):
                _, current, seconds, resources = _index_step(
                    options[layer][choice], seen
                )
                next_states.append(
                    (
                        (*choices, choice),
                        serial + seconds,
                        tuple(a + b for a, b in zip(total, resources, strict=True)),
                        current,
                    )
                )
        states = next_states
    return states


def choose_index_groups(common, options, records, model, strategy):
    layers = [
        layer for layer in range(model.num_layers) if layer in model.index_source_layers
    ]
    if strategy == "dep4":
        seen = (0,) * WORLD_SIZE
        for layer in layers:
            if layer in model.kv_source_layers:
                seen = (0,) * WORLD_SIZE
            groups, seen, _, _ = _index_step(options[layer][0], seen)
            for rank in range(WORLD_SIZE):
                common[rank].extend(
                    PooledHistoryGroup(f"{layer}.{g.name}", g.work)
                    if isinstance(g, PooledHistoryGroup)
                    else Group(f"{layer}.{g.name}", g.work)
                    for g in groups[rank]
                )
        return common
    base_serial = critical_seconds(common).seconds
    base_resources = _resource_vector(common)
    states = _index_states(layers, options, model, base_serial, base_resources)
    target_rows = sum(row["count"] for row in records[0]["request_rows"])
    source = (
        layers.index(model.candidate_source_layer)
        if model.candidate_source_layer in layers
        else None
    )
    handoff = candidate_handoff(records, model) if source is not None else []
    handoff_seconds = arithmetic.serial_seconds(handoff)
    handoff_resources = _resource_vector([handoff] * WORLD_SIZE)

    def price(state):
        choices, serial, resources, _ = state
        needs_handoff = (
            source is not None
            and choices[source] == 1
            and any(choice == 0 for choice in choices[source + 1 :])
        )
        if needs_handoff:
            serial += handoff_seconds
            resources = tuple(
                a + b for a, b in zip(resources, handoff_resources, strict=True)
            )
        return (
            min(serial, max(_resource_price(resources), serial / 2))
            if target_rows >= 2
            else serial
        )

    choices, _, _, _ = min(states, key=price)
    seen = (0,) * WORLD_SIZE
    for layer, choice in zip(layers, choices, strict=True):
        if layer in model.kv_source_layers:
            seen = (0,) * WORLD_SIZE
        groups, seen, _, _ = _index_step(options[layer][choice], seen)
        for rank in range(WORLD_SIZE):
            common[rank].extend(
                PooledHistoryGroup(f"{layer}.{group.name}", group.work)
                if isinstance(group, PooledHistoryGroup)
                else Group(f"{layer}.{group.name}", group.work)
                for group in groups[rank]
            )
    if (
        source is not None
        and choices[source] == 1
        and any(choice == 0 for choice in choices[source + 1 :])
    ):
        for groups in common:
            groups.extend(handoff)
    return common


def overlap_seconds(rank_groups, target_rows):
    """Two balanced microbatches: resource totals and half the dependency chain."""
    serial = critical_seconds(rank_groups).seconds
    if target_rows < 2:
        return serial
    resource = _resource_price(_resource_vector(rank_groups))
    return min(serial, max(resource, serial / 2))
