import math
from dataclasses import dataclass
from functools import lru_cache

from . import deepseek_accounting_math as arithmetic
from .deepseek_accounting_math import (
    DEEPSEEK_V41_MODEL,
    WORLD_SIZE,
    Group,
    Work,
    critical_seconds,
)


@dataclass(frozen=True, slots=True)
class HistoryGroup(Group):
    reusable_hbm_bytes: float = 0


@dataclass(frozen=True, slots=True)
class PooledHistoryGroup(HistoryGroup):
    pass


def valid_index_pairs(rows, layer, model=DEEPSEEK_V41_MODEL):
    if not layer.owns_index_keys:
        return 0
    limit = model.candidate_positions if layer.uses_candidate_mask else None
    return sum(
        arithmetic.index_score_pairs(
            row["start"], row["count"], layer.compress_ratio, candidate_limit=limit
        )
        for row in rows
    )


def historical_reads(rows, layer, model=DEEPSEEK_V41_MODEL):
    """Minimum distinct historical rows, shared by all heads and queries of a call."""
    kv = keys = 0
    for row in rows:
        start, end = row["start"], row["start"] + row["count"]
        kv += min(start, model.window_size - 1)
        if layer.compress_ratio:
            ratio = layer.compress_ratio
            new = end // ratio - start // ratio
            kv += max(0, min(model.index_topk, end // ratio) - new)
            if layer.owns_index_keys:
                cap = model.candidate_positions if layer.uses_candidate_mask else end
                keys += max(0, min(cap, end // ratio) - new)
    return kv * model.head_dim * 2, keys * model.index_head_dim * 2


@lru_cache(maxsize=8192)
def _collect(kind, dtype, size, *, output=None):
    return arithmetic.collective_work(
        kind, dtype, size, size if output is None else output
    )


def _gather(dtype, output):
    return _collect("all_gather", dtype, math.ceil(output / WORLD_SIZE), output=output)


def _choose(alternatives):
    return min(alternatives, key=lambda ranks: critical_seconds(ranks).seconds)


def _append_query_groups(rows, pairs, keys, count, base, layer, model, queries):
    query_pairs = pairs / WORLD_SIZE
    query_keys = keys / WORLD_SIZE
    query_bytes = count * model.index_heads * (model.index_head_dim + 1) * 2
    top_bytes = (
        sum(
            arithmetic.compressed_pairs(
                row["start"], row["count"], layer.compress_ratio, model.index_topk
            )
            for row in rows
        )
        * 4
    )
    queries.append(
        [
            Group("index.projection", base),
            Group("index.query_gather", _gather("bf16", query_bytes)),
            PooledHistoryGroup(
                "index.score",
                Work(
                    bf16_flops=2
                    * query_pairs
                    * model.index_heads
                    * model.index_head_dim,
                    hbm_bytes=query_keys + query_bytes + top_bytes,
                ),
                reusable_hbm_bytes=query_keys,
            ),
            Group("index.top_gather", _gather("fp32", top_bytes)),
        ]
    )


def index_alternatives(records, layer, model, strategy, weights, key_weights=None):
    """Compare head and query placement with fixed dense and expert placement."""
    heads, queries = [], []
    for record, include_weights in zip(records, weights, strict=True):
        rows = record["request_rows"]
        count = sum(row["count"] for row in rows)
        if not count or not layer.owns_index_keys:
            heads.append(
                [
                    Group(name, Work())
                    for name in (
                        "index.projection",
                        "index.score",
                        "index.communication",
                    )
                ]
            )
            queries.append(
                [
                    Group(name, Work())
                    for name in (
                        "index.projection",
                        "index.query_gather",
                        "index.score",
                        "index.top_gather",
                    )
                ]
            )
            continue
        shards = 1 if strategy == "dep4" else WORLD_SIZE
        one = arithmetic.layer_model(model, layer)
        pairs = valid_index_pairs(rows, layer, model)
        new_keys = (
            sum(
                (row["start"] + row["count"]) // layer.compress_ratio
                - row["start"] // layer.compress_ratio
                for row in rows
            )
            if layer.owns_compressed_kv
            else 0
        )
        base = arithmetic.index_projection_work(
            count,
            new_keys,
            one,
            strategy,
            include_weights,
            None if key_weights is None else key_weights[record["rank"]],
        )
        _, keys = historical_reads(rows, layer, model)
        score = Work(
            bf16_flops=2 * pairs * model.index_heads * model.index_head_dim / shards,
            hbm_bytes=keys,
        )
        comm = _collect("all_reduce", "fp32", pairs * 4) if shards > 1 else Work()
        heads.append(
            [
                Group("index.projection", base),
                HistoryGroup("index.score", score, reusable_hbm_bytes=keys),
                Group("index.communication", comm),
            ]
        )
        if shards == 1:
            queries.append(heads[-1])
            continue
        _append_query_groups(rows, pairs, keys, count, base, layer, model, queries)
    return heads, queries


def engram_alternatives(records, layer, model):
    full, sharded = [], []
    for record in records:
        count = sum(row["count"] for row in record["request_rows"])
        full.append(
            [Group("engram.lookup", arithmetic.engram_lookup_work(count, layer, model))]
        )
        sharded.append(
            [
                Group(
                    "engram.lookup",
                    arithmetic.engram_lookup_work(
                        count, layer, model, shards=WORLD_SIZE
                    ),
                ),
                Group(
                    "engram.reduce",
                    _collect(
                        "all_reduce",
                        "bf16",
                        count * model.engram_columns * model.engram_head_dim * 2,
                    )
                    if count and layer.has_engram
                    else Work(),
                ),
            ]
        )
    return full, sharded


def _route_work(records, rank, layer, model, element_bytes):
    matrix = [record["dispatch_rows"][layer] for record in records]
    wire = arithmetic.route_wire_pressure_bytes(
        matrix, rank, model.hidden_size * element_bytes
    )
    return Work(
        nvlink_bytes=wire,
        collective_seconds=(
            arithmetic.dep4_all_to_all_seconds(
                wire, "bf16" if element_bytes == 2 else "fp32"
            )
            if wire
            else 0
        ),
    )
