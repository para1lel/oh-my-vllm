from .deepseek_accounting_spec import (
    Work,
    _checked_nonnegative_int,
    _checked_positive_int,
)


def _local_shape(out_features, in_features, shard_axis, shards):
    if type(shards) is not int or shards < 1:
        raise ValueError("shards must be a positive integer")
    if shards == 1:
        if shard_axis is not None:
            raise ValueError("shard axis requires more than one shard")
        return out_features, in_features
    if shard_axis not in (0, 1):
        raise ValueError("shard axis must be zero or one")
    axis_dimension = out_features if shard_axis == 0 else in_features
    if axis_dimension % shards:
        raise ValueError("shards must divide the sharded dimension")
    return (
        out_features // shards if shard_axis == 0 else out_features,
        in_features // shards if shard_axis == 1 else in_features,
    )


def _weight_bytes(storage, out_features, in_features):
    if storage == "fp8":
        blocks = ((out_features + 31) // 32) * ((in_features + 31) // 32)
        return out_features * in_features + blocks
    if storage == "fp4":
        if in_features % 32:
            raise ValueError("FP4 storage requires K aligned to 32")
        return out_features * in_features // 2 + out_features * (in_features // 32)
    if storage == "bf16":
        return 2 * out_features * in_features
    if storage == "fp32":
        return 4 * out_features * in_features
    raise ValueError(f"unknown storage format: {storage}")


def _activation_element_bytes(compute):
    if compute in ("fp8", "fp4"):
        return 1, 2
    if compute == "bf16":
        return 2, 2
    if compute == "fp32":
        return 4, 4
    raise ValueError(f"unknown compute format: {compute}")


def _projection_parts(
    rows,
    out_features,
    in_features,
    *,
    storage,
    compute=None,
    shard_axis=None,
    shards=1,
    include_weights=True,
):
    compute = compute or ("fp8" if storage == "fp4" else storage)
    rows = _checked_positive_int(rows, "rows")
    out_features = _checked_positive_int(out_features, "out_features")
    in_features = _checked_positive_int(in_features, "in_features")
    local_out, local_in = _local_shape(out_features, in_features, shard_axis, shards)
    weight = _weight_bytes(storage, local_out, local_in) if include_weights else 0
    input_bytes, output_bytes = _activation_element_bytes(compute)
    activation = rows * local_in * input_bytes + rows * local_out * output_bytes
    flops = 2 * rows * local_out * local_in
    return weight, activation, flops


def projection_work(
    rows,
    out_features,
    in_features,
    *,
    storage,
    compute=None,
    shard_axis=None,
    shards=1,
    include_weights=True,
):
    """Return weight, activation, and useful MMA work for one projection pass.

    FP4 storage uses an FP8 MMA, because the model contract changes FP4
    weights to FP8 before the dot product. Use `compute` to price a BF16
    weight that the model promotes to FP32.
    """
    compute = compute or ("fp8" if storage == "fp4" else storage)
    weight, activation, flops = _projection_parts(
        rows,
        out_features,
        in_features,
        storage=storage,
        compute=compute,
        shard_axis=shard_axis,
        shards=shards,
        include_weights=include_weights,
    )
    if compute not in ("fp8", "bf16", "fp32"):
        raise ValueError(f"unknown compute format: {compute}")
    return Work(
        hbm_bytes=weight + activation,
        **{f"{compute}_flops": flops},
    )


def _prefix_window(length, window):
    if length <= 0:
        return 0
    if length <= window:
        return length * (length + 1) // 2
    return window * (window + 1) // 2 + (length - window) * window


def window_pairs(start, rows, window=128):
    """Return sum of min(window, p+1) for p in [start, start+rows)."""
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    window = _checked_positive_int(window, "window")
    return _prefix_window(start + rows, window) - _prefix_window(start, window)


def _prefix_floor(ratio, limit):
    if limit <= 0:
        return 0
    quotient, remainder = divmod(limit, ratio)
    return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _prefix_capped(ratio, cap, limit):
    if limit <= 0:
        return 0
    cutoff = cap * ratio
    if limit <= cutoff:
        return _prefix_floor(ratio, limit)
    return _prefix_floor(ratio, cutoff) + (limit - cutoff) * cap


def compressed_pairs(start, rows, ratio, topk=512):
    """Return attention top-k pairs. This count is not the indexer count."""
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    ratio = _checked_positive_int(ratio, "ratio")
    topk = _checked_positive_int(topk, "topk")
    return _prefix_capped(ratio, topk, start + rows) - _prefix_capped(
        ratio, topk, start
    )


def index_score_pairs(start, rows, ratio, *, candidate_limit=None):
    """Return compressed positions an index layer scores.

    Without a candidate limit the layer scores all reachable positions.
    With a limit, the layer scores at most that many candidates per query.
    """
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    ratio = _checked_positive_int(ratio, "ratio")
    if candidate_limit is None:
        return _prefix_floor(ratio, start + rows) - _prefix_floor(ratio, start)
    candidate_limit = _checked_positive_int(candidate_limit, "candidate_limit")
    return _prefix_capped(ratio, candidate_limit, start + rows) - _prefix_capped(
        ratio, candidate_limit, start
    )


def candidate_pairs(length, ratio, limit):
    """Count valid keys when the latest eight-position block must be selected."""
    block = 8
    if limit < block or limit % block:
        raise ValueError("candidate limit must contain complete blocks")
    groups, remainder = divmod(length, ratio)
    if not groups:
        return 0

    def prefix(count):
        if count <= limit:
            return count * (count + 1) // 2
        tail = count - limit
        cycles, partial = divmod(tail, block)
        return (
            limit * (limit + 1) // 2
            + tail * (limit - block)
            + cycles * block * (block + 1) // 2
            + partial * (partial + 1) // 2
        )

    last = min(groups, limit - block + (groups - 1) % block + 1)
    return ratio * prefix(groups - 1) + (remainder + 1) * last
