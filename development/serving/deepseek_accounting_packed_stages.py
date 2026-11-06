from dataclasses import replace

from . import deepseek_accounting as source
from . import deepseek_accounting_math as arithmetic


def packed_row_bytes(width, group):
    if width < 1 or width % group or group not in (16, 32):
        raise ValueError("packed row requires complete FP4 scale groups")
    return width // 2 + width // group


def history_work(rows, layer, model, seen):
    """Keep per-consumer reads separate from the shared physical-source union."""
    window = sum(min(row["start"], model.window_size - 1) for row in rows)
    windows = window * model.head_dim * 2
    if not layer.compress_ratio:
        return arithmetic.Work(hbm_bytes=windows), windows
    ratio = layer.compress_ratio
    compressed = keys = 0
    for row in rows:
        start, end = row["start"], row["start"] + row["count"]
        new = end // ratio - start // ratio
        compressed += max(0, min(model.index_topk, end // ratio) - new)
        if layer.layer_id in model.index_source_layers:
            limit = model.candidate_positions if layer.uses_candidate_mask else end
            keys += max(0, min(limit, end // ratio) - new)
    owner = max(owner for owner in model.kv_source_layers if owner <= layer.layer_id)
    packed_kv = compressed * packed_row_bytes(model.head_dim, 16)
    packed_keys = keys * packed_row_bytes(model.index_head_dim, 32)
    old_kv, old_keys = seen.get(owner, (0, 0))
    union = windows + max(0, packed_kv - old_kv) + max(0, packed_keys - old_keys)
    seen[owner] = max(old_kv, packed_kv), max(old_keys, packed_keys)
    decoded = keys * model.index_head_dim * 2
    new_decoded = max(0, keys - old_keys // packed_row_bytes(model.index_head_dim, 32))
    new_decoded *= model.index_head_dim * 2
    # Sparse attention reads encoded KV. Index scoring reads its BF16 workspace.
    traffic = windows + packed_kv + max(0, packed_keys - old_keys)
    traffic += new_decoded + decoded
    return arithmetic.Work(hbm_bytes=traffic), union


def collective_bytes(kind, size):
    if kind == "all_reduce":
        return arithmetic.all_reduce_wire_bytes(4, size)
    if kind == "all_gather":
        return arithmetic.all_gather_wire_bytes(4, size)
    raise ValueError("unknown protocol collective")


def directed_bytes(matrix, rank, row_bytes, *, reverse=False):
    sent = sum(matrix[rank][other] for other in range(4) if other != rank)
    received = sum(matrix[other][rank] for other in range(4) if other != rank)
    if reverse:
        sent, received = received, sent
    return sent * row_bytes, received * row_bytes


def _expert_projection(rows, touched, outputs, width, model, strategy):
    if not rows:
        return arithmetic.Work()
    options = dict(storage="fp4", include_weights=False)
    if strategy == "tp4":
        options.update(shard_axis=0, shards=4)
    work = arithmetic.projection_work(rows, outputs, width, **options)
    empty = arithmetic.projection_work(1, outputs, width, **options)
    first = arithmetic.projection_work(
        1, outputs, width, **{**options, "include_weights": True}
    )
    return replace(
        work, hbm_bytes=work.hbm_bytes + touched * (first.hbm_bytes - empty.hbm_bytes)
    )


def _append_index(records, counts, layer, one, model, strategy, append, collective):
    projections, scores, sizes = [], [], []
    for rank, record in enumerate(records):
        rows = record["request_rows"]
        new_keys = (
            sum(
                (row["start"] + row["count"]) // layer.compress_ratio
                - row["start"] // layer.compress_ratio
                for row in rows
            )
            if layer.owns_compressed_kv
            else 0
        )
        projections.append(
            arithmetic.index_projection_work(
                counts[rank], new_keys, one, strategy, True
            )
            if counts[rank]
            else arithmetic.Work()
        )
        pairs = sum(
            arithmetic.candidate_pairs(
                row["start"] + row["count"],
                layer.compress_ratio,
                model.candidate_positions,
            )
            - arithmetic.candidate_pairs(
                row["start"], layer.compress_ratio, model.candidate_positions
            )
            if layer.uses_candidate_mask
            else arithmetic.index_score_pairs(
                row["start"], row["count"], layer.compress_ratio
            )
            for row in rows
        )
        shards = 1 if strategy == "dep4" else 4
        scores.append(
            arithmetic.Work(
                bf16_flops=2 * pairs * model.index_heads * model.index_head_dim / shards
            )
        )
        sizes.append(pairs * 4)
    append(f"{layer.layer_id}.index.projection", projections)
    append(f"{layer.layer_id}.index.score", scores)
    if strategy != "dep4":
        collective(f"{layer.layer_id}.index.reduce", "all_reduce", sizes)


def _append_sparse(records, layer, model, strategy, append):
    sparse = []
    for record in records:
        pairs = sum(
            arithmetic.window_pairs(row["start"], row["count"], model.window_size)
            + (
                arithmetic.compressed_pairs(
                    row["start"],
                    row["count"],
                    layer.compress_ratio,
                    model.index_topk,
                )
                if layer.compress_ratio
                else 0
            )
            for row in record["request_rows"]
        )
        sparse.append(
            arithmetic.Work(
                bf16_flops=4
                * pairs
                * model.num_heads
                * model.head_dim
                / (1 if strategy == "dep4" else 4)
            )
        )
    append(f"{layer.layer_id}.sparse_attention", sparse)


def _append_experts(records, counts, layer, model, strategy, append, wire, collective):
    if strategy == "dep4":
        matrix = [record["dispatch_rows"][layer.layer_id] for record in records]
        wire(f"{layer.layer_id}.dispatch.counts", [(3 * 8, 3 * 8)] * 4)
        wire(
            f"{layer.layer_id}.dispatch.payload",
            [directed_bytes(matrix, rank, model.hidden_size * 2) for rank in range(4)],
        )
        wire(
            f"{layer.layer_id}.dispatch.routing_metadata",
            [
                directed_bytes(matrix, rank, model.experts_per_token * (4 + 4))
                for rank in range(4)
            ],
        )
    up, down = [], []
    for rank, record in enumerate(records):
        assignments = source.received_assignments(
            records, rank, layer.layer_id, strategy
        )
        touched = record["touched_experts"][layer.layer_id]
        gate = _expert_projection(
            assignments,
            touched,
            model.moe_intermediate,
            model.hidden_size,
            model,
            strategy,
        )
        up.append(arithmetic.add_work(gate, gate))
        down.append(
            _expert_projection(
                assignments,
                touched,
                model.hidden_size,
                model.moe_intermediate,
                model,
                strategy,
            )
        )
    append(f"{layer.layer_id}.expert.up_gate", up)
    if strategy == "tp4":
        collective(
            f"{layer.layer_id}.expert.activation_gather",
            "all_gather",
            [
                count * model.experts_per_token * model.moe_intermediate * 2
                for count in counts
            ],
        )
    append(f"{layer.layer_id}.expert.down", down)
    if strategy == "dep4":
        wire(
            f"{layer.layer_id}.expert.combine",
            [
                directed_bytes(matrix, rank, model.hidden_size * 4, reverse=True)
                for rank in range(4)
            ],
        )
    elif strategy == "tp4":
        collective(
            f"{layer.layer_id}.expert.output_gather",
            "all_gather",
            [count * model.hidden_size * 4 for count in counts],
        )
    else:
        collective(
            f"{layer.layer_id}.expert.reduce",
            "all_reduce",
            [count * model.hidden_size * 4 for count in counts],
        )


def _append_layer(
    records, counts, layer, model, strategy, seen, unions, append, wire, collective
):
    one = arithmetic.layer_model(model, layer)
    local = [
        dict(arithmetic.local_work(count, one, strategy, True)) if count else {}
        for count in counts
    ]
    for operation in ("hyper_connection", "attention_input", "compressor"):
        append(
            f"{layer.layer_id}.{operation}",
            [row.get(operation, arithmetic.Work()) for row in local],
        )
    histories = []
    for rank, record in enumerate(records):
        history, union = history_work(record["request_rows"], layer, model, seen[rank])
        histories.append(history)
        unions[rank] += union
    append(f"{layer.layer_id}.history.read_decode", histories)
    if layer.layer_id in model.index_source_layers:
        _append_index(records, counts, layer, one, model, strategy, append, collective)
    _append_sparse(records, layer, model, strategy, append)
    append(
        f"{layer.layer_id}.attention_output",
        [row.get("attention_output", arithmetic.Work()) for row in local],
    )
    if strategy != "dep4":
        collective(
            f"{layer.layer_id}.attention.reduce",
            "all_reduce",
            [count * model.hidden_size * 4 for count in counts],
        )
    append(
        f"{layer.layer_id}.router",
        [row.get("router", arithmetic.Work()) for row in local],
    )
    _append_experts(records, counts, layer, model, strategy, append, wire, collective)
    append(
        f"{layer.layer_id}.shared_expert",
        [row.get("shared_expert", arithmetic.Work()) for row in local],
    )
    # Keep the local Engram lookup before its dependent projections.
    append(
        f"{layer.layer_id}.engram.lookup",
        [
            arithmetic.engram_lookup_work(counts[rank], layer, model)
            for rank in range(4)
        ],
    )
    for operation in ("engram_projection", "engram_gate"):
        append(
            f"{layer.layer_id}.{operation}",
            [row.get(operation, arithmetic.Work()) for row in local],
        )


def call_groups(records, *, strategy, model=arithmetic.DEEPSEEK_V41_MODEL):
    """Return ordered stages and each rank's sent/received and physical union."""
    ranks = [[] for _ in range(4)]
    directions = [[0, 0] for _ in range(4)]
    unions = [0] * 4
    seen = [{} for _ in range(4)]
    counts = [sum(row["count"] for row in record["request_rows"]) for record in records]

    def append(name, works):
        for rank, work in enumerate(works):
            ranks[rank].append(arithmetic.Group(name, work))

    def wire(name, values):
        for rank, (sent, received) in enumerate(values):
            directions[rank][0] += sent
            directions[rank][1] += received
        append(name, [arithmetic.Work(nvlink_bytes=max(pair)) for pair in values])

    def collective(name, kind, sizes):
        wire(name, [(collective_bytes(kind, size),) * 2 for size in sizes])

    append(
        "embedding",
        [
            arithmetic.embedding_work(count, model, strategy)
            if count
            else arithmetic.Work()
            for count in counts
        ],
    )
    if strategy != "dep4":
        collective(
            "embedding.reduce",
            "all_reduce",
            [count * model.hidden_size * 2 for count in counts],
        )
    for layer in arithmetic.layer_table(model):
        _append_layer(
            records,
            counts,
            layer,
            model,
            strategy,
            seen,
            unions,
            append,
            wire,
            collective,
        )
    logits = [
        sum(row["logit_rows"] for row in record["request_rows"]) for record in records
    ]
    append(
        "head.projection",
        [
            arithmetic.output_head_work(count, model, strategy)
            if count
            else arithmetic.Work()
            for count in logits
        ],
    )
    if strategy != "dep4":
        collective(
            "head.gather",
            "all_gather",
            [count * model.vocab_size * 4 for count in logits],
        )
    return ranks, directions, unions
