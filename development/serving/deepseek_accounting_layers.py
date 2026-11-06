from functools import lru_cache

from .deepseek_accounting_projection import (
    _projection_parts,
    _weight_bytes,
    projection_work,
)
from .deepseek_accounting_spec import (
    WORLD_SIZE,
    Work,
    _checked_nonnegative_int,
    add_work,
    layer_table,
)


def _tensor_shards(strategy):
    return WORLD_SIZE if strategy in ("tp4", "tep4") else 1


def _expert_shards(strategy):
    return WORLD_SIZE if strategy == "tp4" else 1


def attention_input_work(rows, model, strategy, *, include_weights=True):
    """Return wq_a, wq_b, and wkv work for the credited rows."""
    shards = _tensor_shards(strategy)
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.q_lora_rank,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.num_heads * model.head_dim,
                model.q_lora_rank,
                storage="fp8",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.head_dim,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def attention_output_work(rows, model, strategy, *, include_weights=True):
    """Return wo_a and wo_b work for the credited rows."""
    shards = _tensor_shards(strategy)
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.o_groups * model.o_lora_rank,
                model.num_heads * model.head_dim // model.o_groups,
                storage="bf16",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.hidden_size,
                model.o_groups * model.o_lora_rank,
                storage="fp8",
                shard_axis=1 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
        )
    return work


def routed_expert_work(
    assignments,
    touched_experts,
    model,
    strategy,
    *,
    include_weights=True,
):
    """Return routed FP4 expert work for the local assignments.

    TP4 keeps every assignment on each rank and shards output channels.
    TEP4 and DEP4 keep the full expert on its owner rank.
    """
    assignments = _checked_nonnegative_int(assignments, "assignments")
    touched_experts = _checked_nonnegative_int(touched_experts, "touched_experts")
    shards = _expert_shards(strategy)
    rows = max(assignments, 1)
    up = _projection_parts(
        rows,
        model.moe_intermediate,
        model.hidden_size,
        storage="fp4",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
    )
    down = _projection_parts(
        rows,
        model.hidden_size,
        model.moe_intermediate,
        storage="fp4",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
    )
    weight = touched_experts * (2 * up[0] + down[0]) if include_weights else 0
    activation = 0 if not assignments else 2 * up[1] + down[1]
    flops = 0 if not assignments else 2 * up[2] + down[2]
    return Work(hbm_bytes=weight + activation, fp8_flops=flops)


def shared_expert_work(rows, model, strategy, *, include_weights=True):
    """Return the replicated shared expert work."""
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.moe_intermediate,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.moe_intermediate,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.hidden_size,
                model.moe_intermediate,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def router_work(rows, model, *, include_weights=True):
    """Return the FP32 router projection work."""
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.routed_experts,
                model.hidden_size,
                storage="bf16",
                compute="fp32",
                include_weights=include_weights,
            ),
        )
    return work


def hyper_connection_work(rows, model, *, include_weights=True):
    """Return Hyper-Connection mixes in FP32."""
    hc_width = model.hc_mult * model.hidden_size
    flops = (
        rows
        * model.num_layers
        * (2 * 2 * model.hc_mix * hc_width + model.hc_sinkhorn_iters * model.hc_mult**3)
    )
    weights = 0
    if include_weights:
        weights = model.num_layers * 2 * _weight_bytes("fp32", model.hc_mix, hc_width)
    activations = (
        rows
        * model.num_layers
        * (2 * hc_width * 4 + 2 * model.hc_mix * 4 + 2 * model.hc_mult**2 * 4)
    )
    return Work(hbm_bytes=weights + activations, fp32_flops=flops)


def compressor_work(rows, model, *, include_weights=True):
    """Return compressor projections for the layers that own compressed KV."""
    work = Work()
    for layer in layer_table(model):
        if not layer.owns_compressed_kv:
            continue
        if layer.compress_ratio > 1:
            work = add_work(
                work,
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="fp32",
                    compute="fp32",
                    include_weights=include_weights,
                ),
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="fp32",
                    compute="fp32",
                    include_weights=include_weights,
                ),
            )
        else:
            work = add_work(
                work,
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="bf16",
                    compute="bf16",
                    include_weights=include_weights,
                ),
            )
    return work


def engram_projection_work(rows, model, *, include_weights=True):
    """Return the Engram value projection for each Engram layer."""
    work = Work()
    for _layer in layer_table(model):
        if not _layer.has_engram:
            continue
        work = add_work(
            work,
            projection_work(
                rows,
                model.hidden_size * (model.hc_mult + 1),
                model.engram_columns * model.engram_head_dim,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def engram_gate_work(rows, model):
    """Return the FP32 Engram gate on the residual stream."""
    flops = rows * len(model.engram_layers) * 6 * model.hc_mult * model.hidden_size
    return Work(fp32_flops=flops)


def output_head_work(logit_rows, model, strategy, *, include_weights=True):
    """Return the FP32 output head for the credited logit rows."""
    shards = _tensor_shards(strategy)
    return projection_work(
        logit_rows,
        model.vocab_size,
        model.hidden_size,
        storage="fp32",
        compute="fp32",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
        include_weights=include_weights,
    )


def embedding_work(rows, model, strategy):
    """Return the sharded embedding lookup work.

    The selected-row read is per call. The embedding table is not a dense
    weight read, so a later prefill chunk keeps its own row reads.
    """
    shards = _tensor_shards(strategy)
    selected = min(rows, model.vocab_size)
    local_rows = (selected + shards - 1) // shards
    weight = local_rows * model.hidden_size * 2
    activation = rows * model.hidden_size * 2
    return Work(hbm_bytes=weight + activation)


@lru_cache(maxsize=512)
def index_projection_work(
    rows, new_keys, model, strategy, include_weights, key_weights=None
):
    shards = 1 if strategy == "dep4" else WORLD_SIZE
    common = dict(
        shard_axis=0 if shards > 1 else None,
        shards=shards,
        include_weights=include_weights,
    )
    work = add_work(
        projection_work(
            rows,
            model.index_heads * model.index_head_dim,
            model.q_lora_rank,
            storage="fp8",
            **common,
        ),
        projection_work(
            rows, model.index_heads, model.hidden_size, storage="bf16", **common
        ),
    )
    if new_keys:
        work = add_work(
            work,
            projection_work(
                new_keys,
                model.index_head_dim,
                model.head_dim,
                storage="bf16",
                include_weights=include_weights if key_weights is None else key_weights,
            ),
        )
    return work


@lru_cache(maxsize=512)
def local_work(rows, model, strategy, include_weights):
    return (
        (
            "attention_input",
            attention_input_work(
                rows, model, strategy, include_weights=include_weights
            ),
        ),
        (
            "attention_output",
            attention_output_work(
                rows, model, strategy, include_weights=include_weights
            ),
        ),
        (
            "shared_expert",
            shared_expert_work(rows, model, strategy, include_weights=include_weights),
        ),
        ("router", router_work(rows, model, include_weights=include_weights)),
        (
            "hyper_connection",
            hyper_connection_work(rows, model, include_weights=include_weights),
        ),
        ("compressor", compressor_work(rows, model, include_weights=include_weights)),
        (
            "engram_projection",
            engram_projection_work(rows, model, include_weights=include_weights),
        ),
        ("engram_gate", engram_gate_work(rows, model)),
    )


def engram_lookup_work(rows, layer, model, *, shards=1):
    # Disjoint hash columns prove this floor without token/hash identities.
    unique = model.engram_columns if rows and layer.has_engram else 0
    packed = model.engram_head_dim + model.engram_scale_bytes
    return Work(
        host_bytes=unique * packed / shards,
        hbm_bytes=unique * (packed + 2 * model.engram_head_dim) / shards,
    )
