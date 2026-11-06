from itertools import accumulate, pairwise

from ...operators.fa_fp8.plan import make_plan as fa_plan
from ...operators.ffn.plan import make_plan as ffn_plan
from ...operators.gdn_radix.api import Schedule
from ...operators.gdn_radix.plan import chunk_workspace
from ...operators.gdn_radix.plan import make_plan as gdn_plan
from ...operators.mtp_fp8.plan import accept_plan, context_plan, draft_plan
from ...operators.output.plan import make_plan as output_plan
from ...operators.runtime.storage import storage_bytes
from ...operators.vision.plan import make_plan as vision_plan
from .cache_budget import target_token_limit

MAX_RETAINED_WORKSPACES = 8


def schedule(lengths, verify):
    starts = tuple(accumulate((0, *lengths)))
    rows, batch = starts[-1], len(lengths)
    checkpoints = ()
    if not verify:
        checkpoints = tuple(
            row
            for first, last in pairwise(starts)
            for row in range(first, last)
            if (row - first + 1) % 8192 == 0 or row + 1 == last
        )
    return Schedule(
        starts,
        (0,) * batch,
        tuple(range(1, rows + 1)) if verify else (-1,) * rows,
        checkpoints,
        tuple(range(1, len(checkpoints) + 1)),
        (0,) * batch,
        65536,
        verify,
    )


def table_width(config):
    return (config["max_model_len"] + 127) // 128


def target_storage(
    rows, batch, maximum, mtp, recurrent, context=262144, minimum=0, width=2048
):
    attention = fa_plan(
        rows=rows,
        batch=batch,
        table_width=width,
        max_queries=maximum,
        max_context=context,
        min_context=minimum,
    )
    fields = (
        attention.storage
        + recurrent.storage
        + ffn_plan(rows=rows).storage
        + (((rows, 5120), "bfloat16"),) * 2
        + output_plan(rows=rows, hidden_only=True).storage
    )
    fields += (
        ((3, rows), "int64"),
        ((rows,), "int32"),
        ((batch, width), "int32"),
        ((batch + 1,), "int32"),
        ((rows,), "int32"),
        ((rows,), "int64"),
        ((batch + 1,), "int32"),
        ((batch,), "int32"),
        ((rows,), "int32"),
        ((rows,), "uint8"),
    )
    return attention, fields


def context_storage(rows, batch, maximum, parallel, config):
    plan = context_plan(
        rows=rows,
        batch=batch,
        max_queries=maximum,
        table_width=table_width(config),
        max_context=config["max_model_len"],
    )
    fields = (*plan.storage, *plan.attention.storage, *plan.metadata_storage)
    if parallel:
        fields += (((parallel * 5, 5120), "bfloat16"),)
    return plan, fields


def prefill_lengths(rows, batch, context):
    lengths = []
    for remaining in range(min(rows, batch) - 1, -1, -1):
        count = min(context, rows - remaining)
        lengths.append(count)
        rows -= count
    return tuple(lengths)


def prefill_bytes(config):
    mtp = bool(config["num_speculative_tokens"])
    batch = min(config["max_num_seqs"], 4 if mtp else 20)
    limit = min(target_token_limit(config), config["max_model_len"] * batch)
    batch = min(batch, limit)
    maximum = 0
    candidates = {min(limit, value) for value in (*range(1, 129), 4096, 4097, limit)}
    for rows in sorted(candidates):
        lengths = prefill_lengths(rows, batch, config["max_model_len"])
        recurrent = gdn_plan(schedule(lengths, False))
        _, fields = target_storage(
            rows,
            len(lengths),
            max(lengths),
            mtp,
            recurrent,
            config["max_model_len"],
            width=table_width(config),
        )
        if recurrent.chunks:
            chunks = (rows + 63) // 64 + (rows + 8191) // 8192 + 3 * batch - 2
            fields += chunk_workspace(max(0, chunks - recurrent.chunks), batch)
        fields += (((batch * table_width(config) + batch * 3,), "int32"),)
        maximum = max(maximum, storage_bytes(fields))
    if mtp:
        context_limit = min(32768, limit + batch, config["max_model_len"] * batch)
        context_maximum = 0
        for rows in {
            min(context_limit, count) for count in (*range(1, 33), context_limit)
        }:
            _, fields = context_storage(
                rows, min(batch, rows), min(rows, config["max_model_len"]), 0, config
            )
            context_maximum = max(context_maximum, storage_bytes(fields))
        maximum += context_maximum + storage_bytes((((limit, 5120), "bfloat16"),))
    return maximum


def decode_shapes(batch, maximum):
    for queries in range(1, maximum + 1):
        for rows in range(queries + batch - 1, batch * queries + 1):
            remaining = rows - queries
            lengths = [queries]
            for sequences in range(batch - 1, 0, -1):
                count = min(queries, remaining - sequences + 1)
                lengths.append(count)
                remaining -= count
            yield rows, queries, gdn_plan(schedule(tuple(lengths), True))


def sampler_bytes(rows, *, output=True, history=True, hidden=True, native_sort=None):
    fields = output_plan(rows=rows).storage if output else ()
    if hidden:
        fields += (((rows, 5120), "bfloat16"), ((rows, 5120), "float32"))
    fields += (((rows,), "int32"),)
    if history:
        fields += (
            ((rows, 2, 248320), "int32"),
            ((rows, 248320 // 8), "uint8"),
            ((rows, 248320 // 8), "uint8"),
            ((rows, 7), "float32"),
            ((rows, 2), "uint64"),
        )
    if history:
        common = (
            ((rows, 248320 // 8), "uint8"),
            ((rows, 7), "float32"),
            ((rows, 2), "uint64"),
        )
        fields += (*common, ((rows, 248320 // 8), "uint8")) * 4
        fields += (*common, ((rows, 2, 248320), "int32")) * 2
    if output and native_sort is not None:
        fields += (((max(1, native_sort[rows]),), "uint8"),)
    return storage_bytes(fields)


def draft_bytes(batch, first, config, *, history=True, native_sort=None):
    plan = draft_plan(
        rows=batch,
        first=first,
        table_width=table_width(config),
        max_context=config["max_model_len"],
    )
    fields = (
        plan.storage
        + plan.attention.storage
        + plan.ffn.storage
        + plan.output.storage
        + (((batch, 5120), "bfloat16"),) * 2
        + (((batch,), "int32"),) * 2
        + (
            ((3, batch), "int64"),
            ((batch,), "int32"),
            ((batch, table_width(config)), "int32"),
            ((batch + 1,), "int32"),
            ((batch,), "int32"),
            ((batch,), "int64"),
        )
    )
    if native_sort is not None:
        fields += (((max(1, native_sort[batch]),), "uint8"),)
    return storage_bytes(fields) + sampler_bytes(
        batch, output=False, history=history, hidden=False
    )


def retained_targets(config):
    mtp = bool(config["num_speculative_tokens"])
    parallel = min(config["max_num_seqs"], 4 if mtp else 20)
    queries = 5 if mtp else 1
    targets, contexts = [], []
    for batch in range(1, parallel + 1):
        for rows, maximum, recurrent in decode_shapes(batch, queries):
            for context, minimum in (
                (maximum, 0),
                (config["max_model_len"], 0),
                (config["max_model_len"], config["max_model_len"]),
            ):
                _, fields = target_storage(
                    rows,
                    batch,
                    maximum,
                    mtp,
                    recurrent,
                    context,
                    minimum,
                    table_width(config),
                )
                targets.append(storage_bytes(fields))
            if mtp:
                _, fields = context_storage(
                    rows, batch, maximum, config["max_num_seqs"], config
                )
                contexts.append(storage_bytes(fields))
    return MAX_RETAINED_WORKSPACES * (
        max(targets, default=0) + max(contexts, default=0)
    )


def retained_bytes(config, native_sort=None):
    mtp = bool(config["num_speculative_tokens"])
    parallel = min(config["max_num_seqs"], 4 if mtp else 20)
    queries = 5 if mtp else 1
    total = retained_targets(config)
    total += sum(
        sampler_bytes(rows, native_sort=native_sort)
        for rows in range(1, parallel * queries + 1)
    )
    if not mtp:
        return total + 256
    for batch in range(1, parallel + 1):
        total += sum(
            draft_bytes(batch, first, config, native_sort=native_sort)
            for first in (False, True)
        )
        total += 2 * sum(
            draft_bytes(
                batch, stage == 0, config, history=True, native_sort=native_sort
            )
            for stage in range(4)
        )
        if batch >= 2:
            total += sum(
                draft_bytes(batch, stage == 0, config, native_sort=native_sort)
                for stage in range(4)
            )
            total += (
                storage_bytes(
                    (((batch * 2, 1024), "uint8"), ((batch * 2, 4), "float32"))
                )
                * 3
            )
        fields = (
            *accept_plan(batch=batch).storage,
            ((batch, 5, 248320), "float32"),
            ((batch, 4, 248320), "float32"),
            ((batch * 12,), "int64"),
            ((batch * 10,), "int32"),
            ((batch * 8,), "int64"),
            ((2 * 4 * (batch * table_width(config) + batch * 4 + 1),), "int32"),
        )
        total += storage_bytes(fields)
    return (
        total + storage_bytes((((config["max_num_seqs"], 4, 248320), "float32"),)) + 256
    )


def media_bytes(config, weights):
    fields = tuple(
        field
        for variant in ("input", "default", "default", "merge")
        for field in vision_plan(rows=65536, grid=(256, 256), variant=variant).storage
    )
    fields += (
        ((512 * 1024 * 1024,), "uint8"),
        ((config["max_num_seqs"], config["max_model_len"], 5120), "bfloat16"),
        ((3, min(32768, config["max_num_batched_tokens"]), 5120), "bfloat16"),
    )
    total = storage_bytes(fields)
    if weights.visual is None:
        for name, filename in weights.checkpoint.index.items():
            if name.startswith("model.visual."):
                header, _ = weights.checkpoint.header(filename)
                first, last = header[name]["data_offsets"]
                total += (last - first + 255) // 256 * 256
    return total


def text_workspace_budget(config, *, native_sort=None):
    return {
        "prefill": prefill_bytes(config),
        "retained": retained_bytes(config, native_sort),
        "native_sort_queried": native_sort is not None,
        "unverified_peak_components": ["cuda_graph", "allocator_reserved", "driver"],
        "table_width": table_width(config),
        "mtp_backup_bytes": 3
        * storage_bytes(
            (
                ((config["max_num_seqs"] * 2, 1024), "uint8"),
                ((config["max_num_seqs"] * 2, 4), "float32"),
            )
        )
        if config["num_speculative_tokens"]
        else 0,
    }


def workspace_budget(config, weights, *, native_sort):
    return {
        **text_workspace_budget(config, native_sort=native_sort),
        "media": media_bytes(config, weights),
    }
