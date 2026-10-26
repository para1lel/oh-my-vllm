from itertools import accumulate, pairwise

from ...operators.fa.plan import make_plan as fa_plan
from ...operators.ffn.plan import make_plan as ffn_plan
from ...operators.gdn.api import Schedule
from ...operators.gdn.plan import chunk_workspace
from ...operators.gdn.plan import make_plan as gdn_plan
from ...operators.mtp.plan import accept_plan, context_plan, draft_plan
from ...operators.output.plan import make_plan as output_plan
from ...operators.runtime.storage import storage_bytes
from ...operators.vision.plan import make_plan as vision_plan


def schedule(lengths, verify):
    starts = tuple(accumulate((0, *lengths)))
    rows, batch = starts[-1], len(lengths)
    checkpoints = ()
    if not verify:
        checkpoints = tuple(
            row
            for first, last in pairwise(starts)
            for row in range(first, last)
            if (row - first + 1) % 784 == 0 or row + 1 == last
        )
    return Schedule(
        starts,
        tuple(range(batch)),
        tuple(range(batch, batch + rows)) if verify else (-1,) * rows,
        checkpoints,
        tuple(range(batch, batch + len(checkpoints))),
        (0,) * batch,
        batch + max(rows if verify else len(checkpoints), 1),
        verify,
    )


def target_storage(rows, batch, maximum, mtp, recurrent, context=262144, minimum=0):
    attention = fa_plan(
        rows=rows,
        batch=batch,
        table_width=335,
        max_queries=maximum,
        max_context=context,
        min_context=minimum,
    )
    fields = (
        attention.storage
        + recurrent.storage
        + ffn_plan(rows=rows).storage
        + (((rows, 5120), "bfloat16"),) * 2
    )
    if mtp:
        fields += output_plan(rows=rows, hidden_only=True).storage
    fields += (
        ((3, rows), "int64"),
        ((rows,), "int32"),
        ((batch, 335), "int32"),
        ((batch + 1,), "int32"),
        ((rows,), "int32"),
        ((rows,), "int64"),
        ((batch + 1,), "int32"),
        ((batch,), "int32"),
        ((rows,), "int32"),
    )
    return attention, fields


def context_storage(rows, batch, maximum, parallel):
    plan = context_plan(rows=rows, batch=batch, max_queries=maximum)
    fields = (
        *plan.storage,
        *plan.attention.storage,
        ((rows,), "int64"),
        ((rows,), "int32"),
        ((rows,), "int32"),
        ((3, rows), "int64"),
        ((rows,), "int32"),
        ((batch, 335), "int32"),
        ((batch + 1,), "int32"),
        ((rows,), "int32"),
    )
    if parallel:
        fields += (((parallel * 5, 5120), "bfloat16"),)
    return plan, fields


def prefill_bytes(config):
    mtp = bool(config["num_speculative_tokens"])
    limit = min(32764 if mtp else 32768, config["max_num_batched_tokens"])
    batch = min(config["max_num_seqs"], 4 if mtp else 20, limit)
    maximum = 0
    candidates = {min(limit, value) for value in (*range(1, 129), 4096, 4097, limit)}
    for rows in sorted(candidates):
        recurrent = gdn_plan(schedule((rows,), False))
        _, fields = target_storage(rows, 1, rows, mtp, recurrent)
        if recurrent.chunks:
            # Sum of ceiling divisions is at most ceil(total/divisor) + batch - 1.
            chunks = (rows + 63) // 64 + (rows + 783) // 784 + 3 * batch - 2
            fields += chunk_workspace(chunks - recurrent.chunks, batch)
        fields += (((batch * 335 + batch * 3,), "int32"),)
        maximum = max(maximum, storage_bytes(fields))
    if mtp:
        context_limit = min(32768, limit + batch)
        context_maximum = 0
        for rows in {
            min(context_limit, count) for count in (*range(1, 33), context_limit)
        }:
            _, fields = context_storage(rows, min(batch, rows), rows, 0)
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


def sampler_bytes(rows, *, output=True, history=True, hidden=True):
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
    return storage_bytes(fields)


def draft_bytes(batch, first, *, history=True):
    plan = draft_plan(rows=batch, first=first)
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
            ((batch, 335), "int32"),
            ((batch + 1,), "int32"),
            ((batch,), "int32"),
            ((batch,), "int64"),
        )
    )
    return storage_bytes(fields) + sampler_bytes(
        batch, output=False, history=history, hidden=False
    )


def retained_bytes(config):
    mtp = bool(config["num_speculative_tokens"])
    parallel = min(config["max_num_seqs"], 4 if mtp else 20)
    queries = 5 if mtp else 1
    targets, contexts = {}, {}
    for batch in range(1, parallel + 1):
        for rows, maximum, recurrent in decode_shapes(batch, queries):
            for context, minimum in ((maximum, 0), (32768, 0), (32768, 32768)):
                attention, fields = target_storage(
                    rows, batch, maximum, mtp, recurrent, context, minimum
                )
                targets[(attention.reuse_key, recurrent.reuse_key)] = storage_bytes(
                    fields
                )
            if mtp:
                plan, fields = context_storage(
                    rows, batch, maximum, config["max_num_seqs"]
                )
                contexts[plan.reuse_key] = storage_bytes(fields)
    total = sum(targets.values()) + sum(contexts.values())
    total += sum(sampler_bytes(rows) for rows in range(1, parallel * queries + 1))
    if not mtp:
        return total
    for batch in range(1, parallel + 1):
        total += sum(draft_bytes(batch, first) for first in (False, True))
        total += 2 * sum(
            draft_bytes(batch, stage == 0, history=stage == 0) for stage in range(4)
        )
        fields = (
            *accept_plan(batch=batch).storage,
            ((batch, 5, 248320), "float32"),
            ((batch, 4, 248320), "float32"),
            ((batch * 12,), "int64"),
            ((batch * 10,), "int32"),
            ((batch * 8,), "int64"),
            ((2 * 4 * (batch * 335 + batch * 4 + 1),), "int32"),
        )
        total += storage_bytes(fields)
    return total + storage_bytes((((config["max_num_seqs"], 4, 248320), "float32"),))


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


def workspace_budget(config, weights):
    return {
        "prefill": prefill_bytes(config),
        "retained": retained_bytes(config),
        "media": media_bytes(config, weights),
    }
