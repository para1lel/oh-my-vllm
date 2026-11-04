"""Batched target execution with independent request positions and commits."""

from dataclasses import dataclass

import torch

from . import kernel, model
from .decode import (
    DecodeBindings,
    DecodeGraph,
    EngramStage,
    capture_decode_graph,
    native_sparse_graph_eligible,
)
from .model import context_token_bucket
from .speculation import verify_greedy_drafts
from .trace import RoutingTrace, record_accounting_call, target_call_id


@dataclass
class _TargetResult:
    logits: torch.Tensor
    hidden: torch.Tensor | None
    execution: str
    routes: RoutingTrace | None
    bindings: DecodeBindings
    call: int


def require_batch_configuration(args):
    released = dict(
        dim=5120,
        hc_mult=4,
        dtype="fp8",
        expert_dtype="fp4",
        n_heads=64,
        head_dim=512,
        n_routed_experts=384,
        n_activated_experts=6,
        index_n_heads=32,
        index_head_dim=128,
        window_size=128,
        index_topk=512,
    )
    for name, expected in released.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"batched CUDA target requires {name}={expected!r}; got {actual!r}"
            )
    if (
        model.fp8_block_size != 32
        or model.fp4_block_size != 32
        or model.scale_fmt != "ue8m0"
        or model.scale_dtype != torch.float8_e8m0fnu
    ):
        raise ValueError("batched CUDA target requires UE8M0 K32 projection scales")
    if kernel._SPARSE_NATIVE_FAILED:
        raise ValueError("batched CUDA target cannot reuse a failed native sparse path")


def target_decode_batch(
    worker,
    tokens,
    positions,
    slots,
    *,
    plan=None,
    valid_rows=None,
    dispatch_combine=None,
):
    if not tokens or len(tokens) != len(positions) or len(tokens) != len(slots):
        raise ValueError("decode batch needs matching requests, positions, and slots")
    length = len(tokens[0])
    if not 1 <= length <= 6 or any(len(row) != length for row in tokens):
        raise ValueError("decode batch needs one prepared verification length")
    if torch.device(worker.device).type == "cuda":
        require_batch_configuration(worker.args)
    window = worker.args.window_size
    groups = {
        (
            value % worker.decode_cycle,
            context_token_bucket(value + length),
            min(window, value) if length > 1 else window,
        )
        for value in positions
    }
    if len(groups) != 1:
        raise ValueError(
            "decode requests need the same context bucket and compressor branch"
        )
    for name in ("decode_bindings", "decode_engram", "decode_graphs"):
        if not hasattr(worker, name):
            setattr(worker, name, {})
    slots = tuple(slots)
    binding_key = ("batch", slots, length)
    bindings = worker.decode_bindings.get(binding_key)
    if bindings is None:
        limit = max(1, worker.decode_graph_limit)
        protected = {id(graph.bindings) for graph in worker.decode_graphs.values()}
        while len(worker.decode_bindings) >= limit:
            unused = next(
                (
                    key
                    for key, value in worker.decode_bindings.items()
                    if id(value) not in protected
                ),
                None,
            )
            if unused is None:
                break
            worker.decode_bindings.pop(unused).close()
            worker.decode_engram.pop(unused, None)
        bindings = DecodeBindings(
            len(slots), slots, worker.device, length=length, varlen=True
        )
        worker.decode_bindings[binding_key] = bindings
    bindings.bind_valid_rows(valid_rows)
    bindings.bind(tokens, positions)
    start_pos = max(positions)
    context = context_token_bucket(start_pos + length)
    worker.model.reserve_cache(list(slots), context)
    stage = worker.decode_engram.get(binding_key)
    if stage is None:
        stage = EngramStage(worker.model, len(slots), length=length)
        worker.decode_engram[binding_key] = stage
    engram_values = stage.stage(bindings, start_pos)
    tracing = getattr(worker, "routing_trace_enabled", False)
    collect_hidden = getattr(worker, "dspark", False)
    key = (
        "batch",
        slots,
        length,
        context,
        start_pos % worker.decode_cycle,
        min(start_pos, window) if length > 1 else window,
        collect_hidden,
        tracing,
        id(plan),
        id(dispatch_combine),
    )
    graph = worker.decode_graphs.get(key)
    enabled = (
        getattr(worker, "decode_graph_enabled", False)
        and bindings.storage.is_cuda
        and native_sparse_graph_eligible(worker.model)
    )
    if (
        graph is None
        and enabled
        and len(worker.decode_graphs) < worker.decode_graph_limit
    ):
        routes = (
            RoutingTrace(
                len(worker.model.layers),
                len(slots) * length,
                rank=worker.rank,
                strategy=worker.strategy,
                device=worker.device,
            )
            if tracing
            else None
        )
        graph = DecodeGraph(
            worker.model,
            bindings,
            start_pos,
            collect_main_hidden=collect_hidden,
            engram_values=engram_values,
            full_logits=length > 1,
            segmented=True,
            routing_trace=routes,
            plan=plan,
            dispatch_combine=dispatch_combine,
        )
        capture_decode_graph(graph, worker.rank)
        worker.decode_graphs[key] = graph
    worker.target_call = getattr(worker, "target_call", 0) + 1
    if graph is not None and enabled:
        graph.replay()
        call = target_call_id(worker, worker.target_call)
        return _TargetResult(
            graph.logits,
            graph.main_hidden,
            "graph",
            graph.routing_trace,
            bindings,
            call,
        )
    routes = (
        RoutingTrace(
            len(worker.model.layers),
            len(slots) * length,
            rank=worker.rank,
            strategy=worker.strategy,
            device=worker.device,
        )
        if tracing
        else None
    )
    position = (
        None if all(value == start_pos for value in positions) else bindings.position
    )
    _, logits, hidden = worker.model(
        bindings.token_ids,
        start_pos=start_pos,
        cache_slots=bindings.cache_slots,
        position=position,
        sample_output=False,
        cache_slots_cpu=slots,
        cache_positions_cpu=positions,
        full_logits=length > 1,
        logical_rows=1,
        collect_main_hidden=collect_hidden,
        engram_values=engram_values,
        reserve_cache=False,
        routing_trace=routes,
        execution_plan=plan,
        valid_rows=bindings.valid_rows,
        dispatch_combine=dispatch_combine,
    )
    call = target_call_id(worker, worker.target_call)
    return _TargetResult(logits, hidden, "eager", routes, bindings, call)


def execute_decode_group(
    worker, rows, *, plan=None, valid_rows=None, dispatch_combine=None
):
    prepared = []
    try:
        for row in rows:
            request, slot, tokens, start = worker._prepare_row(row)
            prepared.append((row, request, slot, tokens, start))
            known = min(len(tokens), len(request.history) - start)
            if known != 1 or not 1 <= len(tokens) <= 6:
                raise ValueError(
                    "batched decode requires one accepted input and at most five drafts"
                )
            if hasattr(worker, "cache"):
                worker.cache.begin_step(
                    row,
                    slot,
                    start_pos=start,
                    end_pos=start + len(tokens),
                    minimum_end=start + 1,
                )
        result = target_decode_batch(
            worker,
            [item[3] for item in prepared],
            tuple(item[4] for item in prepared),
            tuple(item[2] for item in prepared),
            plan=plan,
            valid_rows=valid_rows,
            dispatch_combine=dispatch_combine,
        )
        predictions = result.logits.argmax(dim=-1).tolist()
        outputs, credited, request_rows = {}, [], []
        for index, (row, request, slot, token_ids, start_pos) in enumerate(prepared):
            drafts = token_ids[1:]
            if drafts:
                verification = verify_greedy_drafts(drafts, predictions[index])
                tokens, text = request.append(list(verification.token_ids))
                accepted = min(
                    verification.accepted_draft_tokens, max(0, len(tokens) - 1)
                )
            else:
                tokens, text = request.append([int(predictions[index])])
                accepted = 0
            count = accepted + 1
            accepted_end = start_pos + count
            hidden = (
                result.hidden[index : index + 1, :count]
                if result.hidden is not None
                else None
            )
            mtp_updated = worker._seed_mtp(hidden, start_pos, slot, source_length=count)
            proposals = worker._next_proposals(
                request, tokens, hidden, accepted_end, slot
            )
            if hasattr(worker, "cache"):
                worker.cache.commit(
                    row["request_id"],
                    row,
                    slot,
                    start_pos=start_pos,
                    accepted_end=accepted_end,
                    include_mtp=mtp_updated,
                )
                if drafts:
                    worker.cache.restore(
                        row["request_id"],
                        slot,
                        row,
                        start_pos=accepted_end,
                        use_prefix_state=False,
                    )
            request.state_position = accepted_end
            outputs[row["request_id"]] = worker._output(
                request, row, tokens, text, accepted, proposals
            )
            credited.append(count)
            request_rows.append(
                dict(
                    id=row["request_id"],
                    start=start_pos,
                    count=count,
                    accepted=accepted,
                    proposed=len(drafts),
                    logit_rows=count,
                )
            )
        worker._target_execution = result.execution
        worker.trace.record(
            "target",
            rank=getattr(worker, "rank", 0),
            verify=len(prepared[0][3]) > 1,
            accepted=sum(item["accepted"] for item in request_rows),
            proposed=sum(item["proposed"] for item in request_rows),
            execution=result.execution,
            requests=[
                dict(
                    id=row["request_id"],
                    start=start,
                    count=len(tokens),
                    prompt=request.prompt_len,
                )
                for row, request, _slot, tokens, start in prepared
            ],
        )
        if result.routes is not None:
            mask = result.bindings.credit_rows(credited)
            mask = mask & result.bindings.valid_rows
            record_accounting_call(
                worker,
                call=result.call,
                phase="decode",
                request_rows=request_rows,
                counters=result.routes.counters(mask),
            )
        return outputs
    except BaseException:
        for _row, request, _slot, _tokens, _start in prepared:
            worker._free_slot(request)
            request.state_position = 0
        raise
