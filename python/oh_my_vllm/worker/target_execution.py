from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from oh_my_vllm.models.qwen import Batch
from oh_my_vllm.worker.batch_plan import BLOCK, PlannedRequest
from oh_my_vllm.worker.tensors import device_page_tables, device_tensor, device_vectors

if TYPE_CHECKING:
    from oh_my_vllm.worker.model_runner import OhMyVllmWorker

logger = logging.getLogger("oh_my_vllm.worker.model_runner")


@dataclass
class TargetOutput:
    hidden: torch.Tensor
    features: torch.Tensor | None
    logits: torch.Tensor | None
    request_starts: list[int]


def run_target(worker: OhMyVllmWorker, plans: list[PlannedRequest]) -> TargetOutput:
    ids, positions, sequence_ids, writes, slots, pages = [], [], [], [], [], []
    starts, page_starts, last_lengths = [0], [0], []
    for sequence, plan in enumerate(plans):
        req = plan.request
        count = len(req.token_ids)
        end = req.num_computed_tokens + count
        ids.extend(req.token_ids)
        positions.extend(range(req.num_computed_tokens, end))
        sequence_ids.extend([sequence] * count)
        writes.extend(plan.writes)
        slots.extend(
            req.fa_block_table[p // BLOCK] * BLOCK + p % BLOCK
            for p in range(req.num_computed_tokens, end)
        )
        starts.append(len(ids))
        pages.extend(req.fa_block_table[: (end + BLOCK - 1) // BLOCK])
        page_starts.append(len(pages))
        last_lengths.append((end - 1) % BLOCK + 1)
    cpu_starts = torch.tensor(starts, dtype=torch.int32)
    extent = min(worker.config.max_model_len, ((max(positions) + 4096) // 4096) * 4096)
    decode_only = not any(p.prefill for p in plans)
    key = (
        worker.config.decode_graph_key([len(p.writes) for p in plans], extent)
        if decode_only
        else ()
    )
    family = "target_dspark" if worker._is_dspark() else "target"
    use_graph = (
        decode_only
        and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1"
        and worker.graph_cache.should_use(family, key)
    )
    if not use_graph:
        worker.attention.plan(
            cpu_starts,
            torch.tensor(page_starts, dtype=torch.int32),
            torch.tensor(pages, dtype=torch.int32),
            torch.tensor(last_lengths, dtype=torch.int32),
            24,
            4,
            256,
        )
    metadata = device_vectors(
        [
            positions,
            starts,
            sequence_ids,
            [p.source for p in plans],
            writes,
            [p.writes[-1] for p in plans],
            slots,
            ids,
        ]
    )
    batch = Batch(
        positions=metadata[0],
        # Torch/CUDA prefill consumes int32 offsets; convert once per batch.
        starts=metadata[1].to(torch.int32),
        sequence_ids=metadata[2],
        state_reads=metadata[3],
        state_writes=metadata[4],
        final_state_writes=metadata[5],
        fa_slots=metadata[6],
        attention=worker.attention,
        prefill_sequences=sum(p.prefill for p in plans),
        prefill_tokens=sum(len(p.writes) for p in plans if p.prefill),
    )
    token_tensor = metadata[7]
    selected = [
        starts[i] + row for i, p in enumerate(plans) for row in p.sample_indices
    ]
    if (
        not decode_only
        and len(selected) < len(ids)
        and (worker._proposer() is None or worker._is_dspark())
    ):
        # MTP consumes historical target hidden rows; ordinary and DSpark
        # consume only logits here. DSpark feature taps precede layer 63.
        batch.output_indices = device_tensor(selected, dtype=torch.int64)
        if selected:
            from oh_my_vllm.worker.decode_graph import DecodeAttention

            selected_tables = device_page_tables(
                [p.request.fa_block_table for p in plans],
                (extent + BLOCK - 1) // BLOCK,
                counts=[len(p.sample_indices) for p in plans],
            )
            lengths = device_tensor(
                [positions[row] + 1 for row in selected], dtype=torch.int32
            )
            batch.output_attention = DecodeAttention(selected_tables, lengths, extent)
    hidden = None
    graph_logits = None
    features = None
    if use_graph:
        from oh_my_vllm.worker.decode_graph import DecodeGraph

        width = (extent + BLOCK - 1) // BLOCK
        tables = device_page_tables(
            [p.request.fa_block_table for p in plans],
            width,
            counts=[len(p.writes) for p in plans],
            dtype=torch.int32,
            device="cuda",
        )

        def capture():
            pool_name = "dspark_graph_pool" if worker._is_dspark() else "graph_pool"
            if getattr(worker, pool_name) is None:
                setattr(worker, pool_name, torch.cuda.graph_pool_handle())
            logger.info(
                "Capture target graph: tokens=%d requests=%d "
                "extent=%d max_query_len=%d",
                *key,
            )
            try:
                return DecodeGraph(
                    worker.model,
                    worker.caches,
                    token_tensor,
                    batch,
                    tables,
                    extent,
                    pool=getattr(worker, pool_name),
                    compile_model=os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1",
                    max_query_len=key[-1],
                    capture_features=worker._is_dspark(),
                )
            except Exception:
                if not worker.graph_cache.has_family(family):
                    setattr(worker, pool_name, None)
                raise

        graph = worker.graph_cache.get_or_create(family, key, capture)
        if graph is None:
            worker.attention.plan(
                cpu_starts,
                torch.tensor(page_starts, dtype=torch.int32),
                torch.tensor(pages, dtype=torch.int32),
                torch.tensor(last_lengths, dtype=torch.int32),
                24,
                4,
                256,
            )
        else:
            hidden, graph_logits = graph.replay(token_tensor, batch, tables)
            features = graph.features if worker._is_dspark() else None
    if not decode_only and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1":
        from oh_my_vllm.worker.prefill_budget import prefill_capture_headroom
        from oh_my_vllm.worker.prefill_graph import PrefillGraph

        prefill_key = (
            tuple(len(p.writes) for p in plans),
            tuple(p.request.num_computed_tokens for p in plans),
            batch.prefill_sequences,
            tuple(selected) if batch.output_indices is not None else None,
        )
        prefill_family = "prefill_dspark" if worker._is_dspark() else "prefill"
        headroom = prefill_capture_headroom(
            worker.model, worker.caches, token_tensor, batch, writes
        )
        if worker.prefill_graph_cache.should_use(
            prefill_family, prefill_key, headroom_bytes=headroom
        ):
            attention_plan = worker.attention._ir_reference_plan

            def capture_prefill():
                logger.info(
                    "Capture prefill graph: queries=%s contexts=%s headroom_bytes=%d",
                    *prefill_key[:2],
                    headroom,
                )
                pool_name = (
                    "prefill_dspark_graph_pool"
                    if worker._is_dspark()
                    else "prefill_graph_pool"
                )
                if getattr(worker, pool_name) is None:
                    setattr(worker, pool_name, torch.cuda.graph_pool_handle())
                try:
                    return PrefillGraph(
                        worker.model,
                        worker.caches,
                        token_tensor,
                        batch,
                        attention_plan,
                        pool=getattr(worker, pool_name),
                        capture_features=worker._is_dspark(),
                    )
                except Exception:
                    if not worker.prefill_graph_cache.has_family(prefill_family):
                        setattr(worker, pool_name, None)
                    raise

            graph = worker.prefill_graph_cache.get_or_create(
                prefill_family,
                prefill_key,
                capture_prefill,
                headroom_bytes=headroom,
            )
            if graph is not None:
                hidden, features = graph.replay(token_tensor, batch, attention_plan)
    if hidden is None:
        if worker._is_dspark():
            hidden, features = worker.feature_unit(token_tensor, batch, worker.caches)
        else:
            hidden = worker.forward_unit(token_tensor, batch, worker.caches)
    logits = None
    if selected:
        if graph_logits is None:
            logits = worker.logits_unit(hidden[selected])
        elif selected == list(range(len(ids))):
            logits = graph_logits
        else:
            logits = graph_logits[selected]
    return TargetOutput(hidden, features, logits, starts)
