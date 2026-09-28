"""Qwen MTP proposals with token-indexed immutable prefix pages.

An MTP row at position p pairs target hidden[p-1] with input token[p]. RoPE
positions shift uniformly by one, preserving relative attention. Position zero
is absent. This aligns every cached row with the token represented by Rust's
prefix hash; boundary rows no longer depend on a token outside that prefix.
"""

import logging
import os
from itertools import pairwise

import torch

from oh_my_vllm.kernels.mtp_attention import MTPAttention
from oh_my_vllm.models.qwen import AttentionBatch, Qwen
from oh_my_vllm.worker.batch_plan import BLOCK, PlannedRequest
from oh_my_vllm.worker.protocol import RequestOutput
from oh_my_vllm.worker.tensors import device_tensor, device_vectors

logger = logging.getLogger(__name__)


class MTP:
    def __init__(self, model: Qwen, capacity: int, max_tokens: int) -> None:
        if model.mtp is None:
            raise ValueError("MTP weights were not loaded")
        self.model = model
        self.device = model.embedding.device
        self.max_tokens = max_tokens
        self.cache = torch.zeros(
            capacity, 2, BLOCK, 4, 256, device=self.device, dtype=torch.bfloat16
        )
        self.boundary_hidden = torch.zeros(
            capacity, 5120, device=self.device, dtype=torch.bfloat16
        )
        self.attention = MTPAttention(max_tokens)
        self.next_position: dict[int, int] = {}
        self.graphs = {}
        self.chains = {}

    def forget(self, request_id: int) -> None:
        self.next_position.pop(request_id, None)

    def validate_state(self, plan: PlannedRequest, count: int) -> None:
        """Check request-local proposal state before accepting its target tokens."""
        req = plan.request
        start = req.num_computed_tokens
        expected = self.next_position.get(req.request_id, max(1, start))
        if expected not in (max(1, start), start + 1):
            raise ValueError("MTP cache position disagrees with target state")
        if start > 0 and expected == start and start % BLOCK:
            raise ValueError("missing MTP state away from a block boundary")
        if count <= 0:
            raise ValueError("MTP target commit must include an input token")

    def _run(
        self,
        tokens: list[int],
        hidden: torch.Tensor,
        starts: list[int],
        tables: list[list[int]],
        positions: list[int],
    ) -> torch.Tensor:
        decode_mode = max(b - a for a, b in pairwise(starts)) <= 5
        slots = [
            table[p // BLOCK] * BLOCK + p % BLOCK
            for i, table in enumerate(tables)
            for p in positions[starts[i] : starts[i + 1]]
        ]
        position_tensor, slot_tensor, token_tensor = device_vectors(
            [positions, slots, tokens], device=self.device
        )
        batch = AttentionBatch(position_tensor, slot_tensor, self.attention)
        if decode_mode and os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1":
            from oh_my_vllm.worker.decode_graph import DraftGraph

            extent = min(self.max_tokens, ((max(positions) + 4096) // 4096) * 4096)
            key = (len(tokens), len(starts) - 1, extent)
            if key in self.graphs or len(self.graphs) + len(self.chains) < 32:
                width = (self.max_tokens + BLOCK - 1) // BLOCK
                tables_tensor = device_tensor(
                    [
                        table + [0] * (width - len(table))
                        for i, table in enumerate(tables)
                        for _ in range(starts[i + 1] - starts[i])
                    ],
                    device=self.device,
                    dtype=torch.int32,
                )
                starts_tensor = (
                    device_tensor(starts, device=self.device, dtype=torch.int32)
                    if len(tokens) > len(starts) - 1
                    else None
                )
                if key not in self.graphs:
                    logger.info("Capture draft graph: %s", key)
                    self.graphs[key] = DraftGraph(
                        self.model,
                        self.cache,
                        token_tensor,
                        hidden,
                        batch,
                        tables_tensor,
                        extent,
                        starts_tensor,
                    )
                return self.graphs[key].replay(
                    token_tensor, hidden, batch, tables_tensor, starts_tensor
                )
        self.attention.plan(starts, tables, positions)
        return self.model.draft(token_tensor, hidden, batch, self.cache)

    def _proposal_graph(self, hidden: torch.Tensor, eligible) -> list[list[int]] | None:
        # All three subsequent writes must fit Rust's private allocated pages.
        # Near a boundary, retain the stepwise path that shrinks the active batch.
        if (
            self.device.type != "cuda"
            or os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") == "1"
            or any(computed + 3 >= limit for _, _, computed, limit in eligible)
        ):
            return None
        from oh_my_vllm.worker.decode_graph import ProposalGraph

        last_position = max(p for _, _, p, _ in eligible) + 3
        extent = min(self.max_tokens, ((last_position + 4096) // 4096) * 4096)
        key = (len(eligible), extent)
        if key not in self.chains and len(self.graphs) + len(self.chains) >= 32:
            return None
        width = (self.max_tokens + BLOCK - 1) // BLOCK
        tables = device_tensor(
            [table + [0] * (width - len(table)) for _, table, _, _ in eligible],
            device=self.device,
            dtype=torch.int64,
        )
        positions = device_tensor([p for _, _, p, _ in eligible], device=self.device)
        if key not in self.chains:
            logger.info("Capture proposal graph: %s", key)
            self.chains[key] = ProposalGraph(
                self.model, self.cache, hidden, positions, tables, extent
            )
        return self.chains[key].replay(hidden, positions, tables).tolist()

    def propose(
        self,
        plans: list[PlannedRequest],
        starts: list[int],
        counts: list[int],
        hidden: torch.Tensor,
        histories: dict[int, list[int]],
        outputs: list[RequestOutput],
    ) -> dict[int, list[int]]:
        # Retain target boundary features alongside the FA page that owns them.
        rows, pages = [], []
        for i, (plan, count) in enumerate(zip(plans, counts, strict=True)):
            req = plan.request
            begin, end = req.num_computed_tokens, req.num_computed_tokens + count
            for boundary in range((begin // BLOCK + 1) * BLOCK, end + 1, BLOCK):
                rows.append(starts[i] + boundary - begin - 1)
                pages.append(req.fa_block_table[boundary // BLOCK - 1])
        if rows:
            self.boundary_hidden[device_tensor(pages, device=self.device)] = hidden[
                rows
            ]
        tokens, positions, tables, parts, query_starts = [], [], [], [], [0]
        eligible, selected = [], []
        for i, (plan, count, output) in enumerate(
            zip(plans, counts, outputs, strict=True)
        ):
            req, rid = plan.request, plan.request.request_id
            start, computed = req.num_computed_tokens, req.num_computed_tokens + count
            limit = min(len(req.fa_block_table) * BLOCK, self.max_tokens)
            expected = self.next_position.get(rid, max(1, start))
            restore_boundary = start > 0 and expected == start
            if expected not in (max(1, start), start + 1):
                raise ValueError("MTP cache position disagrees with target state")
            first = start if restore_boundary else start + 1
            end = min(computed + 1, limit)
            if first >= end:
                self.next_position[rid] = first
                continue
            if restore_boundary:
                if start % BLOCK:
                    raise ValueError("missing MTP state away from a block boundary")
                page = req.fa_block_table[start // BLOCK - 1]
                parts.append(self.boundary_hidden[page : page + 1])
            target_count = max(0, end - (start + 1))
            if target_count:
                parts.append(hidden[starts[i] : starts[i] + target_count])
            tokens.extend(histories[rid][first:end])
            positions.extend(range(first, end))
            tables.append(req.fa_block_table)
            query_starts.append(len(tokens))
            self.next_position[rid] = end
            if (
                output.token_ids
                and output.finish_reason is None
                and end == computed + 1
            ):
                eligible.append((rid, req.fa_block_table, computed, limit))
                selected.append(len(tokens) - 1)
        proposals = {p.request.request_id: [] for p in plans}
        if not tokens:
            return proposals
        draft_hidden = self._run(
            tokens, torch.cat(parts, 0), query_starts, tables, positions
        )
        if not eligible:
            return proposals
        last_hidden = draft_hidden[selected]
        graph_tokens = self._proposal_graph(last_hidden, eligible)
        if graph_tokens is not None:
            for (rid, _, _, _), row in zip(eligible, graph_tokens, strict=True):
                proposals[rid] = row
            return proposals
        next_tokens = self.model.logits(last_hidden).argmax(-1).tolist()
        for (rid, _, _, _), token in zip(eligible, next_tokens, strict=True):
            proposals[rid].append(token)
        for step in range(1, 4):
            active = [
                i
                for i, (_, _, computed, limit) in enumerate(eligible)
                if computed + step < limit
            ]
            if not active:
                break
            eligible = [eligible[i] for i in active]
            tokens = [next_tokens[i] for i in active]
            positions = [computed + step for _, _, computed, _ in eligible]
            tables = [table for _, table, _, _ in eligible]
            last_hidden = self._run(
                tokens,
                last_hidden
                if active == list(range(len(last_hidden)))
                else last_hidden[active],
                list(range(len(active) + 1)),
                tables,
                positions,
            )
            next_tokens = self.model.logits(last_hidden).argmax(-1).tolist()
            for (rid, _, _, _), token in zip(eligible, next_tokens, strict=True):
                proposals[rid].append(token)
        return proposals
