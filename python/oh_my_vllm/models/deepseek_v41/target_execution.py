import torch

from .adapter import (
    DS_PARK_DRAFT_TOKENS,
)
from .decode import (
    DecodeBindings,
    DecodeGraph,
    EngramStage,
    capture_decode_graph,
    native_sparse_graph_eligible,
)
from .model import context_token_bucket
from .proposal_graph import ProposalGraph


class TargetExecution:
    @staticmethod
    def _host_input_buffer(rows, columns):
        try:
            return torch.empty(
                (rows, columns),
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            return torch.empty((rows, columns), dtype=torch.long, device="cpu")

    def _input_tokens(self, token_batches):
        rows = len(token_batches)
        if not rows:
            raise ValueError("DeepSeek token input must contain a row")
        length = len(token_batches[0])
        if (
            length < 1
            or any(len(tokens) != length for tokens in token_batches)
            or rows > self._input_device.size(0)
            or length > self._input_device.size(1)
        ):
            raise ValueError("DeepSeek token input exceeds the prepared buffer")
        event = self._input_copy_event
        if event is not None:
            event.synchronize()
        source = torch.as_tensor(token_batches, dtype=torch.long, device="cpu")
        host = self._input_host[:rows, :length]
        host.copy_(source)
        device = self._input_device[:rows, :length]
        device.copy_(host, non_blocking=True)
        if event is None:
            event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._input_copy_event = event
        return device

    def _decode_graph_step(
        self, token, position, slot, *, collect_main_hidden, routes=None
    ):
        """Replay a captured decode step, or return None for the Eager path.

        A graph is captured on the first step of each (slot, context bucket,
        position residue) group. The graph cache is bounded, so a step outside a
        captured group falls back to Eager.
        """
        if not self.decode_graph_enabled or not native_sparse_graph_eligible(
            self.model
        ):
            return None
        slots = (slot,)
        bindings = self.decode_bindings.get(slots)
        if bindings is None:
            bindings = DecodeBindings(1, slots, self.device)
            self.decode_bindings[slots] = bindings
        stage = self.decode_engram.get(slots)
        if stage is None:
            stage = EngramStage(self.model, 1)
            self.decode_engram[slots] = stage
        context = context_token_bucket(position + 1)
        key = (
            slots,
            context,
            position % self.decode_cycle,
            collect_main_hidden,
            routes is not None,
        )
        graph = self.decode_graphs.get(key)
        if graph is None:
            if len(self.decode_graphs) >= self.decode_graph_limit:
                return None
            bindings.bind(token, position)
            self.model.reserve_cache(list(slots), context)
            values = stage.stage(bindings, position)
            graph = DecodeGraph(
                self.model,
                bindings,
                position,
                collect_main_hidden=collect_main_hidden,
                engram_values=values,
                routing_trace=routes,
            )
            capture_decode_graph(graph, self.rank)
            self.decode_graphs[key] = graph
        else:
            bindings.bind(token, position)
            self.model.reserve_cache(list(slots), context)
            stage.stage(bindings, position)
        graph.replay()
        self._graph_routes = graph.routing_trace
        self.model._bind_state_exports(bindings.cache_slots, bindings.cache_slots_cpu)
        self.cache.export_decoded_step(slots, position, position + 1)
        return graph.logits, graph.main_hidden

    def _target(
        self,
        token_ids,
        start_pos,
        slot,
        *,
        full_logits=False,
        compute_logits=True,
        logit_start=0,
        accounting=None,
    ):
        cache_slots = torch.tensor([slot], device=self.device)
        output_ids = logits = main_hidden = None
        logit_chunks = [] if full_logits else None
        limit_rows = self._target_row_limit(token_ids, start_pos, full_logits)
        execution = "eager"
        for offset in range(0, len(token_ids), limit_rows):
            chunk = token_ids[offset : offset + limit_rows]
            chunk_start = start_pos + offset
            chunk_compute_logits = compute_logits and (
                full_logits or offset + len(chunk) == len(token_ids)
            )
            routes = None if accounting is None else accounting.routes(len(chunk))
            graph = None
            if (
                chunk_compute_logits
                and not full_logits
                and len(chunk) == 1
                and chunk_start > 0
            ):
                graph = self._decode_graph_step(
                    chunk[0],
                    chunk_start,
                    slot,
                    collect_main_hidden=self.dspark,
                    **({"routes": routes} if routes is not None else {}),
                )
            if graph is not None:
                execution = "graph"
                routes = self._graph_routes if accounting is not None else None
                chunk_logits, chunk_hidden = graph
                output_ids = None
            else:
                output_ids, chunk_logits, chunk_hidden = self._eager_target(
                    chunk,
                    chunk_start,
                    slot,
                    cache_slots,
                    full_logits,
                    chunk_compute_logits,
                    logit_start,
                    offset,
                    routes,
                )
            if accounting is not None:
                accounting.append(
                    routes,
                    chunk_start,
                    len(chunk),
                    chunk_compute_logits,
                    verification=full_logits,
                )
            logits = chunk_logits
            if logit_chunks is not None:
                logit_chunks.append(chunk_logits)
            main_hidden = self._collect_target_hidden(
                chunk_hidden,
                main_hidden,
                chunk_start,
                cache_slots,
                (slot,),
                start_pos + len(token_ids),
            )
            if isinstance(chunk_hidden, torch.Tensor):
                main_hidden = self._main_hidden_window(main_hidden, full_logits)
            del chunk_hidden
        self._target_execution = execution
        if logit_chunks is not None:
            logits = torch.cat(logit_chunks, dim=1)
        return output_ids, logits, main_hidden

    def _target_row_limit(self, token_ids, start_pos, full_logits):
        limit_rows = self.prefill_rows
        if full_logits and start_pos > 0 and len(token_ids) <= DS_PARK_DRAFT_TOKENS + 1:
            limit_rows = max(limit_rows, len(token_ids))

        return limit_rows

    def _eager_target(
        self,
        chunk,
        chunk_start,
        slot,
        cache_slots,
        full_logits,
        chunk_compute_logits,
        logit_start,
        offset,
        routes,
    ):
        inputs = self._input_tokens((chunk,))
        chunk_logit_start = max(0, logit_start - offset) if full_logits else 0
        output_ids, chunk_logits, chunk_hidden = self.model(
            inputs,
            start_pos=chunk_start,
            cache_slots=cache_slots,
            full_logits=full_logits,
            compute_logits=chunk_compute_logits,
            collect_main_hidden=self.dspark,
            logit_start=chunk_logit_start,
            sample_output=not full_logits,
            cache_slots_cpu=(slot,),
            **({"routing_trace": routes} if routes is not None else {}),
        )

        return output_ids, chunk_logits, chunk_hidden

    def _collect_target_hidden(
        self,
        chunk_hidden,
        main_hidden,
        chunk_start,
        cache_slots,
        slots_cpu,
        final_end,
    ):
        if isinstance(chunk_hidden, torch.Tensor):
            self._seed_mtp_checkpoints(
                chunk_hidden,
                main_hidden,
                chunk_start,
                cache_slots,
                slots_cpu,
                final_end,
            )
            if isinstance(main_hidden, torch.Tensor):
                main_hidden = torch.cat((main_hidden, chunk_hidden), dim=1)
            else:
                main_hidden = chunk_hidden
        elif main_hidden is None:
            main_hidden = chunk_hidden

        return main_hidden

    def _main_hidden_window(self, main_hidden, full_logits):
        limit = self.args.window_size
        limit += DS_PARK_DRAFT_TOKENS if full_logits else 0
        return main_hidden[:, -limit:].clone()

    def _main_hidden_tail(self, results):
        hidden = [
            result[2] for result in results if isinstance(result[2], torch.Tensor)
        ]
        if not hidden:
            return results[-1][2]
        limit = self.args.window_size
        tail = hidden[0]
        for value in hidden[1:]:
            tail = torch.cat((tail, value), dim=1)
            if tail.size(1) > limit:
                tail = tail[:, -limit:]
        return tail[:, -limit:]

    def _proposal_graph_step(self, input_ids, main_hidden, start_pos, slot):
        if (
            not getattr(self, "proposal_graph_enabled", False)
            or self.decode_graph_limit == 0
            or main_hidden.shape[:2] != (1, 1)
            or self.model.temperature != 0
            or any(start_pos < layer.attn.window_size - 1 for layer in self.model.mtp)
        ):
            return None
        graph = self.proposal_graphs.get(slot)
        if graph is not None and not graph.bindings_valid():
            self._close_proposal_graph(slot)
            graph = None
        if graph is None:
            graph = ProposalGraph(self.model, slot, main_hidden)
            try:
                graph.bind(input_ids, main_hidden, start_pos)
                capture_decode_graph(graph, self.rank)
            except BaseException:
                graph.close()
                raise
            self.proposal_graphs[slot] = graph
        try:
            graph.bind(input_ids, main_hidden, start_pos)
            return graph.replay()
        except BaseException:
            self._close_proposal_graph(slot)
            raise

    def _close_proposal_graph(self, slot):
        graph = getattr(self, "proposal_graphs", {}).pop(slot, None)
        if graph is not None:
            graph.close()

    def _proposal(
        self, token, main_hidden, start_pos, slot, *, input_ids=None, cache_slots=None
    ):
        if not self.dspark or main_hidden is None:
            return []
        if input_ids is None:
            input_ids = torch.tensor([[token]], device=self.device, dtype=torch.long)
        options = {}
        if cache_slots is None:
            cache_slots = torch.tensor([slot], device=self.device)
        else:
            options["cache_slots_cpu"] = (slot,)
        result = self._proposal_graph_step(input_ids, main_hidden, start_pos, slot)
        if result is None:
            result = self.model.forward_spec(
                input_ids,
                main_hidden,
                start_pos=start_pos,
                cache_slots=cache_slots,
                **options,
            )
        if result is None:
            return []
        output_ids, _, _ = result
        return output_ids[0, 1:].tolist()

    def _next_proposals(
        self,
        request,
        tokens,
        main_hidden,
        accepted_end,
        slot,
        *,
        input_ids=None,
        cache_slots=None,
    ):
        remaining = max(0, request.limit - request.generated)
        if (
            not self.dspark
            or request.finish
            or not tokens
            or remaining <= 1
            or main_hidden is None
            or accepted_end + DS_PARK_DRAFT_TOKENS > self.runtime_max_context
        ):
            return []
        options = {}
        if input_ids is not None:
            options["input_ids"] = input_ids
        if cache_slots is not None:
            options["cache_slots"] = cache_slots
        return self._proposal(
            tokens[-1], main_hidden[:, -1:], accepted_end - 1, slot, **options
        )[: remaining - 1]

    def _seed_mtp(
        self, main_hidden, start_pos, slot, *, source_length=None, cache_slots=None
    ):
        if main_hidden is None:
            return False
        options = {}
        if cache_slots is None:
            cache_slots = torch.tensor([slot], device=main_hidden.device)
        else:
            options["cache_slots_cpu"] = (slot,)
        return self._seed_mtp_batch(
            main_hidden,
            start_pos,
            cache_slots,
            source_length=source_length,
            **options,
        )

    def _mtp_source_length(self, length):
        window_size = self.args.window_size
        return length if length > window_size else None

    def _seed_mtp_checkpoints(
        self, hidden, previous, start_pos, slots, slots_cpu, final_end
    ):
        if not self.dspark:
            return
        ends = self.cache.mtp_checkpoint_ends(
            slots_cpu, start_pos, start_pos + hidden.size(1), final_end
        )
        width = self.args.window_size
        for end in ends:
            count = end - start_pos
            selected = hidden[:, max(0, count - width) : count]
            if selected.size(1) < width and isinstance(previous, torch.Tensor):
                selected = torch.cat(
                    (previous[:, -(width - selected.size(1)) :], selected), dim=1
                )
            self._seed_mtp_batch(selected, end - selected.size(1), slots)
