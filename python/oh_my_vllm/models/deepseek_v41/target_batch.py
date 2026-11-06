import torch

from .adapter import (
    DS_PARK_DRAFT_TOKENS,
)
from .worker_limits import (
    MAX_NATIVE_ROWS,
    prefill_subbatch_limit,
)


class BatchExecution:
    def _target_batch(
        self,
        token_batches,
        start_pos,
        cache_slots,
        *,
        full_logits=False,
        compute_logits=True,
        accounting=None,
    ):
        if not token_batches:
            raise ValueError("DeepSeek target batch must contain a request")
        length = len(token_batches[0])
        if length < 1 or any(len(tokens) != length for tokens in token_batches):
            raise ValueError("DeepSeek target batch rows must have one length")
        groups = []
        subbatch_limit = prefill_subbatch_limit(self.prefill_rows)
        for offset in range(0, len(token_batches), subbatch_limit):
            groups.append(
                self._target_batch_group(
                    token_batches[offset : offset + subbatch_limit],
                    start_pos,
                    cache_slots[offset : offset + subbatch_limit],
                    full_logits=full_logits,
                    compute_logits=compute_logits,
                    accounting=None
                    if accounting is None
                    else accounting.subset(offset, offset + subbatch_limit),
                )
            )
        if len(groups) == 1:
            return groups[0]
        output_ids = self._merge_batch_values(groups, 0)
        logits = self._merge_batch_values(groups, 1)
        hidden = self._merge_batch_values(groups, 2)
        return output_ids, logits, hidden

    @staticmethod
    def _merge_batch_values(results, index):
        values = [result[index] for result in results]
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if len(tensors) != len(values):
            return values[-1]
        return torch.cat(tensors, dim=0)

    def _target_batch_group(
        self,
        token_batches,
        start_pos,
        cache_slots,
        *,
        full_logits=False,
        compute_logits=True,
        accounting=None,
    ):
        if full_logits and not compute_logits:
            raise ValueError("full logits require vocab projection")
        length = len(token_batches[0])
        limit_rows = self.prefill_rows
        chunk_size = max(1, limit_rows // len(token_batches))
        cache_slots_cpu = cache_slots.tolist()
        output_ids = logits = main_hidden = None
        logit_chunks = [] if full_logits else None
        execution = "eager"
        for offset in range(0, length, chunk_size):
            chunk_length = min(chunk_size, length - offset)
            chunk_compute_logits = compute_logits and (
                full_logits or offset + chunk_length == length
            )
            routes = None if accounting is None else accounting.routes(chunk_length)
            graph = self._batch_decode_graph(
                token_batches,
                start_pos,
                offset,
                length,
                chunk_compute_logits,
                full_logits,
                cache_slots_cpu,
                routes,
            )
            if graph is not None:
                execution = "graph"
                routes = self._graph_routes if accounting is not None else None
                output_ids = None
                chunk_logits, chunk_hidden = graph
            else:
                output_ids, chunk_logits, chunk_hidden = self._eager_target_batch(
                    token_batches,
                    offset,
                    chunk_size,
                    start_pos,
                    cache_slots,
                    full_logits,
                    chunk_compute_logits,
                    cache_slots_cpu,
                    routes,
                )
            if accounting is not None:
                accounting.append(
                    routes,
                    start_pos + offset,
                    chunk_length,
                    chunk_compute_logits,
                    verification=full_logits,
                )
            logits = chunk_logits
            if logit_chunks is not None:
                logit_chunks.append(chunk_logits)
            main_hidden = self._collect_target_hidden(
                chunk_hidden,
                main_hidden,
                start_pos + offset,
                cache_slots,
                cache_slots_cpu,
                start_pos + length,
            )
            if isinstance(chunk_hidden, torch.Tensor):
                main_hidden = self._main_hidden_window(main_hidden, full_logits)
            del chunk_hidden
        self._target_execution = execution
        if logit_chunks is not None:
            logits = torch.cat(logit_chunks, dim=1)
        return output_ids, logits, main_hidden

    def _batch_decode_graph(
        self,
        token_batches,
        start_pos,
        offset,
        length,
        chunk_compute_logits,
        full_logits,
        cache_slots_cpu,
        routes,
    ):
        graph = None
        if (
            chunk_compute_logits
            and not full_logits
            and len(token_batches) == 1
            and length == 1
            and start_pos + offset > 0
        ):
            graph = self._decode_graph_step(
                token_batches[0][0],
                start_pos + offset,
                cache_slots_cpu[0],
                collect_main_hidden=self.dspark,
                **({"routes": routes} if routes is not None else {}),
            )

        return graph

    def _eager_target_batch(
        self,
        token_batches,
        offset,
        chunk_size,
        start_pos,
        cache_slots,
        full_logits,
        chunk_compute_logits,
        cache_slots_cpu,
        routes,
    ):
        inputs = self._input_tokens(
            [tokens[offset : offset + chunk_size] for tokens in token_batches]
        )
        output_ids, chunk_logits, chunk_hidden = self.model(
            inputs,
            start_pos=start_pos + offset,
            cache_slots=cache_slots,
            full_logits=full_logits,
            compute_logits=chunk_compute_logits,
            collect_main_hidden=self.dspark,
            cache_slots_cpu=cache_slots_cpu,
            **({"routing_trace": routes} if routes is not None else {}),
        )

        return output_ids, chunk_logits, chunk_hidden

    def _seed_mtp_batch(
        self,
        main_hidden,
        start_pos,
        cache_slots,
        *,
        source_length=None,
        cache_slots_cpu=None,
    ):
        if not self.dspark or main_hidden is None:
            return False
        batch_size, hidden_length, _ = main_hidden.size()
        source_length = hidden_length if source_length is None else source_length
        if type(source_length) is not int or source_length < hidden_length:
            raise ValueError("DeepSeek MTP source length is shorter than hidden state")
        window_size = self.args.window_size
        if self.cache.exports.steps:
            window_size += DS_PARK_DRAFT_TOKENS
        hidden_start = start_pos + source_length - hidden_length
        if hidden_length > window_size:
            main_hidden = main_hidden[:, -window_size:]
            hidden_start += hidden_length - window_size
            hidden_length = window_size
        chunk_size = max(1, MAX_NATIVE_ROWS // batch_size)
        options = {}
        if cache_slots_cpu is not None:
            options["cache_slots_cpu"] = cache_slots_cpu
        for offset in range(0, hidden_length, chunk_size):
            self.model.forward_spec_seed(
                main_hidden[:, offset : offset + chunk_size],
                start_pos=hidden_start + offset,
                cache_slots=cache_slots,
                **options,
            )
        return True
