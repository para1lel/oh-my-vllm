from contextlib import ExitStack

import numpy as np
import torch

from ...operators.output import api as output
from ...operators.output.plan import make_plan
from ...operators.output.workspace import prepare
from ...worker.constraints import batch_masks
from ...worker.metadata import Metadata


class SamplingInputs:
    def __init__(self, rows):
        self.full_history = torch.empty((rows, 2, 248320), dtype=torch.int32)
        self.presence_history = torch.empty((rows, 248320 // 8), dtype=torch.uint8)
        self.history = self.full_history
        self.compact = False
        self.mask = torch.empty((rows, 248320 // 8), dtype=torch.uint8)
        self.mask_keys = [None] * rows
        self.mask_changed = True
        self.parameters = torch.empty((rows, 7), dtype=torch.float32)
        self.random = torch.empty((rows, 2), dtype=torch.uint64)
        self.history_needed = False
        self.field = "random_offset"
        # Host metadata stays on the CPU. The packing and copy path uses NumPy
        # views because per-element Torch dispatch dominates these small arrays.
        self.full_history_view = self.full_history.numpy()
        self.presence_history_view = self.presence_history.numpy()
        self.mask_view = self.mask.numpy()
        self.parameters_view = self.parameters.numpy()
        self.random_view = self.random.numpy()

    def update(self, requests, prefixes, stream):
        history_view = self._update_parameters(requests, stream)
        self.mask_changed = False
        pending = []
        for index, (history, request, prefix) in enumerate(
            zip(history_view[: len(requests)], requests, prefixes, strict=True)
        ):
            if self.history_needed:
                if self.compact:
                    history[:] = request.presence
                else:
                    history[:] = request.counts
            constraint, grammar_prefix, thinking, thinking_count = self._prefix_state(
                history, request, prefix
            )
            generated = request.generated + len(prefix)
            guard = request.options.get("sampling", {}).get("repetition_guard", True)
            tokenizer = request.tokenizer
            budget = request.options.get("thinking_token_budget")
            key = (
                tokenizer.valid_tokens,
                constraint.tokenizer.vocab_size,
                tuple(sorted(tokenizer.eos)),
                tokenizer.think_end,
                thinking,
                thinking
                and budget is not None
                and budget >= 0
                and thinking_count >= budget,
                thinking or generated < request.minimum,
                guard,
                (constraint.cache_identity, request.generated, tuple(prefix))
                if constraint.matcher is not None or guard
                else None,
            )
            if key != self.mask_keys[index]:
                self.mask_changed = True
                self.mask_keys[index] = None
                if key in request.sample_masks:
                    self.mask_view[index] = np.frombuffer(
                        request.sample_masks[key], dtype=np.uint8
                    )
                    self.mask_keys[index] = key
                else:
                    pending.append(
                        (
                            index,
                            request,
                            constraint,
                            thinking,
                            generated,
                            thinking_count,
                            prefix,
                            grammar_prefix,
                            key,
                        )
                    )
            seed = request.seed ^ (0x9E3779B97F4A7C15 if stream == "proposal" else 0)
            self.random_view[index, 0] = seed
            self.random_view[index, 1] = getattr(request, self.field)
        self._fill_masks(requests, pending)

    def _update_parameters(self, requests, stream):
        parameters = {}
        for request in requests:
            if id(request) not in parameters:
                parameters[id(request)] = request.parameters()
        if requests:
            count = len(requests)
            self.parameters_view[:count] = np.stack(
                [parameters[id(request)] for request in requests]
            )
            self.history_needed = bool(
                np.any(self.parameters_view[:count, 4:6] != 0)
                or np.any(self.parameters_view[:count, 6] != 1)
            )
            self.compact = bool(
                np.all(self.parameters_view[:count, 4] == 0)
                and np.all(self.parameters_view[:count, 6] == 1)
            )
        else:
            self.parameters_view[:] = 0
            self.random_view[:] = 0
            self.history_needed = False
            self.compact = True
        self.history = self.presence_history if self.compact else self.full_history
        self.field = "proposal_offset" if stream == "proposal" else "random_offset"
        history_view = (
            self.presence_history_view if self.compact else self.full_history_view
        )

        return history_view

    def _prefix_state(self, history, request, prefix):
        constraint = request.constraint
        grammar_prefix = []
        thinking, thinking_count = request.thinking, request.thinking_count
        for token in prefix:
            if self.history_needed:
                if self.compact:
                    history[token // 8] |= 1 << (token % 8)
                else:
                    history[1, token] += 1
            if thinking:
                thinking_count += 1
                if token == request.tokenizer.think_end:
                    thinking = False
            else:
                grammar_prefix.append(token)

        return constraint, grammar_prefix, thinking, thinking_count

    def _fill_masks(self, requests, pending):
        packed_grammar = len({id(request) for request in requests}) >= 2
        constrained = [
            item for item in pending if not item[3] and item[2].matcher is not None
        ]
        masks = (
            iter(
                batch_masks(
                    [item[2] for item in constrained],
                    [item[7] for item in constrained],
                    **({"packed": True} if packed_grammar else {}),
                )
            )
            if constrained
            else iter(())
        )
        for (
            index,
            request,
            constraint,
            thinking,
            generated,
            thinking_count,
            prefix,
            _grammar_prefix,
            key,
        ) in pending:
            mask = (
                next(masks) if not thinking and constraint.matcher is not None else None
            )
            if packed_grammar and mask is not None:
                packed = request.packed_mask(
                    mask, thinking, generated, thinking_count, prefix
                )
            else:
                mask = request.mask(
                    constraint, thinking, generated, thinking_count, prefix, mask
                )
                packed = _pack_mask(mask)
            self.mask_view[index] = packed
            request.sample_masks[key] = packed.tobytes()
            self.mask_keys[index] = key


def _pack_mask(mask: np.ndarray):
    """Pack a one-dimensional uint8 NumPy mask in little-endian bit order."""
    if mask.size % 8:
        raise ValueError("sampling masks must contain a multiple of eight tokens")
    return np.packbits(mask, bitorder="little")


class Sampling:
    def __init__(self, device, weights, rows, *, hidden=None, operation=None):
        self.device = device
        self.weights = weights
        allocate = device.scratch
        self.operation = operation or prepare(
            make_plan(rows=rows), device, (weights,), allocate=allocate
        )
        self.token_rows = device.view(self.operation.outputs.tokens, (rows, 1))
        self.value, self.residual = (
            hidden
            if hidden is not None
            else (
                allocate((rows, 5120), "bfloat16"),
                allocate((rows, 5120), "float32"),
            )
        )
        self.projection_inputs = output.Inputs(self.value, self.residual)
        self.indices = allocate((rows,), "int32")
        self.history = device.zero(allocate((rows, 2, 248320), "int32"))
        self.presence_history = device.zero(allocate((rows, 248320 // 8), "uint8"))
        self.mask = allocate((rows, 248320 // 8), "uint8")
        self.parameters = allocate((rows, 7), "float32")
        self.rng = allocate((rows, 2), "uint64")
        self.inputs = SamplingInputs(rows)
        self.uploads = device.uploads()
        self.resources = ExitStack()
        self.greedy = False
        self.replays = {}
        self.last_indices = None
        self.packed_inputs = {}
        self.packed_input_key = None
        self.separate_inputs = (
            self.presence_history,
            self.history,
            self.mask,
            self.parameters,
            self.rng,
        )

    def close(self):
        self.resources.close()
        self.uploads.close()
        for inputs in self.packed_inputs.values():
            inputs.close()
        self.operation.close()

    def project(self):
        self.operation.project(self.weights, self.projection_inputs)

    def settings(self):
        return output.Settings(
            self.presence_history if self.inputs.compact else self.history,
            self.mask,
            self.parameters,
            self.greedy,
        )

    def draw(self):
        return self.operation.draw(self.settings(), self.rng)

    def distribution(self):
        return self.operation.distribution(self.settings())

    def execute(self, key, operation, graph):
        if graph:
            if key not in self.replays:
                self.replays[key] = self.resources.enter_context(
                    self.device.graph(operation)
                )
            self.replays[key]()
        else:
            operation()

    def select(
        self,
        value,
        residual,
        indices,
        requests,
        graph,
        **options,
    ):
        d = self.device
        if self.last_indices != indices:
            self.uploads.copy(self.indices, torch.as_tensor(indices, dtype=torch.int32))
            self.last_indices = list(indices)
        d.module.gather_rows(self.value, value, self.indices)
        d.module.gather_rows(self.residual, residual, self.indices)
        self.execute("project", self.project, graph)
        return self.select_from_logits(requests, graph, **options)

    def select_hidden(self, hidden, requests, graph):
        self.device.copy(self.operation.outputs.hidden, hidden)
        self.execute(
            "project_hidden",
            lambda: self.operation.project_hidden(
                self.weights, self.operation.outputs.hidden
            ),
            graph,
        )
        return self.select_from_logits(requests, graph)

    def upload_packed(self, select_token):
        values = (
            self.inputs.presence_history_view
            if self.inputs.compact
            else self.inputs.full_history_view,
            self.inputs.mask_view,
            self.inputs.parameters_view,
            self.inputs.random_view,
        )
        key = (self.inputs.compact, self.inputs.history_needed, select_token)
        if key != self.packed_input_key:
            self.resources.close()
            self.replays.clear()
            if key not in self.packed_inputs:
                self.packed_inputs[key] = Metadata(
                    self.device, values, single_upload=True
                )
            else:
                self.packed_inputs[key].update(values)
            inputs = self.packed_inputs[key]
            history, self.mask, self.parameters, self.rng = inputs.arrays
            if self.inputs.compact:
                self.presence_history = history
            else:
                self.history = history
            self.packed_input_key = key
        else:
            self.packed_inputs[key].update(values)

    def restore_inputs(self):
        self.resources.close()
        self.replays.clear()
        (self.presence_history, self.history, self.mask, self.parameters, self.rng) = (
            self.separate_inputs
        )
        self.packed_input_key = None
        self.inputs.mask_changed = True

    def select_from_logits(
        self,
        requests,
        graph,
        *,
        prefixes=None,
        stream="target",
        sample=True,
        readback=None,
    ):
        prefixes = [[] for _ in requests] if prefixes is None else prefixes
        d = self.device
        self.inputs.update(requests, prefixes, stream)
        self.greedy = bool(np.all(self.inputs.parameters_view[: len(requests), 0] == 0))
        select_token = sample or self.greedy
        if len({id(request) for request in requests}) >= 2:
            self.upload_packed(select_token)
        else:
            if self.packed_input_key is not None:
                self.restore_inputs()
            if self.inputs.history_needed:
                self.uploads.copy(
                    self.presence_history if self.inputs.compact else self.history,
                    self.inputs.history,
                )
            if self.inputs.mask_changed:
                self.uploads.copy(self.mask, self.inputs.mask)
            self.uploads.copy(self.parameters, self.inputs.parameters)
            if select_token:
                self.uploads.copy(self.rng, self.inputs.random)
        self.execute(
            (self.greedy, self.inputs.compact, select_token),
            self.draw if select_token else self.distribution,
            graph,
        )
        tokens = None
        if sample:
            tokens = (
                d.download(self.operation.outputs.tokens, None)
                if readback is None
                else readback(self.operation.outputs.tokens)
            )
            counters = [
                int(offset) + int(temperature > 0)
                for offset, temperature in zip(
                    self.inputs.random_view[:, 1],
                    self.inputs.parameters_view[:, 0],
                    strict=True,
                )
            ]
            for request, counter in zip(requests, counters, strict=True):
                setattr(request, self.inputs.field, int(counter))
        return tokens, self.operation.outputs.probabilities
