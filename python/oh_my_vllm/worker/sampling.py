from contextlib import ExitStack

import numpy as np
import tvm_ffi

from ..operators.output import api as output
from ..operators.output.plan import make_plan
from ..operators.output.workspace import prepare
from .constraints import batch_masks


class SamplingInputs:
    def __init__(self, rows):
        self.full_history = np.empty((rows, 2, 248320), dtype=np.int32)
        self.presence_history = np.empty((rows, 248320 // 8), dtype=np.uint8)
        self.history = self.full_history
        self.compact = False
        self.mask = np.empty((rows, 248320 // 8), dtype=np.uint8)
        self.mask_keys = [None] * rows
        self.mask_changed = True
        self.parameters = np.empty((rows, 7), dtype=np.float32)
        self.random = np.empty((rows, 2), dtype=np.uint64)
        self.history_needed = False
        self.field = "random_offset"

    def update(self, requests, prefixes, stream):
        parameters = {}
        for request in requests:
            if id(request) not in parameters:
                parameters[id(request)] = request.parameters()
        self.parameters[:] = [parameters[id(request)] for request in requests]
        self.history_needed = bool(
            np.any(self.parameters[:, 4:6] != 0) or np.any(self.parameters[:, 6] != 1)
        )
        self.compact = bool(
            np.all(self.parameters[:, 4] == 0) and np.all(self.parameters[:, 6] == 1)
        )
        self.history = self.presence_history if self.compact else self.full_history
        self.field = "proposal_offset" if stream == "proposal" else "random_offset"
        self.mask_changed = False
        pending = []
        for index, (history, request, prefix) in enumerate(
            zip(self.history, requests, prefixes, strict=True)
        ):
            if self.history_needed:
                if self.compact:
                    history[:] = request.presence
                else:
                    history[:] = request.counts
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
                    self.mask[index] = np.frombuffer(
                        request.sample_masks[key], np.uint8
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
            self.random[index] = (seed, getattr(request, self.field))
        constrained = [
            item for item in pending if not item[3] and item[2].matcher is not None
        ]
        masks = (
            iter(
                batch_masks(
                    [item[2] for item in constrained], [item[7] for item in constrained]
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
            mask = request.mask(
                constraint, thinking, generated, thinking_count, prefix, mask
            )
            self.mask[index] = np.packbits(mask, bitorder="little")
            request.sample_masks[key] = self.mask[index].tobytes()
            self.mask_keys[index] = key


class Sampling:
    def __init__(self, device, weights, rows, *, hidden=None, operation=None):
        self.device = device
        self.weights = weights
        self.operation = operation or prepare(make_plan(rows=rows), device, (weights,))
        self.token_rows = device.view(self.operation.outputs.tokens, (rows, 1))
        self.value, self.residual = (
            hidden
            if hidden is not None
            else (
                device.empty((rows, 5120), "bfloat16"),
                device.empty((rows, 5120), "float32"),
            )
        )
        self.projection_inputs = output.Inputs(self.value, self.residual)
        self.indices = device.empty((rows,), "int32")
        self.history = device.zeros((rows, 2, 248320), "int32")
        self.presence_history = device.zeros((rows, 248320 // 8), "uint8")
        self.mask = device.empty((rows, 248320 // 8), "uint8")
        self.parameters = device.empty((rows, 7), "float32")
        self.rng = device.empty((rows, 2), "uint64")
        self.inputs = SamplingInputs(rows)
        self.resources = ExitStack()
        self.greedy = False
        self.replays = {}
        self.last_indices = None

    def close(self):
        self.resources.close()
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
            d.module.copy_bytes(
                self.indices, tvm_ffi.from_dlpack(np.asarray(indices, np.int32))
            )
            self.last_indices = list(indices)
        d.module.gather_rows(self.value, value, self.indices)
        d.module.gather_rows(self.residual, residual, self.indices)
        self.execute("project", self.project, graph)
        return self.select_from_logits(requests, graph, **options)

    def select_from_logits(
        self, requests, graph, *, prefixes=None, stream="target", sample=True
    ):
        prefixes = [[] for _ in requests] if prefixes is None else prefixes
        d = self.device
        self.inputs.update(requests, prefixes, stream)
        self.greedy = bool(np.all(self.inputs.parameters[:, 0] == 0))
        if self.inputs.history_needed:
            d.module.copy_bytes(
                self.presence_history if self.inputs.compact else self.history,
                tvm_ffi.from_dlpack(self.inputs.history),
            )
        if self.inputs.mask_changed:
            d.module.copy_bytes(self.mask, tvm_ffi.from_dlpack(self.inputs.mask))
        d.module.copy_bytes(
            self.parameters, tvm_ffi.from_dlpack(self.inputs.parameters)
        )
        select_token = sample or self.greedy
        if select_token:
            d.module.copy_bytes(self.rng, tvm_ffi.from_dlpack(self.inputs.random))
        self.execute(
            (self.greedy, self.inputs.compact, select_token),
            self.draw if select_token else self.distribution,
            graph,
        )
        tokens = None
        if sample:
            tokens = d.download(self.operation.outputs.tokens, np.int64)
            counters = self.inputs.random[:, 1] + (self.inputs.parameters[:, 0] > 0)
            for request, counter in zip(requests, counters, strict=True):
                setattr(request, self.inputs.field, int(counter))
        return tokens, self.operation.outputs.probabilities
