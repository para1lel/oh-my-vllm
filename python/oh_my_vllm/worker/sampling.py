from contextlib import ExitStack

import numpy as np
import tvm_ffi

from ..kernels.sample import Sample


class SamplingInputs:
    def __init__(self, rows):
        self.history = np.empty((rows, 2, 248320), dtype=np.int32)
        self.mask = np.empty((rows, 248320), dtype=np.uint8)
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
        self.field = "proposal_offset" if stream == "proposal" else "random_offset"
        self.mask_changed = False
        for index, (history, request, prefix) in enumerate(
            zip(self.history, requests, prefixes, strict=True)
        ):
            if self.history_needed:
                history[:] = request.counts
            constraint = request.constraint
            if constraint.matcher is not None:
                constraint = constraint.fork()
            thinking, thinking_count = request.thinking, request.thinking_count
            for token in prefix:
                if self.history_needed:
                    history[1, token] += 1
                if thinking:
                    thinking_count += 1
                    if token == request.tokenizer.think_end:
                        thinking = False
                else:
                    constraint.accept([token])
            generated = request.generated + len(prefix)
            key = None
            if constraint.matcher is None and not request.options.get(
                "sampling", {}
            ).get("repetition_guard", True):
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
                )
            if key is None or key != self.mask_keys[index]:
                self.mask[index] = request.mask(
                    constraint, thinking, generated, thinking_count, prefix
                )
                self.mask_changed = True
                self.mask_keys[index] = key
            seed = request.seed ^ (0x9E3779B97F4A7C15 if stream == "proposal" else 0)
            self.random[index] = (seed, getattr(request, self.field))


class Sampling:
    def __init__(self, device, weights, rows):
        self.device = device
        self.operation = Sample(weights, rows, device.empty)
        self.token_rows = device.view(self.operation.buffers[7], (rows, 1))
        self.value, self.residual = (
            device.empty((rows, 5120), "bfloat16") for _ in range(2)
        )
        self.indices = device.empty((rows,), "int32")
        self.history = device.zeros((rows, 2, 248320), "int32")
        self.mask = device.empty((rows, 248320), "uint8")
        self.parameters = device.empty((rows, 7), "float32")
        self.rng = device.empty((rows, 2), "uint64")
        self.inputs = SamplingInputs(rows)
        self.resources = ExitStack()
        self.greedy = False
        self.replays = {}

    def close(self):
        self.resources.close()

    def run(self):
        return self.operation(
            self.value,
            self.residual,
            self.history,
            self.mask,
            self.parameters,
            self.rng,
            greedy=self.greedy,
        )

    def select(
        self,
        value,
        residual,
        indices,
        requests,
        graph,
        *,
        prefixes=None,
        stream="target",
        sample=True,
    ):
        prefixes = [[] for _ in requests] if prefixes is None else prefixes
        self.inputs.update(requests, prefixes, stream)
        self.greedy = bool(np.all(self.inputs.parameters[:, 0] == 0))
        d = self.device
        if self.inputs.history_needed:
            d.module.copy_bytes(self.history, tvm_ffi.from_dlpack(self.inputs.history))
        if self.inputs.mask_changed:
            d.module.copy_bytes(self.mask, tvm_ffi.from_dlpack(self.inputs.mask))
        for destination, source in (
            (self.indices, np.asarray(indices, dtype=np.int32)),
            (self.parameters, self.inputs.parameters),
            (self.rng, self.inputs.random),
        ):
            d.module.copy_bytes(destination, tvm_ffi.from_dlpack(source))
        d.module.gather_rows(self.value, value, self.indices)
        d.module.gather_rows(self.residual, residual, self.indices)
        if graph:
            if self.greedy not in self.replays:
                self.replays[self.greedy] = self.resources.enter_context(
                    d.graph(self.run)
                )
            self.replays[self.greedy]()
        else:
            self.run()
        tokens = None
        if sample:
            tokens = d.download(self.operation.buffers[7], np.int64)
            counters = self.inputs.random[:, 1] + (self.inputs.parameters[:, 0] > 0)
            for request, counter in zip(requests, counters, strict=True):
                setattr(request, self.inputs.field, int(counter))
        return tokens, self.operation.buffers[6]
