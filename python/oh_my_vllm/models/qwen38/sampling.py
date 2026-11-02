from contextlib import ExitStack

import torch

from ...operators.output import api as output
from ...operators.output.plan import make_plan
from ...operators.output.workspace import prepare
from ...worker.constraints import batch_masks


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

    def update(self, requests, prefixes, stream):
        parameters = {}
        for request in requests:
            if id(request) not in parameters:
                parameters[id(request)] = request.parameters()
        if requests:
            self.parameters.copy_(
                torch.stack(
                    [
                        torch.as_tensor(parameters[id(request)], dtype=torch.float32)
                        for request in requests
                    ]
                )
            )
            self.history_needed = bool(
                torch.any(self.parameters[:, 4:6] != 0)
                or torch.any(self.parameters[:, 6] != 1)
            )
            self.compact = bool(
                torch.all(self.parameters[:, 4] == 0)
                and torch.all(self.parameters[:, 6] == 1)
            )
        else:
            self.parameters.zero_()
            self.random.zero_()
            self.history_needed = False
            self.compact = True
        self.history = self.presence_history if self.compact else self.full_history
        self.field = "proposal_offset" if stream == "proposal" else "random_offset"
        # Torch rejects a uint64 scalar above the int64 maximum.
        seeds = self.random.view(torch.int64)[:, 0]
        self.mask_changed = False
        pending = []
        for index, (history, request, prefix) in enumerate(
            zip(self.history[: len(requests)], requests, prefixes, strict=True)
        ):
            if self.history_needed:
                if self.compact:
                    history.copy_(torch.as_tensor(request.presence, dtype=torch.uint8))
                else:
                    history.copy_(torch.as_tensor(request.counts, dtype=torch.int32))
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
                    cached = torch.frombuffer(
                        bytearray(request.sample_masks[key]), dtype=torch.uint8
                    )
                    self.mask[index].copy_(cached)
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
            seeds[index] = seed if seed < 1 << 63 else seed - (1 << 64)
            self.random[index, 1] = getattr(request, self.field)
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
            packed = _pack_mask(torch.as_tensor(mask, dtype=torch.uint8))
            self.mask[index].copy_(packed)
            request.sample_masks[key] = self.mask[index].numpy().tobytes()
            self.mask_keys[index] = key


def _pack_mask(mask):
    if mask.numel() % 8:
        raise ValueError("sampling masks must contain a multiple of eight tokens")
    mask = mask.reshape(-1, 8)
    return (
        mask[:, 0]
        | (mask[:, 1] << 1)
        | (mask[:, 2] << 2)
        | (mask[:, 3] << 3)
        | (mask[:, 4] << 4)
        | (mask[:, 5] << 5)
        | (mask[:, 6] << 6)
        | (mask[:, 7] << 7)
    )


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
        self.uploads = device.uploads()
        self.resources = ExitStack()
        self.greedy = False
        self.replays = {}
        self.last_indices = None

    def close(self):
        self.resources.close()
        self.uploads.close()
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

    def select_from_logits(
        self, requests, graph, *, prefixes=None, stream="target", sample=True
    ):
        prefixes = [[] for _ in requests] if prefixes is None else prefixes
        d = self.device
        self.inputs.update(requests, prefixes, stream)
        self.greedy = bool(torch.all(self.inputs.parameters[:, 0] == 0))
        if self.inputs.history_needed:
            self.uploads.copy(
                self.presence_history if self.inputs.compact else self.history,
                self.inputs.history,
            )
        if self.inputs.mask_changed:
            self.uploads.copy(self.mask, self.inputs.mask)
        self.uploads.copy(self.parameters, self.inputs.parameters)
        select_token = sample or self.greedy
        if select_token:
            self.uploads.copy(self.rng, self.inputs.random)
        self.execute(
            (self.greedy, self.inputs.compact, select_token),
            self.draw if select_token else self.distribution,
            graph,
        )
        tokens = None
        if sample:
            tokens = d.download(self.operation.outputs.tokens, None)
            counters = [
                int(offset) + int(temperature > 0)
                for offset, temperature in zip(
                    self.inputs.random[:, 1], self.inputs.parameters[:, 0], strict=True
                )
            ]
            for request, counter in zip(requests, counters, strict=True):
                setattr(request, self.inputs.field, int(counter))
        return tokens, self.operation.outputs.probabilities
