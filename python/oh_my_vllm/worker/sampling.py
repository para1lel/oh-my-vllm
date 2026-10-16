from contextlib import ExitStack

import numpy as np
import tvm_ffi

from ..kernels.sample import Sample


class Sampling:
    def __init__(self, device, weights, rows):
        self.device = device
        self.operation = Sample(weights, rows, device.empty)
        self.value, self.residual = (
            device.empty((rows, 5120), "bfloat16") for _ in range(2)
        )
        self.indices = device.empty((rows,), "int32")
        self.history = device.empty((rows, 2, 248320), "int32")
        self.mask = device.empty((rows, 248320), "uint8")
        self.parameters = device.empty((rows, 7), "float32")
        self.rng = device.empty((rows, 2), "uint64")
        self.resources = ExitStack()
        self.replay = None

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
        histories, masks, parameters, random = [], [], [], []
        prefixes = [[] for _ in requests] if prefixes is None else prefixes
        for request, prefix in zip(requests, prefixes, strict=True):
            history = request.counts.copy()
            constraint = request.constraint.fork()
            thinking, thinking_count = request.thinking, request.thinking_count
            for token in prefix:
                history[1, token] += 1
                if thinking:
                    thinking_count += 1
                    if token == request.tokenizer.think_end:
                        thinking = False
                else:
                    constraint.accept([token])
            histories.append(history)
            masks.append(
                request.mask(
                    constraint,
                    thinking,
                    request.generated + len(prefix),
                    thinking_count,
                    prefix,
                )
            )
            parameters.append(request.parameters())
            field = "proposal_offset" if stream == "proposal" else "random_offset"
            seed = request.seed ^ (0x9E3779B97F4A7C15 if stream == "proposal" else 0)
            random.append((seed, getattr(request, field)))
        d = self.device
        random = np.asarray(random, dtype=np.uint64)
        for destination, source in (
            (self.indices, np.asarray(indices, dtype=np.int32)),
            (self.history, np.asarray(histories, dtype=np.int32)),
            (self.mask, np.asarray(masks, dtype=np.uint8)),
            (self.parameters, np.asarray(parameters, dtype=np.float32)),
            (self.rng, random),
        ):
            d.module.copy_bytes(destination, tvm_ffi.from_dlpack(source))
        d.module.gather_rows(self.value, value, self.indices)
        d.module.gather_rows(self.residual, residual, self.indices)
        if graph:
            if self.replay is None:
                self.run()
                d.module.copy_bytes(self.rng, tvm_ffi.from_dlpack(random))
                self.replay = self.resources.enter_context(d.graph(self.run))
            self.replay()
        else:
            self.run()
        tokens = d.download(self.operation.buffers[7], np.int64)
        if sample:
            counters = d.download(self.rng, np.uint64)
            for request, counter in zip(requests, counters[:, 1], strict=True):
                setattr(request, field, int(counter))
        return tokens, self.operation.buffers[6]
