"""Separate CUDA graph lifecycles for DSpark context writes and proposals."""

import logging

import torch

from oh_my_vllm.worker.batch_plan import BLOCK

logger = logging.getLogger(__name__)


class DSparkContextGraph:
    """Capture small context commits; cache destinations stay transactional.

    The caller validates physical slots against its target page allocation.
    Compilation, warmup and capture can all write those slots. Save only these
    rows and restore them even after a failure; replay performs the real commit.
    Context graphs use a pool separate from proposal and target graphs.
    """

    MAX_ROWS = 32

    @staticmethod
    def _validate_inputs(features, positions, slots):
        if (
            features.ndim != 2
            or not 0 < len(features) <= DSparkContextGraph.MAX_ROWS
            or features.shape[1] <= 0
            or features.dtype != torch.bfloat16
            or not features.is_contiguous()
        ):
            raise ValueError("DSpark context graph requires 1..32 dense BF16 rows")
        for vector in (positions, slots):
            if (
                vector.shape != (len(features),)
                or vector.dtype != torch.int64
                or vector.device != features.device
                or not vector.is_contiguous()
            ):
                raise ValueError(
                    "DSpark context graph requires matching dense int64 metadata"
                )
        if features.device.type != "cuda":
            raise ValueError("DSpark context CUDA graphs require a CUDA device")

    @torch.inference_mode()
    def __init__(self, unit, caches, features, positions, slots, *, pool=None):
        self._validate_inputs(features, positions, slots)
        if not caches or any(
            cache.ndim != 5
            or cache.shape[1:3] != (2, BLOCK)
            or cache.dtype != torch.bfloat16
            or cache.device != features.device
            or not cache.is_contiguous()
            for cache in caches
        ):
            raise ValueError("DSpark context graph requires dense paged BF16 caches")
        self.caches = tuple(caches)
        self.features = features.clone()
        self.positions = positions.clone()
        self.slots = slots.clone()
        self.graph = torch.cuda.CUDAGraph()
        pages, offsets = slots // BLOCK, slots % BLOCK
        saved = [cache[pages, :, offsets].clone() for cache in self.caches]
        inputs = (self.features, self.positions, self.slots, self.caches)

        def restore():
            for cache, backup in zip(self.caches, saved, strict=True):
                cache[pages, :, offsets] = backup

        stream = torch.cuda.Stream(device=features.device)
        current = torch.cuda.current_stream(features.device)
        stream.wait_stream(current)
        try:
            try:
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        unit(*inputs)
            finally:
                current.wait_stream(stream)
            restore()
            torch.cuda.synchronize(features.device)
            logger.info(
                "Capturing CUDA graph: family=dspark_context key=%s rows=%d",
                (len(features),),
                len(features),
            )
            with torch.cuda.graph(self.graph, pool=pool):
                self.outputs = unit(*inputs)
        finally:
            restore()

    @torch.inference_mode()
    def replay(self, features, positions, slots):
        self._validate_inputs(features, positions, slots)
        inputs = (features, positions, slots)
        destinations = (self.features, self.positions, self.slots)
        if any(
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
            for destination, source in zip(destinations, inputs, strict=True)
        ):
            raise ValueError("DSpark context graph input shape/dtype/device changed")
        for destination, source in zip(destinations, inputs, strict=True):
            destination.copy_(source)
        self.graph.replay()
        return self.outputs


class DSparkGraph:
    @torch.inference_mode()
    def __init__(
        self, unit, caches, tokens, positions, tables, lengths, *, pool=None, key=None
    ):
        if tokens.device.type != "cuda":
            raise ValueError("DSpark CUDA graphs require a CUDA device")
        self.tokens = tokens.clone()
        self.positions = positions.clone()
        self.tables = tables.clone()
        self.lengths = lengths.clone()
        self.graph = torch.cuda.CUDAGraph()
        inputs = (self.tokens, self.positions, self.tables, self.lengths, caches)
        # Compilation and operator initialization finish outside stream capture.
        stream = torch.cuda.Stream(device=tokens.device)
        stream.wait_stream(torch.cuda.current_stream(tokens.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                unit(*inputs)
        torch.cuda.current_stream(tokens.device).wait_stream(stream)
        torch.cuda.synchronize(tokens.device)
        logger.info(
            "Capturing CUDA graph: family=dspark key=%s rows=%d",
            key,
            tokens.numel(),
        )
        with torch.cuda.graph(self.graph, pool=pool):
            self.outputs = unit(*inputs)

    def replay(self, tokens, positions, tables, lengths):
        for destination, source in zip(
            (self.tokens, self.positions, self.tables, self.lengths),
            (tokens, positions, tables, lengths),
            strict=True,
        ):
            if (
                destination.shape != source.shape
                or destination.dtype != source.dtype
                or destination.device != source.device
            ):
                raise ValueError("DSpark graph input shape/dtype/device changed")
            destination.copy_(source)
        self.graph.replay()
        # Outputs are consumed before another capture/replay in this family.
        return self.outputs
