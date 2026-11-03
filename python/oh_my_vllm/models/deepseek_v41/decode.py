"""Fixed device inputs for one decode step.

A captured decode graph reads the token, position and cache slots from device
buffers. The caller binds them, replays the graph, then samples on the host.
"""

import math
import operator
from contextlib import contextmanager

import torch

from .kernel import require_sparse_status
from .model import context_token_bucket, sample


class DecodeBindings:
    """Reusable device inputs and cache slots for one decode batch shape."""

    def __init__(self, batch: int, cache_slots, device):
        slots = tuple(operator.index(slot) for slot in cache_slots)
        if (
            len(slots) != batch
            or len(set(slots)) != len(slots)
            or any(slot < 0 for slot in slots)
        ):
            raise ValueError("invalid DeepSeek decode cache slots")
        self.batch = batch
        self.cache_slots_cpu = slots
        self.cache_slots = torch.tensor(slots, dtype=torch.int64, device=device)
        self.token_ids = torch.zeros((batch, 1), dtype=torch.int64, device=device)
        self.position = torch.zeros((), dtype=torch.int64, device=device)
        self.position_cpu = None

    def bind(self, token_ids, position):
        """Write the next token and its position into the fixed device buffers."""
        tokens = torch.as_tensor(
            token_ids, dtype=torch.int64, device=self.token_ids.device
        ).reshape(self.batch, 1)
        self.token_ids.copy_(tokens)
        self.position_cpu = operator.index(position)
        self.position.fill_(self.position_cpu)


def sampled_token_ids(logits, temperature=0.0):
    """Sample tokens from the graph logits and read them on the host."""
    return sample(logits, temperature).tolist()


def decode_cycle(model) -> int:
    """Position period of the decode branches that bake `start_pos % ratio`."""
    cycle = 1
    for layer in model.layers:
        ratio = getattr(layer.attn, "compress_ratio", 0)
        if ratio > 1:
            cycle = math.lcm(cycle, ratio)
    return cycle


@contextmanager
def pause_state_export(model):
    """Stop the host state snapshots while a step is recorded.

    The snapshots use the host position, and a graph must not repeat the
    capture-time indices. The serving path exports the state after replay.
    """
    export = getattr(model, "_state_export", None)
    if export is None:
        yield
        return
    saved = export.steps, export.batch
    export.steps = {}
    try:
        yield
    finally:
        export.steps, export.batch = saved


def retained_graph_resources(model):
    """Hold the model buffers that a captured step addresses.

    The mHC workspace grows after a larger prefill chunk, and a routed expert
    rebind can drop the prepared weights. A graph keeps its capture-time
    buffers alive, so its recorded addresses stay valid. Packed expert weights
    stay alive in the kernel weight cache, which no runtime path clears.
    """
    retained = []
    for layer in getattr(model, "layers", ()):
        workspace = getattr(layer, "_hc_workspace", None)
        if workspace is not None:
            retained.append(workspace)
        state = getattr(getattr(layer, "ffn", None), "_routed_state", None)
        if state is not None:
            retained.append(state.weights)
            retained.append(state.pointers)
    return tuple(retained)


class DecodeGraph:
    """One captured decode step for a fixed batch, context bucket and residue.

    The shapes and the compressor branch come from `start_pos` when the graph is
    recorded. Every replay must use a position in the same token bucket and with
    the same `start_pos % cycle` residue. The caller keys its graph cache with
    those values and the bound cache slots.
    The caller updates the bindings before replay. All ranks must record the same
    shapes in the same step.

    Paged sequence caches keep the recorded step from reserving rows. Call
    `reserve()` before capture when those caches are active.
    """

    def __init__(
        self,
        model,
        bindings,
        start_pos,
        *,
        collect_main_hidden=False,
        engram_values=None,
    ):
        self.model = model
        self.bindings = bindings
        self.start_pos = start_pos
        self.collect_main_hidden = collect_main_hidden
        self.engram_values = engram_values
        self.cycle = decode_cycle(model)
        self.residue = start_pos % self.cycle
        self.context_bucket = context_token_bucket(start_pos + 1)
        engram_layers = {
            index
            for index, layer in enumerate(model.layers)
            if layer.engram is not None
        }
        if engram_values is None and engram_layers:
            raise ValueError("captured decode needs staged engram values")
        if engram_values is not None and set(engram_values) != engram_layers:
            raise ValueError("staged engram values must cover every engram layer")
        self.stream = torch.cuda.Stream(device=bindings.position.device)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None
        self.retained = ()

    def reserve(self):
        """Reserve paged cache rows through the end of the captured bucket."""
        self.model.reserve_cache(
            list(self.bindings.cache_slots_cpu), self.context_bucket
        )

    def _forward(self):
        return self.model(
            self.bindings.token_ids,
            start_pos=self.start_pos,
            cache_slots=self.bindings.cache_slots,
            position=self.bindings.position,
            sample_output=False,
            cache_slots_cpu=self.bindings.cache_slots_cpu,
            collect_main_hidden=self.collect_main_hidden,
            check_status=False,
            engram_values=self.engram_values,
            reserve_cache=False,
        )

    def capture(self):
        """Run the operator warmups, then record the step."""
        if int(self.bindings.position) != self.start_pos:
            raise ValueError("bind the capture position before recording")
        current = torch.cuda.current_stream()
        with pause_state_export(self.model):
            torch.cuda.synchronize()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    self._forward()
            current.wait_stream(self.stream)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = self._forward()
        self.retained = retained_graph_resources(self.model)
        torch.cuda.synchronize()
        return self

    @property
    def logits(self):
        return self.output[1]

    @property
    def main_hidden(self):
        return self.output[2]

    def replay(self):
        position = self.bindings.position_cpu
        if position is None:
            raise ValueError("bind the decode position before replay")
        if (
            context_token_bucket(position + 1) != self.context_bucket
            or position % self.cycle != self.residue
        ):
            raise ValueError("decode position is outside the captured graph")
        self.graph.replay()
        require_sparse_status()


class EngramStage:
    """Runs the host engram lookup outside a captured step.

    The lookup reads a file-backed host table, so it cannot run during capture.
    `stage` writes the selected embedding rows into fixed device buffers that the
    graph reads. Call it before each replay; decode tokens are text, so no token
    mask applies.
    """

    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
        self.layers = {
            index: layer.engram
            for index, layer in enumerate(model.layers)
            if layer.engram is not None
        }
        if self.layers:
            device = next(iter(self.layers.values())).q_weight.device
        else:
            device = torch.get_default_device()
        self.buffers = {
            index: torch.zeros(
                (batch, 1, engram.n_hash_cols * engram.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            for index, engram in self.layers.items()
        }

    @torch.inference_mode()
    def stage(self, bindings, start_pos):
        """Refresh the staged rows for the bound token and position."""
        if bindings.batch != self.batch:
            raise ValueError("engram stage batch differs from the bindings")
        if not self.layers:
            return self.buffers
        hashes = self.model.engram_hash(
            bindings.token_ids,
            start_pos,
            None,
            bindings.cache_slots,
            position=bindings.position,
        )
        for index, engram in self.layers.items():
            ids = hashes[:, :, engram.layer_hash_index, :]
            self.buffers[index].copy_(engram.embed(ids).flatten(-2))
        return self.buffers
