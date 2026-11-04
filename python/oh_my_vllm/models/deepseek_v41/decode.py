"""Fixed device inputs for one decode step.

A captured decode graph reads the token, position and cache slots from device
buffers. The caller binds them, replays the graph, then samples on the host.
"""

import math
import operator
import os
import sys
import traceback
from contextlib import contextmanager, nullcontext

import torch

from .decode_state import GraphStateExports, redirect_state_exports
from .kernel import (
    capture_cached_tensors,
    native_sparse_graph_enabled,
    require_sparse_status,
)
from .model import Attention, context_token_bucket, sample


class DecodeBindings:
    """Reusable device inputs and cache slots for one decode batch shape."""

    def __init__(self, batch: int, cache_slots, device, *, length=1, varlen=False):
        slots = tuple(operator.index(slot) for slot in cache_slots)
        if (
            batch < 1
            or length < 1
            or len(slots) != batch
            or len(set(slots)) != len(slots)
            or any(slot < 0 for slot in slots)
        ):
            raise ValueError("invalid DeepSeek decode cache slots")
        self.batch, self.length, self.varlen = batch, length, varlen
        self.cache_slots_cpu = slots
        self.cache_slots = torch.tensor(slots, dtype=torch.int64, device=device)
        positions = batch if varlen else 1
        self.storage = torch.zeros(
            batch * length + positions, dtype=torch.int64, device=device
        )
        self.token_ids = self.storage[: batch * length].view(batch, length)
        self.position = self.storage[batch * length :].view(
            (batch, 1) if varlen else ()
        )
        self.position_cpu = None
        self.positions_cpu = None
        self.valid_rows = torch.ones(batch * length, dtype=torch.bool, device=device)
        self.credited = torch.empty((batch, length), dtype=torch.bool, device=device)
        self.upload_device = self.uploads = None
        if varlen and self.storage.is_cuda:
            from ...operators.runtime.device import Device

            self.upload_device = Device()
            self.uploads = self.upload_device.uploads()

    def bind(self, token_ids, position):
        """Copy host metadata before replay into the fixed device inputs."""
        if not self.varlen:
            cursor = operator.index(position)
            if cursor < 0:
                raise ValueError("decode position must be nonnegative")
            tokens = torch.as_tensor(
                token_ids, dtype=torch.int64, device=self.token_ids.device
            )
            self.token_ids.copy_(tokens.reshape(self.batch, self.length))
            self.position.fill_(cursor)
            self.position_cpu = cursor
            self.positions_cpu = (cursor,) * self.batch
            return
        positions = tuple(operator.index(value) for value in position)
        if len(positions) != self.batch or any(value < 0 for value in positions):
            raise ValueError("decode positions must cover every request")
        self.position_cpu = max(positions)
        self.positions_cpu = positions
        position_tensor = torch.tensor(positions, dtype=torch.int64, device="cpu")
        if isinstance(token_ids, torch.Tensor) and token_ids.is_cuda:
            self.token_ids.copy_(token_ids.reshape(self.batch, self.length))
            self.uploads.copy(self.position, position_tensor.reshape_as(self.position))
            return
        tokens = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu").reshape(-1)
        if tokens.numel() != self.batch * self.length:
            raise ValueError("decode tokens must cover the prepared batch shape")
        source = torch.cat((tokens, position_tensor))
        if self.uploads is None:
            self.storage.copy_(source)
        else:
            self.uploads.copy(self.storage, source)

    def bind_valid_rows(self, valid_rows):
        if valid_rows is None:
            self.valid_rows.fill_(True)
        else:
            if (
                valid_rows.shape != self.valid_rows.shape
                or valid_rows.dtype != torch.bool
                or valid_rows.device != self.valid_rows.device
            ):
                raise ValueError("valid rows must be one device boolean per target row")
            self.valid_rows.copy_(valid_rows)

    def credit_rows(self, counts):
        if len(counts) != self.batch or any(
            not 0 <= value <= self.length for value in counts
        ):
            raise ValueError("credited counts differ from the prepared batch")
        source = (
            torch.arange(self.length, device="cpu")[None, :]
            < torch.tensor(counts, device="cpu")[:, None]
        )
        if self.uploads is None:
            self.credited.copy_(source)
        else:
            self.uploads.copy(self.credited, source)
        return self.credited.flatten()

    def close(self):
        if self.uploads is not None:
            self.uploads.close()
        if self.upload_device is not None:
            self.upload_device.close()


def native_sparse_graph_eligible(model):
    if not native_sparse_graph_enabled():
        return False
    for layer in getattr(model, "layers", ()):
        attention = getattr(layer, "attn", None)
        if not isinstance(attention, Attention):
            continue
        window = attention.window_kv_cache
        if (
            attention.head_dim != 512
            or attention.n_local_heads not in (16, 64)
            or not window.is_cuda
            or window.dtype != torch.bfloat16
            or not window.is_contiguous()
        ):
            return False
    return True


def capture_decode_graph(graph, rank):
    try:
        graph.capture()
    except Exception as error:
        message = f"DeepSeek graph capture failed on rank {rank}: {error!r}"
        try:
            print(message, file=sys.stderr, flush=True)
            trace_path = os.environ.get("OH_MY_VLLM_DEEPSEEK_GRAPH_TRACE")
            if trace_path:
                with open(
                    trace_path.replace("{rank}", str(rank)), "a", encoding="utf-8"
                ) as output:
                    output.write(f"{message}\n{traceback.format_exc()}\n")
        finally:
            os._exit(1)


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
    saved = export.steps, export.batch, getattr(export, "positions", None)
    export.steps = {}
    try:
        yield
    finally:
        export.steps, export.batch, export.positions = saved


def retained_graph_resources(model):
    """Hold the model buffers that a captured step addresses.

    The mHC workspace grows after a larger prefill chunk, and a routed expert
    rebind can drop the prepared weights. A graph keeps its capture-time
    buffers alive, so its recorded addresses stay valid. Replay separately
    checks that the expert binding generation has not changed.
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
    cache = getattr(model, "_physical_cache", None)
    if cache is not None and cache.page_table is not None:
        retained.append(cache.page_table.values)
        for history in cache.histories:
            retained.extend(
                value
                for value in (
                    history.codes,
                    history.scales,
                    getattr(history, "values", None),
                )
                if value is not None
            )
        if cache.index_workspace is not None:
            retained.append(cache.index_workspace.values)
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
        full_logits=False,
        segmented=False,
        routing_trace=None,
        plan=None,
        dispatch_combine=None,
    ):
        self.plan = plan
        self.dispatch_combine = dispatch_combine
        self.model = model
        self.bindings = bindings
        self.start_pos = start_pos
        self.collect_main_hidden = collect_main_hidden
        self.engram_values = engram_values
        self.full_logits = full_logits
        self.segmented = segmented
        self.routing_trace = routing_trace
        self.state_exports = GraphStateExports(bindings, start_pos)
        self.cycle = decode_cycle(model)
        self.residue = start_pos % self.cycle
        self.context_bucket = context_token_bucket(start_pos + bindings.length)
        self.window_prefixes = (
            tuple(
                (layer.attn.window_size, min(start_pos, layer.attn.window_size))
                for layer in model.layers
            )
            if bindings.length > 1
            else ()
        )
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
        self.routed_bindings = ()
        self.cache_lease = None
        self.closed = False

    def reserve(self):
        """Reserve paged cache rows through the end of the captured bucket."""
        self.model.reserve_cache(
            list(self.bindings.cache_slots_cpu), self.context_bucket
        )

    def _forward(self):
        context = (
            redirect_state_exports(self.model, self.state_exports)
            if self.segmented
            else nullcontext()
        )
        with context:
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
                full_logits=self.full_logits,
                logical_rows=1 if self.segmented else None,
                cache_positions_cpu=(
                    self.bindings.positions_cpu if self.bindings.varlen else None
                ),
                routing_trace=self.routing_trace,
                execution_plan=self.plan,
                valid_rows=self.bindings.valid_rows,
                dispatch_combine=self.dispatch_combine,
            )

    def _saved_recurrent_state(self):
        if self.bindings.length == 1:
            return ()
        values = []
        for layer in self.model.layers:
            attention = layer.attn
            values.append(attention.window_kv_cache)
            compressor = getattr(attention, "compressor", None)
            if compressor is not None and compressor.compress_ratio > 1:
                values.extend((compressor.kv_state, compressor.score_state))
        return tuple(
            (value, value.index_select(0, self.bindings.cache_slots).clone())
            for value in values
        )

    def capture(self):
        if self.closed or self.cache_lease is not None or self.output is not None:
            raise RuntimeError("capture a DeepSeek decode graph only once")
        if not native_sparse_graph_eligible(self.model):
            raise RuntimeError(
                "DeepSeek reference sparse attention requires Eager execution"
            )
        cache = getattr(self.model, "_physical_cache", None)
        if cache is not None and cache.page_table is not None:
            self.cache_lease = cache.page_table.acquire()
        try:
            return self._capture()
        except BaseException:
            self.close()
            raise

    def _capture(self):
        """Run the operator warmups, then record the step."""
        if self.bindings.position_cpu != self.start_pos:
            raise ValueError("bind the capture position before recording")
        self._validate_positions()
        current = torch.cuda.current_stream()
        with pause_state_export(self.model):
            saved = self._saved_recurrent_state()
            torch.cuda.synchronize()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    for value, previous in saved:
                        value.index_copy_(0, self.bindings.cache_slots, previous)
                    self._forward()
                for value, previous in saved:
                    value.index_copy_(0, self.bindings.cache_slots, previous)
            current.wait_stream(self.stream)
            torch.cuda.synchronize()
            with (
                capture_cached_tensors() as cached,
                torch.cuda.graph(self.graph, stream=self.stream),
            ):
                self.output = self._forward()
        self.retained = (*retained_graph_resources(self.model), *cached.values())
        self.routed_bindings = tuple(
            (layer, state, state.generation)
            for layer in self.model.layers
            if (state := getattr(layer.ffn, "_routed_state", None)) is not None
        )
        torch.cuda.synchronize()
        return self

    @property
    def logits(self):
        return self.output[1]

    @property
    def main_hidden(self):
        return self.output[2]

    def _validate_positions(self):
        positions = self.bindings.positions_cpu
        if positions is None:
            raise ValueError("bind the decode position before replay")
        if any(
            context_token_bucket(value + self.bindings.length) != self.context_bucket
            or value % self.cycle != self.residue
            or any(
                min(value, window) != prefix for window, prefix in self.window_prefixes
            )
            for value in positions
        ):
            raise ValueError("decode position is outside the captured graph")

    def replay(self):
        if self.closed:
            raise RuntimeError("DeepSeek decode graph is closed")
        if self.cache_lease is not None:
            self.cache_lease.validate()
        if any(
            getattr(layer.ffn, "_routed_state", None) is not state
            or state.generation != generation
            for layer, state, generation in self.routed_bindings
        ):
            raise RuntimeError("DeepSeek graph expert bindings have changed")
        self._validate_positions()
        self.graph.replay()
        require_sparse_status()
        export = getattr(self.model, "_state_export", None)
        if self.segmented and export is not None:
            self.state_exports.publish(export)

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.bindings.position.device)
        self.graph.reset()
        self.output = None
        self.retained = ()
        self.routed_bindings = ()
        if self.cache_lease is not None:
            self.cache_lease.close()
            self.cache_lease = None
        self.closed = True


class EngramStage:
    """Runs the host engram lookup outside a captured step.

    The lookup reads a file-backed host table, so it cannot run during capture.
    `stage` writes the selected embedding rows into fixed device buffers that the
    graph reads. Call it before each replay; decode tokens are text, so no token
    mask applies.
    """

    def __init__(self, model, batch, *, length=1):
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
                (batch, length, engram.n_hash_cols * engram.head_dim),
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
