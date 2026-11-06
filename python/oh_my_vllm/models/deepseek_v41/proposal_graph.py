"""One request's complete DSpark proposal with fixed input shapes."""

import torch

from .decode import DecodeBindings
from .decode_state import GraphStateExports, redirect_state_exports
from .kernel import capture_cached_tensors, require_sparse_status


class ProposalGraph:
    def __init__(self, model, slot, main_hidden):
        if main_hidden.shape[:2] != (1, 1) or model.temperature != 0:
            raise ValueError("MTP Graph requires one greedy proposal row")
        self.model = model
        self.cache = getattr(model, "_physical_cache", None)
        self.bindings = DecodeBindings(1, (slot,), main_hidden.device)
        self.main_hidden = torch.empty_like(main_hidden)
        self.window_start = max(layer.attn.window_size - 1 for layer in model.mtp)
        self.max_position = min(
            layer.attn.freqs_cis.size(0) - layer.block_size - 1 for layer in model.mtp
        )
        self.state_exports = GraphStateExports(self.bindings, self.window_start)
        self.stream = torch.cuda.Stream(device=main_hidden.device)
        self.graph = torch.cuda.CUDAGraph()
        self.output = None
        self.retained = ()
        self.binding_owners = ()
        self.tensor_owners = ()
        self.routed_bindings = ()
        self.cache_lease = None
        self.closed = False

    def bind(self, input_ids, main_hidden, position):
        if self.closed:
            raise RuntimeError("DeepSeek proposal graph is closed")
        if not self.window_start <= position <= self.max_position:
            raise ValueError("MTP Graph position is outside the full window branch")
        if main_hidden.shape != self.main_hidden.shape or (
            main_hidden.dtype != self.main_hidden.dtype
            or main_hidden.device != self.main_hidden.device
        ):
            raise ValueError("MTP Graph hidden state binding has changed")
        self.bindings.bind(input_ids, position)
        self.main_hidden.copy_(main_hidden)

    def _forward(self):
        with redirect_state_exports(self.model, self.state_exports):
            return self.model.forward_spec(
                self.bindings.token_ids,
                self.main_hidden,
                start_pos=self.window_start,
                cache_slots=self.bindings.cache_slots,
                cache_slots_cpu=self.bindings.cache_slots_cpu,
                position=self.bindings.position,
                check_status=False,
            )

    def _retain_bindings(self):
        owners, tensors = [], []
        owners.append((self.model._modules, "mtp", self.model.mtp))
        for path, module in self.model.mtp.named_modules():
            # Expert changes already advance their prepared binding generation.
            if ".experts." in f".{path}.":
                continue
            owners.extend(
                (module._modules, name, value)
                for name, value in module._modules.items()
            )
            for storage in (module._parameters, module._buffers):
                tensors.extend(
                    (storage, name, value, value.data_ptr())
                    for name, value in storage.items()
                    if value is not None
                )
        self.binding_owners = tuple(owners)
        self.tensor_owners = tuple(tensors)
        self.routed_bindings = tuple(
            (layer.ffn, layer.ffn._routed_state, layer.ffn._routed_state.generation)
            for layer in self.model.mtp
        )
        self.retained += tuple(
            resource
            for layer in self.model.mtp
            for resource in (
                layer._hc_workspace,
                layer.ffn._routed_state.weights,
                layer.ffn._routed_state.pointers,
            )
        )

    def bindings_valid(self):
        if self.cache_lease is not None:
            try:
                self.cache_lease.validate()
            except RuntimeError:
                return False
        return (
            not self.closed
            and getattr(self.model, "_physical_cache", None) is self.cache
            and all(
                storage.get(name) is value
                for storage, name, value in self.binding_owners
            )
            and all(
                storage.get(name) is value and value.data_ptr() == pointer
                for storage, name, value, pointer in self.tensor_owners
            )
            and all(
                owner._routed_state is state and state.generation == generation
                for owner, state, generation in self.routed_bindings
            )
        )

    def capture(self):
        if self.closed or self.output is not None or self.cache_lease is not None:
            raise RuntimeError("capture a DeepSeek proposal graph only once")
        if self.bindings.position_cpu is None:
            raise ValueError("bind the MTP inputs before capture")
        try:
            table = getattr(self.cache, "page_table", None)
            if table is not None:
                self.cache_lease = table.acquire()
            self._capture()
        except BaseException:
            self.close()
            raise
        return self

    def _capture(self):
        current = torch.cuda.current_stream(self.main_hidden.device)
        saved = tuple(
            (
                layer.attn.window_kv_cache,
                layer.attn.window_kv_cache.index_select(
                    0, self.bindings.cache_slots
                ).clone(),
            )
            for layer in self.model.mtp
        )
        torch.cuda.synchronize(self.main_hidden.device)
        self.stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    for value, previous in saved:
                        value.index_copy_(0, self.bindings.cache_slots, previous)
                    self._forward()
                for value, previous in saved:
                    value.index_copy_(0, self.bindings.cache_slots, previous)
            current.wait_stream(self.stream)
            torch.cuda.synchronize(self.main_hidden.device)
            with (
                capture_cached_tensors() as cached,
                torch.cuda.graph(self.graph, stream=self.stream),
            ):
                self.output = self._forward()
            self.retained = tuple(cached.values())
            self._retain_bindings()
        finally:
            current.wait_stream(self.stream)
            for value, previous in saved:
                value.index_copy_(0, self.bindings.cache_slots, previous)
            torch.cuda.synchronize(self.main_hidden.device)

    def replay(self):
        if not self.bindings_valid():
            raise RuntimeError("DeepSeek proposal graph bindings have changed")
        if self.output is None:
            raise RuntimeError("capture the MTP Graph before replay")
        if self.cache_lease is not None:
            self.cache_lease.validate()
        self.graph.replay()
        require_sparse_status()
        export = getattr(self.model, "_state_export", None)
        if export is not None:
            self.state_exports.publish(export, preserve_existing=True)
        return self.output

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.main_hidden.device)
        self.graph.reset()
        self.output = None
        self.retained = self.binding_owners = self.tensor_owners = (
            self.routed_bindings
        ) = ()
        self.state_exports.windows.clear()
        self.state_exports.tails.clear()
        if self.cache_lease is not None:
            self.cache_lease.close()
            self.cache_lease = None
        self.bindings.close()
        self.model = self.cache = self.bindings = self.main_hidden = None
        self.state_exports = self.stream = None
        self.closed = True
