"""CUDA Graph paired timings; fixture construction stays outside measurements."""

import torch


def reference_operator(module, name):
    """Bind the whole frozen operation, including Python-side copies/allocations."""
    from importlib import import_module

    if (module, name) == ("fp8", "add_norm_linear"):
        # Compose original frozen kernels. The reference files and pin stay equal.
        normalization = reference_operator("normalization", "add_rms_norm")
        projection = reference_operator("fp8", "linear")

        def frozen_chain(x, residual, gamma, weight, scale):
            summed, normalized = normalization(x, residual, gamma)
            return summed, projection(normalized, weight, scale)

        return frozen_chain
    if module == "dspark_attention":
        path = "oh_my_vllm.kernels.dspark_tilelang_reference"
        function = getattr(import_module(path), name)
        if function.__module__ != path:
            raise ValueError("comparison must bind the supplemental operation directly")
        return function
    if (module, name) == ("attention_prepare", "prepare_attention"):
        # The production pre-fusion chain, composed only of immutable kernels.
        # Its V.contiguous() is a no-op when the original view is contiguous.
        from oh_my_vllm.kernels.attention_prepare import tilelang_prepare

        return tilelang_prepare
    path = f"oh_my_vllm.kernels.tilelang_reference.{module}"
    function = getattr(import_module(path), name)
    if function.__module__ != path:
        raise ValueError("comparison must bind the frozen operation directly")
    return function


def measure(
    reference, candidate, *, rounds=3, pairs=20, repeats=100, qk_inputs=None, audit=None
):
    """Callables use fixed inputs and repeatable destination state.

    DSpark append can share a destination after independent output verification:
    its immutable K/V and slots repeat the same writes. Other mutable fixtures
    keep isolated destination pools. Keep returned tensors alive during capture.
    The graph includes complete
    production operations, including required snapshots and reduction kernels.
    External timing events are captured around all operations in each graph.
    A replay orders both timestamps on the device, excluding host submission
    gaps before or after replay as well as fixture construction.
    """
    if qk_inputs is not None:
        if audit is None:
            raise ValueError("shared Q/K timing requires an audit destination")
        return _measure_shared_qk(
            reference,
            candidate,
            qk_inputs,
            rounds=rounds,
            pairs=pairs,
            repeats=repeats,
            audit=audit,
        )
    if min(rounds, pairs, repeats) <= 0:
        raise ValueError("positive measurement counts required")
    graphs = []
    outputs = []
    events = []
    for function in (reference, candidate):
        for _ in range(10):
            output = function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            start.record()
            for _ in range(repeats):
                output = function()
            end.record()
        outputs.append(output)
        graphs.append(graph)
        events.append((start, end))
    results = []
    for round_index in range(rounds):
        for graph in graphs:
            for _ in range(5):
                graph.replay()
        torch.cuda.synchronize()
        samples = []
        for pair_index in range(pairs):
            times = [None, None]
            order = (0, 1) if (round_index + pair_index) % 2 == 0 else (1, 0)
            for index in order:
                start, end = events[index]
                graphs[index].replay()
                end.synchronize()
                times[index] = start.elapsed_time(end) / repeats
            samples.append(times)
        results.append(samples)
    return results


def _storage_span(value):
    storage = value.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def _overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _qk_output_pointers(output, inputs, input_spans):
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("shared Q/K timing requires two returned tensors")
    if any(
        not isinstance(value, torch.Tensor)
        or value.shape != source.shape
        or value.dtype != source.dtype
        or value.device != source.device
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.untyped_storage().nbytes() != value.numel() * value.element_size()
        for value, source in zip(output, inputs, strict=True)
    ):
        raise ValueError("shared Q/K timing requires two full contiguous outputs")
    output_spans = [_storage_span(value) for value in output]
    if _overlap(*output_spans) or any(
        _overlap(destination, source)
        for destination in output_spans
        for source in input_spans
    ):
        raise ValueError("shared Q/K outputs must not alias each other or inputs")
    return [
        (value.data_ptr(), value.numel() * value.element_size()) for value in output
    ]


def _measure_shared_qk(reference, candidate, inputs, *, rounds, pairs, repeats, audit):
    """Time whole stateless Q/K functions with matching output addresses.

    Both graphs read immutable Q/K and write only their two unused outputs.
    PyTorch permits arbitrary serial replay for such output-independent graphs.
    Keep both graphs and their shared capture stream alive through final sync.
    Independent numerical verification is the caller's precondition.
    """
    if min(rounds, pairs, repeats) <= 0:
        raise ValueError("positive measurement counts required")
    if repeats < 2:
        raise ValueError("shared Q/K timing requires at least two repetitions")
    if (
        len(inputs) != 2
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bfloat16
            or not value.is_cuda
            or value.ndim != 3
            or value.shape[-1] != 128
            or value.numel() == 0
            for value in inputs
        )
        or inputs[0].shape != inputs[1].shape
        or inputs[0].device != inputs[1].device
    ):
        raise ValueError("shared Q/K timing requires matching BF16 CUDA head inputs")
    if inputs[0].device.index != torch.cuda.current_device():
        raise ValueError("shared Q/K inputs must use the current CUDA device")
    input_spans = [_storage_span(value) for value in inputs]
    guards = {}
    for value in inputs:
        span = _storage_span(value)
        if (span[1] - span[0]) % value.element_size():
            raise ValueError(
                "shared Q/K input storage must contain complete BF16 values"
            )
        if span not in guards:
            storage = value.as_strided(
                ((span[1] - span[0]) // value.element_size(),),
                (1,),
                storage_offset=0,
            )
            bytes_view = storage.view(torch.uint8)
            guards[span] = (bytes_view, bytes_view.clone())
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=inputs[0].device)
    graphs, events, pointers = [], [], []
    try:
        for function in (reference, candidate):
            for _ in range(10):
                output = function()
            torch.cuda.synchronize(device=inputs[0].device)
            del output
            graph = torch.cuda.CUDAGraph()
            graphs.append(graph)
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            sequence = []
            validation_error = None
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                start.record()
                for _ in range(repeats):
                    output = function()
                    try:
                        current = _qk_output_pointers(output, inputs, input_spans)
                        if sequence and any(
                            _overlap(
                                (address, address + size),
                                (old_address, old_address + old_size),
                            )
                            for address, size in current
                            for old_address, old_size in sequence[-1]
                        ):
                            raise ValueError(
                                "shared Q/K calls must allocate fresh outputs"
                            )
                    except ValueError as error:
                        validation_error = str(error)
                        break
                    sequence.append(current)
                end.record()
            if validation_error is not None:
                torch.cuda.synchronize(device=inputs[0].device)
                raise ValueError(validation_error)
            # A graph owns its pool without a live Python return tensor. Output
            # data is scratch: neither graph consumes the other graph's output.
            del output
            events.append((start, end))
            pointers.append(sequence)
        if pointers[0] != pointers[1]:
            raise ValueError("shared Q/K capture output pointer sequences differ")
        results = []
        for round_index in range(rounds):
            for graph in graphs:
                for _ in range(5):
                    graph.replay()
            torch.cuda.synchronize(device=inputs[0].device)
            samples = []
            for pair_index in range(pairs):
                times = [None, None]
                order = (0, 1) if (round_index + pair_index) % 2 == 0 else (1, 0)
                for index in order:
                    start, end = events[index]
                    graphs[index].replay()
                    end.synchronize()
                    times[index] = start.elapsed_time(end) / repeats
                samples.append(times)
            results.append(samples)
        torch.cuda.synchronize(device=inputs[0].device)
        if any(
            not torch.equal(storage, original) for storage, original in guards.values()
        ):
            raise ValueError("shared Q/K timing changed input storage")
        if audit is not None:
            import hashlib
            import json

            audit.update(
                matching_capture_output_addresses=True,
                checked_calls_per_backend=repeats,
                output_input_storage_disjoint=True,
                input_storage_unchanged=True,
                output_allocations=len(
                    {address for pair in pointers[0] for address, _ in pair}
                ),
                pointer_sequence_sha256=hashlib.sha256(
                    json.dumps(pointers[0], separators=(",", ":")).encode()
                ).hexdigest(),
            )
        return results
    finally:
        try:
            torch.cuda.synchronize(device=inputs[0].device)
        finally:
            errors = []
            for graph in graphs:
                try:
                    graph.reset()
                except RuntimeError as error:
                    errors.append(str(error))
            graphs.clear()
            if errors:
                raise RuntimeError(
                    "shared Q/K graph cleanup failed: " + "; ".join(errors)
                )
