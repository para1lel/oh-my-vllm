"""CUDA Graph paired timings; fixture construction stays outside measurements."""

import torch


def reference_operator(module, name):
    """Bind the whole frozen operation, including Python-side copies/allocations."""
    from importlib import import_module

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


def measure(reference, candidate, *, rounds=3, pairs=20, repeats=100):
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
