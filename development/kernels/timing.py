"""CUDA Graph paired timings; fixture construction stays outside measurements."""

import torch


def reference_operator(module, name):
    """Bind the whole frozen operation, including Python-side copies/allocations."""
    from importlib import import_module

    path = f"oh_my_vllm.kernels.tilelang_reference.{module}"
    function = getattr(import_module(path), name)
    if function.__module__ != path:
        raise ValueError("comparison must bind the frozen operation directly")
    return function


def measure(reference, candidate, *, rounds=3, pairs=20, repeats=100):
    """Callables must use fixed inputs and repeatable, isolated destination state.

    Keep returned tensors alive during capture. The graph includes complete
    production operations, including required snapshots and reduction kernels.
    CUDA Event times exclude graph launch host overhead and fixture construction.
    """
    if min(rounds, pairs, repeats) <= 0:
        raise ValueError("positive measurement counts required")
    graphs = []
    outputs = []
    for function in (reference, candidate):
        for _ in range(10):
            output = function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                output = function()
        outputs.append(output)
        graphs.append(graph)
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
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                times[index] = start.elapsed_time(end) / repeats
            samples.append(times)
        results.append(samples)
    return results
