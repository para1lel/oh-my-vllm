import numpy as np


def observe_cdf(prepared, operation):
    if prepared.case.options.get("greedy"):
        return np.tile([0.0, 1.0], (prepared.case.shape["rows"], 1))
    ordered = prepared.download(operation.buffers[5])
    probability = prepared.download(operation.buffers[6])
    selected = prepared.fixture.expected["tokens"]
    edges = []
    for order, mass, token in zip(ordered, probability, selected, strict=True):
        positions = np.flatnonzero(order == token)
        if len(positions) != 1:
            raise ValueError("native sampling order has no unique reference token")
        position = int(positions[0])
        cumulative = np.cumsum(mass[order], dtype=np.float64)
        edges.append(
            [cumulative[position - 1] if position else 0.0, cumulative[position]]
        )
    return np.asarray(edges, dtype=np.float64)
