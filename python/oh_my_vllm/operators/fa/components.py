from typing import NamedTuple


class Storage(NamedTuple):
    value: object
    residual: object
    quantized: object
    scales: object
    projection: object
    query: object
    gate: object
    attention_partials: object
    attention_sums: object
    output_quantized: object
    output_scales: object
    projection_partials: object


class ContextProjection(NamedTuple):
    residual: object
    quantized: object
    scales: object
    projection: object
    partials: object


def context_projection(workspace, device):
    plan, storage = workspace.plan, workspace.storage
    if plan.variant != 1 or plan.first_token != 1 or plan.max_queries != 4:
        raise ValueError("FA context projection requires four KV-only queries")
    return ContextProjection(
        storage.residual,
        storage.quantized,
        storage.scales,
        storage.projection,
        device.view(storage.attention_partials, (8, plan.rows, 5120)),
    )
