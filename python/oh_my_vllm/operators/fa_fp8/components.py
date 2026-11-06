from typing import NamedTuple


class Storage(NamedTuple):
    value: object
    residual: object
    quantized: object
    scales: object
    query_projection: object
    kv_projection: object
    query: object
    gate: object
    attention_partials: object
    attention_sums: object
    output_quantized: object
    output_scales: object
    projection_partials: object
