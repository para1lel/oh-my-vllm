"""Rank-local attention projections and selected-row attention.

Encode first, stage private KV, then attend on the same stream. The caller
selects causal rows, supplies window/compressed RoPE coefficients, performs
collectives, and accepts state only after every status is zero. These methods
do not select or publish logical state. Retain inputs, including coefficients
and KV stores, through Eager completion or graph destruction.
"""

from typing import NamedTuple

from ...operators.linear import api as linear
from ...operators.linear.plan import make_plan as linear_plan
from ...operators.linear.workspace import prepare as prepare_linear
from ...operators.projection import api as projection
from ...operators.projection.plan import make_plan as projection_plan
from ...operators.projection.workspace import prepare as prepare_projection
from ...operators.quantization import api as quantization
from ...operators.quantization.plan import make_plan as quantization_plan
from ...operators.quantization.workspace import prepare as prepare_quantization
from ...operators.sparse_attention import api as sparse_attention
from ...operators.sparse_attention.plan import make_plan as attention_plan
from ...operators.sparse_attention.workspace import prepare as prepare_attention
from ...operators.transform import api as transform
from ...operators.transform.plan import make_plan as transform_plan
from ...operators.transform.workspace import prepare as prepare_transform
from .workspace import Workspace as RankWorkspace


class Encoded(NamedTuple):
    query: object
    query_latent: object
    window: object
    window_scales: object
    status: tuple


class Outputs(NamedTuple):
    value: object
    status: tuple


def _weights(owner, name):
    if owner.closed or owner._retained:
        raise RuntimeError("rank weights are closed or failed")
    if not isinstance(name, str):
        raise ValueError("attention requires a target or prediction layer name")
    family, _, index = name.partition(".")
    if family not in ("layers", "mtp") or index not in map(
        str, range(40 if family == "layers" else 3)
    ):
        raise ValueError("attention requires a target or prediction layer name")
    if (
        owner.plan.strategy not in ("TP4", "TEP4", "DEP4")
        or type(owner.plan.rank) is not int
        or owner.plan.rank not in range(4)
    ):
        raise ValueError("attention requires a four-rank placement")
    heads = 64 if owner.plan.strategy == "DEP4" else 16
    prefix = f"{name}.attn"
    encoded = {
        key: owner.packed[f"{prefix}.{key}.weight"]
        for key in ("wq_a", "wq_b", "wkv", "wo_b")
    }
    for key, (outputs, width) in {
        "wq_a": (1280, 5120),
        "wq_b": (heads * 512, 1280),
        "wkv": (512, 5120),
        "wo_b": (5120, heads // 8 * 1024),
    }.items():
        value = encoded[key]
        if (value.outputs, value.width, value.bits) != (outputs, width, 8):
            raise ValueError("attention projection differs from the rank plan")
    dense = {
        key: owner.tensors[f"{prefix}.{key}"]
        for key in ("q_norm.weight", "kv_norm.weight", "wo_a.weight", "attn_sink")
    }
    for key, shape, dtype in (
        ("q_norm.weight", (1280,), "bfloat16"),
        ("kv_norm.weight", (512,), "bfloat16"),
        ("wo_a.weight", (heads // 8 * 1024, 4096), "bfloat16"),
        ("attn_sink", (heads,), "float32"),
    ):
        if tuple(dense[key].shape) != shape or str(dense[key].dtype) != dtype:
            raise ValueError("attention dense weight differs from the rank plan")
    return heads, encoded, dense


class Workspace(RankWorkspace):
    def __init__(self, owner, name, rows, slots, allocate, byte_limit):
        heads, weights, self.weights = _weights(owner, name)
        self.plan = attention_plan(rows=rows, heads=heads, slots=slots)
        reads = [
            *self.weights.values(),
            *(
                tensor
                for weight in weights.values()
                for tensor in (weight.encoded, weight.scales)
            ),
        ]
        super().__init__(owner, reads, allocate, byte_limit)
        self._coefficients = None
        with self.preparing():
            self._prepare_encode(weights)
            first = len(self._allocations)
            self._prepare_output(weights)
            self._attend_writes = self.writes_since(first)

    def _quantize(self, width):
        return self.part(
            prepare_quantization,
            quantization_plan(rows=self.plan.rows, width=width, mode="fp8_ue8m0"),
            self.device,
        )

    def _project(self, weight):
        return self.part(
            prepare_projection,
            projection_plan(
                rows=self.plan.rows, outputs=weight.outputs, width=weight.width, bits=8
            ),
            self.device,
            weight,
        )

    def _transform(self, kind, width, **options):
        return self.part(
            prepare_transform,
            transform_plan(kind=kind, rows=self.plan.rows, width=width, **options),
            self.device,
        )

    def _prepare_encode(self, weights):
        rows, heads = self.plan.rows, self.plan.heads
        self.input_quantization = self._quantize(5120)
        self.query_a, self.window_projection = (
            self._project(weights["wq_a"]),
            self._project(weights["wkv"]),
        )
        self.query_norm, self.window_norm = (
            self._transform("rms", width, epsilon=1e-20) for width in (1280, 512)
        )
        self.query_quantization = self._quantize(1280)
        self.query_b = self._project(weights["wq_b"])
        self.query_rope, self.window_rope = (
            self._transform("rope", 512, heads=count) for count in (heads, 1)
        )
        self.window_quantization = self._quantize(512)
        self._query_heads = self.device.view(
            self.query_b.outputs.value, (rows, heads, 512)
        )
        self._window_head = self.device.view(
            self.window_norm.outputs.value, (rows, 1, 512)
        )
        self._rotated_window = self.device.view(
            self.window_rope.outputs.value, (rows, 512)
        )
        self.encoded = Encoded(
            self.query_rope.outputs.value,
            self.query_norm.outputs.value,
            self.window_quantization.outputs.encoded,
            self.window_quantization.outputs.scales,
            (
                self.input_quantization.outputs.status,
                self.query_quantization.outputs.status,
                self.window_quantization.outputs.status,
            ),
        )

    def _prepare_output(self, weights):
        rows, groups = self.plan.rows, self.plan.heads // 8
        self.attention = self.part(prepare_attention, self.plan, self.device)
        self.inverse_rope = self._transform(
            "rope", 512, heads=self.plan.heads, inverse=True
        )
        weight = self.device.view(self.weights["wo_a.weight"], (groups, 1024, 4096))
        self.output_a = self.part(
            prepare_linear,
            linear_plan(rows=rows, outputs=1024, width=4096, groups=groups),
            self.device,
            linear.Weights(weight),
        )
        self.output_quantization = self._quantize(groups * 1024)
        self.output_b = self._project(weights["wo_b"])
        self._output_groups = self.device.view(
            self.inverse_rope.outputs.value, (rows, groups, 4096)
        )
        self._output_latent = self.device.view(
            self.output_a.outputs.value, (rows, groups * 1024)
        )
        self.outputs = Outputs(
            self.output_b.outputs.value,
            (
                *self.encoded.status,
                self.attention.outputs.status,
                self.output_quantization.outputs.status,
            ),
        )

    def encode(self, value, coefficients):
        self.check_open()
        self._coefficients = None
        self.check_input(value, (self.plan.rows, 5120), "bfloat16")
        self.check_input(coefficients, (self.plan.rows, 32, 2), "float32")
        quantized = self.input_quantization.run(quantization.Inputs(value))
        inputs = projection.Inputs(quantized.encoded, quantized.scales)
        latent = self.query_a.run(inputs).value
        latent = self.query_norm.run(
            transform.Inputs(latent, self.weights["q_norm.weight"])
        ).value
        query = self.query_quantization.run(quantization.Inputs(latent))
        self.query_b.run(projection.Inputs(query.encoded, query.scales))
        self.query_rope.run(transform.Inputs(self._query_heads, coefficients))
        window = self.window_projection.run(inputs).value
        self.window_norm.run(transform.Inputs(window, self.weights["kv_norm.weight"]))
        self.window_rope.run(transform.Inputs(self._window_head, coefficients))
        self.window_quantization.run(quantization.Inputs(self._rotated_window))
        self._coefficients = coefficients
        return self.encoded

    def attend(self, indices, window, window_scales, compressed, compressed_scales):
        self.check_open()
        if self._coefficients is None:
            raise RuntimeError("encode before selected-row attention")
        if len(window.shape) != 2 or len(compressed.shape) != 2:
            raise ValueError("attention KV stores must be matrices")
        for value, shape, dtype in (
            (indices, (self.plan.rows, self.plan.slots), "int32"),
            (window, (window.shape[0], 512), "uint8"),
            (window_scales, (window.shape[0], 16), "uint8"),
            (compressed, (compressed.shape[0], 256), "uint8"),
            (compressed_scales, (compressed.shape[0], 32), "uint8"),
        ):
            self.check_input(value, shape, dtype, writes=self._attend_writes)
        attended = self.attention.run(
            sparse_attention.Inputs(
                self.encoded.query,
                self.weights["attn_sink"],
                indices,
                window,
                window_scales,
                compressed,
                compressed_scales,
            )
        ).value
        self.inverse_rope.run(transform.Inputs(attended, self._coefficients))
        self.output_a.run(linear.Inputs(self._output_groups))
        quantized = self.output_quantization.run(
            quantization.Inputs(self._output_latent)
        )
        self.output_b.run(projection.Inputs(quantized.encoded, quantized.scales))
        return self.outputs


def prepare(rank_weights, name, *, rows, slots, allocate=None, byte_limit=256 << 20):
    """Prepare fixed-row computation; KV tables and logical selection stay external."""
    return Workspace(rank_weights, name, rows, slots, allocate, byte_limit)
