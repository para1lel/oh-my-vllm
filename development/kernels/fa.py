import numpy as np
from oh_my_vllm.kernels.fa import FA, Weights

from .build import build
from .ffn import reference_projection
from .reference import bf16, bf16_bits, numerical_error, quantize, rms, sigmoid


def unpack_bf16(device, tensor):
    return (device.download(tensor, np.uint16).astype(np.uint32) << 16).view(np.float32)


def rotary(value, positions):
    result = value.copy()
    frequency = np.arange(32)
    axes = np.zeros(32, dtype=np.int64)
    axes[1:33:3], axes[2:30:3] = 1, 2
    angles = (
        positions[axes].T.astype(np.float32)
        * np.power(np.float32(10000000), -frequency.astype(np.float32) / 32)[None]
    )
    cosine, sine = bf16(np.cos(angles))[:, None], bf16(np.sin(angles))[:, None]
    a, b = value[..., :32], value[..., 32:64]
    result[..., :32] = bf16(bf16(a * cosine) - bf16(b * sine))
    result[..., 32:64] = bf16(bf16(b * cosine) + bf16(a * sine))
    return result


class FAWeights:
    def __init__(self, device, checkpoint, *, prefix=None, variant="full"):
        if prefix is None:
            prefix = "model.language_model.layers.3."
        self.norm = checkpoint.tensor(prefix + "input_layernorm.weight")
        self.query_norm = checkpoint.tensor(prefix + "self_attn.q_norm.weight")
        self.key_norm = checkpoint.tensor(prefix + "self_attn.k_norm.weight")
        names = {"full": ("q", "k", "v"), "kv_only": ("k", "v"), "query_only": ("q",)}[
            variant
        ]
        self.projection = np.concatenate(
            [
                checkpoint.tensor(prefix + f"self_attn.{name}_proj.weight")
                for name in names
            ]
        )
        self.projection_scale = np.concatenate(
            [
                checkpoint.tensor(prefix + f"self_attn.{name}_proj.weight_scale_inv")
                for name in names
            ]
        )
        self.output = checkpoint.tensor(prefix + "self_attn.o_proj.weight")
        self.output_scale = checkpoint.tensor(
            prefix + "self_attn.o_proj.weight_scale_inv"
        )
        self.variant = variant
        self.device_weights = Weights(
            device.upload(self.norm),
            device.upload(self.projection, "float8_e4m3fn"),
            device.upload(self.projection_scale),
            device.upload(self.query_norm),
            device.upload(self.key_norm),
            device.upload(self.output, "float8_e4m3fn"),
            device.upload(self.output_scale),
        )


class FACase:
    def __init__(
        self, device, weights, shape, *, value=None, residual=None, first_token=0
    ):
        build("fa", ("fa",))
        self.device, self.weights, self.shape = device, weights, shape
        self.queries = shape.rows // shape.batch
        if self.queries * shape.batch != shape.rows:
            raise ValueError("FA evaluation needs balanced request rows")
        self.first_token = first_token
        self.context = max(shape.context, self.queries + first_token)
        rng = np.random.default_rng(shape.rows + self.context)
        self.x = (
            bf16(rng.normal(0, 0.2, (shape.rows, 5120))) if value is None else value
        )
        self.residual = (
            bf16(rng.normal(0, 1, (shape.rows, 5120))) if residual is None else residual
        )
        self.inputs = tuple(
            device.upload(bf16_bits(v), "bfloat16") for v in (self.x, self.residual)
        )
        pages = (self.context + 783) // 784
        order = rng.permutation(shape.batch * pages) + 1
        self.tables = order.astype(np.int32).reshape(shape.batch, pages)
        self.sequence_rows = np.arange(shape.batch + 1, dtype=np.int32) * self.queries
        self.lengths = np.tile(
            np.arange(self.context - self.queries + 1, self.context + 1), shape.batch
        ).astype(np.int32)
        logical = np.tile(
            np.arange(self.context - self.queries, self.context), shape.batch
        )
        requests = np.repeat(np.arange(shape.batch), self.queries)
        self.slots = (
            self.tables[requests, logical // 784] * 784 + logical % 784
        ).astype(np.int32)
        self.positions = np.stack(
            (logical, logical // 3 + 13, logical // 5 + 31)
        ).astype(np.int64)
        seed = bf16_bits(rng.normal(0, 0.05, (2, 784, 4, 256)))
        blocks = shape.batch * pages + 2
        self.cache_initial = np.empty((blocks, *seed.shape), dtype=np.uint16)
        seed_values = (seed.astype(np.uint32) << 16).view(np.float32)
        for page in range(blocks):
            self.cache_initial[page] = bf16_bits(
                seed_values * (1 + page / blocks) + page / (20 * blocks)
            )
        self.cache = device.upload(self.cache_initial, "bfloat16")
        self.metadata = tuple(
            device.upload(v)
            for v in (
                self.positions,
                self.slots,
                self.tables,
                self.sequence_rows,
                self.lengths,
            )
        )
        self.operation = FA(
            weights.device_weights,
            shape.rows,
            device.empty,
            variant=weights.variant,
            max_queries=self.queries,
            first_token=first_token,
        )

    def reset(self):
        import tvm_ffi

        self.device.module.copy_bytes(
            self.cache, tvm_ffi.from_dlpack(self.cache_initial)
        )

    def run(self):
        positions, slots, tables, sequences, lengths = self.metadata
        return self.operation(
            *self.inputs, positions, self.cache, slots, tables, sequences, lengths
        )

    def project(self, indices, *, kv=False):
        normalized = rms(
            bf16(self.x[indices] + self.residual[indices]), self.weights.norm
        )
        data, scales = quantize(normalized)
        first = 0 if self.weights.variant == "kv_only" else 12288 if kv else 0
        last = first + 2048 if kv else self.weights.projection.shape[0]
        return reference_projection(
            data,
            scales,
            self.weights.projection[first:last],
            self.weights.projection_scale[first // 128 : last // 128],
        )

    def check(self, *, execute=True):
        actual, summed = self.run() if execute else self.operation.buffers[:2]
        expected_sum = bf16(self.x + self.residual)
        actual_sum = unpack_bf16(self.device, summed)
        residual_check = numerical_error(actual_sum, expected_sum)
        if getattr(self, "exact_residual", True):
            residual_check["passed"] = bool(np.array_equal(actual_sum, expected_sum))
        cache_bits = self.device.download(self.cache, np.uint16)
        cache_values = (cache_bits.astype(np.uint32) << 16).view(np.float32)
        write_checks = []
        written = np.zeros((len(cache_bits), 784), dtype=bool)
        if self.weights.variant != "query_only":
            for first in range(0, self.shape.rows, 64):
                indices = np.arange(first, min(first + 64, self.shape.rows))
                packed = self.project(indices, kv=True).reshape(-1, 2, 4, 256)
                keys = rotary(
                    rms(packed[:, 0], self.weights.key_norm), self.positions[:, indices]
                )
                blocks, offsets = self.slots[indices] // 784, self.slots[indices] % 784
                write_checks.append(
                    numerical_error(cache_values[blocks, 0, offsets], keys, state=True)
                )
                write_checks.append(
                    numerical_error(
                        cache_values[blocks, 1, offsets], packed[:, 1], state=True
                    )
                )
                cache_values[blocks, 0, offsets] = keys
                cache_values[blocks, 1, offsets] = packed[:, 1]
                written[blocks, offsets] = True
        unchanged = all(
            np.array_equal(
                cache_bits[b, :, ~written[b]], self.cache_initial[b, :, ~written[b]]
            )
            for b in range(len(cache_bits))
        )
        indices = getattr(
            self,
            "check_indices",
            np.unique([0, self.shape.rows // 2, self.shape.rows - 1]),
        )
        output_check = {"passed": True}
        finite = True
        if self.weights.variant != "kv_only":
            packed = self.project(indices)
            query_gate = packed[:, :12288].reshape(-1, 24, 512)
            query = rotary(
                rms(query_gate[..., :256], self.weights.query_norm),
                self.positions[:, indices],
            )
            outputs = []
            for sampled, row in enumerate(indices):
                request, length = row // self.queries, self.lengths[row]
                positions = np.arange(self.first_token, length)
                pages, offsets = self.tables[request, positions // 784], positions % 784
                key, value = (
                    cache_values[pages, 0, offsets],
                    cache_values[pages, 1, offsets],
                )
                attended = np.empty((24, 256), dtype=np.float64)
                for head in range(24):
                    scores = (
                        key[:, head // 6].astype(np.float64) @ query[sampled, head] / 16
                    )
                    probability = np.exp(scores - np.max(scores))
                    probability /= probability.sum()
                    attended[head] = bf16(probability @ value[:, head // 6])
                outputs.append(
                    bf16(
                        attended * bf16(sigmoid(query_gate[sampled, :, 256:]))
                    ).reshape(6144)
                )
            data, scales = quantize(np.stack(outputs))
            expected = reference_projection(
                data, scales, self.weights.output, self.weights.output_scale
            )
            result = unpack_bf16(self.device, actual)
            finite = bool(np.all(np.isfinite(result)))
            output_check = numerical_error(result[indices], expected)
            self.expected_output = expected
            self.expected_sum = expected_sum[indices]
        return {
            "passed": residual_check["passed"]
            and unchanged
            and finite
            and output_check["passed"]
            and all(item["passed"] for item in write_checks),
            "residual": residual_check,
            "output": output_check,
            "cache_writes": write_checks,
            "unchanged_cache": unchanged,
            "all_rows_finite": finite,
            "rows_checked": indices.tolist(),
            "output_tolerance_scope": "sampled output rows; all KV writes and guards",
        }
