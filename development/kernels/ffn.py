import numpy as np
from oh_my_vllm.kernels.ffn import FFN, Weights

from .build import build
from .reference import (
    bf16,
    bf16_bits,
    fp8_decode,
    numerical_error,
    quantize,
    residual_rms,
    silu,
)


def reference_projection(data, scales, weight, weight_scales):
    activation = fp8_decode(data) * np.repeat(scales, 128, axis=-1)
    output = np.empty((data.shape[0], weight.shape[0]), dtype=np.float64)
    for first in range(0, weight.shape[0], 1024):
        last = min(first + 1024, weight.shape[0])
        scale = np.repeat(
            np.repeat(weight_scales[first // 128 : last // 128], 128, axis=0),
            128,
            axis=1,
        )
        decoded = fp8_decode(weight[first:last]) * scale
        output[:, first:last] = bf16(activation @ decoded.T)
    return output


class FFNWeights:
    def __init__(self, device, checkpoint):
        prefix = "model.language_model.layers.0."
        self.norm = checkpoint.tensor(prefix + "post_attention_layernorm.weight")
        self.gate_up = np.concatenate(
            [
                checkpoint.tensor(prefix + f"mlp.{name}_proj.weight")
                for name in ("gate", "up")
            ]
        )
        self.gate_up_scale = np.concatenate(
            [
                checkpoint.tensor(prefix + f"mlp.{name}_proj.weight_scale_inv")
                for name in ("gate", "up")
            ]
        )
        self.down = checkpoint.tensor(prefix + "mlp.down_proj.weight")
        self.down_scale = checkpoint.tensor(prefix + "mlp.down_proj.weight_scale_inv")
        self.device_weights = Weights(
            device.upload(self.norm),
            device.upload(self.gate_up, "float8_e4m3fn"),
            device.upload(self.gate_up_scale),
            device.upload(self.down, "float8_e4m3fn"),
            device.upload(self.down_scale),
        )


class FFNCase:
    def __init__(self, device, weights, shape):
        build("ffn", ("ffn",))
        self.device, self.weights, self.shape = device, weights, shape
        rng = np.random.default_rng(shape.rows)
        self.x = bf16(rng.normal(0, 0.2, (shape.rows, 5120)))
        self.residual = bf16(rng.normal(0, 1, (shape.rows, 5120)))
        self.inputs = tuple(
            device.upload(bf16_bits(value), "bfloat16")
            for value in (self.x, self.residual)
        )
        self.operation = FFN(weights.device_weights, shape.rows, device.empty)

    def reset(self):
        pass

    def run(self):
        return self.operation(*self.inputs)

    def check(self, *, execute=True):
        actual, summed = self.run() if execute else self.operation.buffers[:2]
        actual = (self.device.download(actual, np.uint16).astype(np.uint32) << 16).view(
            np.float32
        )
        summed = (self.device.download(summed, np.uint16).astype(np.uint32) << 16).view(
            np.float32
        )
        expected_sum = bf16(self.x + self.residual)
        residual_check = numerical_error(summed, expected_sum)
        residual_check["passed"] = bool(np.array_equal(summed, expected_sum))
        finite = bool(np.all(np.isfinite(actual)))
        indices = np.unique([0, self.shape.rows // 2, self.shape.rows - 1])
        normalized = residual_rms(
            self.x[indices], self.residual[indices], self.weights.norm
        )
        data, scales = quantize(normalized)
        gate_up = reference_projection(
            data, scales, self.weights.gate_up, self.weights.gate_up_scale
        )
        gate, up = np.split(gate_up, 2, axis=-1)
        product = bf16(silu(gate) * up)
        data, scales = quantize(product)
        expected = reference_projection(
            data, scales, self.weights.down, self.weights.down_scale
        )
        output_check = numerical_error(actual[indices], expected)
        return {
            "passed": finite and residual_check["passed"] and output_check["passed"],
            "all_rows_finite": finite,
            "output_tolerance_scope": "sampled rows, all columns",
            "residual": residual_check,
            "output": output_check,
            "rows_checked": indices.tolist(),
        }
