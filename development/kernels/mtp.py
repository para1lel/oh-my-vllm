import numpy as np
from oh_my_vllm.kernels.fa import FUNCTIONS as FA_FUNCTIONS
from oh_my_vllm.kernels.ffn import FFN
from oh_my_vllm.kernels.ffn import FUNCTIONS as FFN_FUNCTIONS
from oh_my_vllm.kernels.ffn import Weights as FFNDeviceWeights
from oh_my_vllm.kernels.mtp import (
    FUNCTIONS,
    InputWeights,
    MTPContext,
    MTPDraft,
    MTPInput,
)

from .build import build
from .fa import FACase, FAWeights, rotary, unpack_bf16
from .ffn import reference_projection
from .reference import (
    bf16,
    bf16_bits,
    linear,
    numerical_error,
    quantize,
    residual_rms,
    rms,
    silu,
)


class MTPWeights:
    def __init__(self, device, checkpoint):
        self.device, self.checkpoint = device, checkpoint
        self.embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.hidden_norm = checkpoint.tensor("mtp.pre_fc_norm_hidden.weight")
        self.embedding_norm = checkpoint.tensor("mtp.pre_fc_norm_embedding.weight")
        self.projection = checkpoint.tensor("mtp.fc.weight")
        self.device_weights = InputWeights(
            device.upload(bf16_bits(self.embedding), "bfloat16"),
            device.upload(self.hidden_norm),
            device.upload(self.embedding_norm),
            device.upload(bf16_bits(self.projection), "bfloat16"),
        )
        self.attention_weights = {}
        self.ffn = None

    def attention(self, variant):
        if variant not in self.attention_weights:
            self.attention_weights[variant] = FAWeights(
                self.device, self.checkpoint, prefix="mtp.layers.0.", variant=variant
            )
        return self.attention_weights[variant]

    def feed_forward(self):
        if self.ffn is None:
            from types import SimpleNamespace

            prefix = "mtp.layers.0."
            weights = SimpleNamespace(
                norm=self.checkpoint.tensor(prefix + "post_attention_layernorm.weight"),
                gate_up=np.concatenate(
                    [
                        self.checkpoint.tensor(prefix + f"mlp.{name}_proj.weight")
                        for name in ("gate", "up")
                    ]
                ),
                gate_up_scale=np.concatenate(
                    [
                        self.checkpoint.tensor(
                            prefix + f"mlp.{name}_proj.weight_scale_inv"
                        )
                        for name in ("gate", "up")
                    ]
                ),
                down=self.checkpoint.tensor(prefix + "mlp.down_proj.weight"),
                down_scale=self.checkpoint.tensor(
                    prefix + "mlp.down_proj.weight_scale_inv"
                ),
            )
            weights.device_weights = FFNDeviceWeights(
                self.device.upload(weights.norm),
                self.device.upload(weights.gate_up, "float8_e4m3fn"),
                self.device.upload(weights.gate_up_scale),
                self.device.upload(weights.down, "float8_e4m3fn"),
                self.device.upload(weights.down_scale),
            )
            self.ffn = weights
        return self.ffn


class MTPContextCase:
    def __init__(self, device, weights, shape):
        build("mtp", FUNCTIONS)
        build("fa", FA_FUNCTIONS)
        self.device, self.weights, self.shape = device, weights, shape
        rng = np.random.default_rng(shape.rows)
        self.hidden = bf16(rng.normal(0, 0.2, (shape.rows, 5120)))
        self.boundary = bf16(rng.normal(0, 0.2, (shape.batch, 5120)))
        self.tokens = rng.integers(0, 248320, shape.rows, dtype=np.int64)
        queries = shape.rows // shape.batch
        self.hidden_rows = np.arange(shape.rows, dtype=np.int32) - 1
        self.hidden_rows[::queries] = -1
        self.boundary_rows = np.repeat(np.arange(shape.batch), queries).astype(np.int32)
        self.input_arguments = (
            device.upload(bf16_bits(self.hidden), "bfloat16"),
            device.upload(self.tokens),
            device.upload(self.hidden_rows),
            device.upload(bf16_bits(self.boundary), "bfloat16"),
            device.upload(self.boundary_rows),
        )
        zeros = np.zeros_like(self.hidden)
        self.fa_case = FACase(
            device,
            weights.attention("kv_only"),
            shape,
            value=zeros,
            residual=zeros,
            first_token=1,
        )
        positions, slots, tables, sequences, lengths = self.fa_case.metadata
        self.attention_arguments = (
            positions,
            self.fa_case.cache,
            slots,
            tables,
            sequences,
            lengths,
        )
        self.operation = MTPContext(
            weights.device_weights,
            self.fa_case.weights.device_weights,
            shape.rows,
            device.empty,
            max_queries=queries,
            batch=shape.batch,
        )
        self.fa_case.operation = self.operation.attention
        self.fa_case.exact_residual = False

    def reset(self):
        self.fa_case.reset()

    def run(self):
        return self.operation(self.input_arguments, self.attention_arguments)

    def input_reference(self):
        expected = np.empty_like(self.hidden)
        for first in range(0, self.shape.rows, 64):
            end = min(first + 64, self.shape.rows)
            indices = self.hidden_rows[first:end]
            hidden = self.hidden[np.maximum(indices, 0)].copy()
            boundary = indices < 0
            hidden[boundary] = self.boundary[self.boundary_rows[first:end][boundary]]
            normalized_embedding = rms(
                self.weights.embedding[self.tokens[first:end]],
                self.weights.embedding_norm,
            )
            normalized_hidden = rms(hidden, self.weights.hidden_norm)
            expected[first:end] = linear(
                np.concatenate((normalized_embedding, normalized_hidden), axis=1),
                self.weights.projection,
            )
        return expected

    def check(self, *, execute=True):
        if execute:
            self.run()
        value, zero = self.operation.input.buffers[1:3]
        expected = self.input_reference()
        full_values = unpack_bf16(self.device, value)
        input_check = numerical_error(full_values, expected)
        indices = self.fa_case.sequence_rows[1:] - 1
        endpoint_values, endpoint_zero = (
            unpack_bf16(self.device, tensor) for tensor in self.operation.endpoints
        )
        endpoint_reference = numerical_error(endpoint_values, expected[indices])
        endpoint_check = bool(
            np.array_equal(endpoint_values, full_values[indices])
            and np.all(endpoint_zero == 0)
            and endpoint_reference["passed"]
        )
        zero_check = bool(np.all(unpack_bf16(self.device, zero) == 0))
        self.fa_case.x = expected
        attention_check = self.fa_case.check(execute=False)
        return {
            "passed": input_check["passed"]
            and zero_check
            and endpoint_check
            and attention_check["passed"],
            "input": input_check,
            "zero_residual": zero_check,
            "endpoints": {"passed": endpoint_check, "reference": endpoint_reference},
            "attention": attention_check,
            "context_pairing": "h[p-1] and embedding[p]; no position zero",
        }


class MTPDraftCase:
    def __init__(self, device, weights, shape, sample_case):
        build("mtp", FUNCTIONS)
        build("fa", FA_FUNCTIONS)
        build("ffn", FFN_FUNCTIONS)
        self.device, self.weights, self.shape = device, weights, shape
        self.sample_case = sample_case
        self.fa_case = FACase(
            device, weights.attention(shape.variant), shape, first_token=1
        )
        self.fa_case.check_indices = np.arange(shape.rows)
        self.input = MTPInput(weights.device_weights, shape.rows, device.empty)
        rng = np.random.default_rng(shape.rows)
        hidden = bf16(rng.normal(0, 0.2, (shape.rows, 5120)))
        tokens = rng.integers(0, 248320, shape.rows, dtype=np.int64)
        self.hidden, self.tokens = hidden, tokens
        self.input_arguments = (
            device.upload(bf16_bits(hidden), "bfloat16"),
            device.upload(tokens),
            device.upload(np.arange(shape.rows, dtype=np.int32)),
            device.upload(bf16_bits(hidden), "bfloat16"),
            device.upload(np.zeros(shape.rows, dtype=np.int32)),
        )
        if shape.variant == "query_only":
            endpoint = linear(
                np.concatenate(
                    (
                        rms(weights.embedding[tokens], weights.embedding_norm),
                        rms(hidden, weights.hidden_norm),
                    ),
                    axis=1,
                ),
                weights.projection,
            )
            self.fa_case.x, self.fa_case.residual = endpoint, np.zeros_like(endpoint)
            self.input_arguments = tuple(
                device.upload(bf16_bits(v), "bfloat16")
                for v in (endpoint, np.zeros_like(endpoint))
            )
            kv = weights.attention("kv_only")
            data, scales = quantize(rms(endpoint, kv.norm))
            packed = reference_projection(
                data, scales, kv.projection, kv.projection_scale
            )
            packed = packed.reshape(shape.rows, 2, 4, 256)
            keys = rotary(rms(packed[:, 0], kv.key_norm), self.fa_case.positions)
            blocks, offsets = self.fa_case.slots // 784, self.fa_case.slots % 784
            self.fa_case.cache_initial[blocks, 0, offsets] = bf16_bits(keys)
            self.fa_case.cache_initial[blocks, 1, offsets] = bf16_bits(packed[:, 1])
            self.fa_case.reset()
        positions, slots, tables, sequences, lengths = self.fa_case.metadata
        self.attention_arguments = (
            positions,
            self.fa_case.cache,
            slots,
            tables,
            sequences,
            lengths,
        )
        self.ffn = FFN(weights.feed_forward().device_weights, shape.rows, device.empty)
        self.operation = MTPDraft(
            self.input,
            self.fa_case.operation,
            self.ffn,
            sample_case.operation,
            variant=shape.variant,
        )

    def reset(self):
        self.fa_case.reset()
        self.sample_case.reset()

    def run(self):
        return self.operation(
            self.input_arguments,
            self.attention_arguments,
            self.sample_case.sampling_arguments,
        )

    def check(self, *, execute=True):
        if execute:
            self.run()
        if self.shape.variant == "full":
            expected = linear(
                np.concatenate(
                    (
                        rms(
                            self.weights.embedding[self.tokens],
                            self.weights.embedding_norm,
                        ),
                        rms(self.hidden, self.weights.hidden_norm),
                    ),
                    axis=1,
                ),
                self.weights.projection,
            )
            self.fa_case.x, self.fa_case.residual = expected, np.zeros_like(expected)
            self.fa_case.exact_residual = False
            input_check = numerical_error(
                unpack_bf16(self.device, self.input.buffers[1]), expected
            )
        else:
            input_check = {
                "passed": True,
                "scope": "context endpoint; FC is outside this boundary",
            }
        attention_check = self.fa_case.check(
            execute=False, output_tolerance="mtp_attention"
        )
        value = self.fa_case.expected_output
        residual = self.fa_case.expected_sum
        w = self.weights.feed_forward()
        summed = bf16(value + residual)
        data, scales = quantize(residual_rms(value, residual, w.norm))
        packed = reference_projection(data, scales, w.gate_up, w.gate_up_scale)
        gate, up = np.split(packed, 2, axis=1)
        data, scales = quantize(bf16(silu(gate) * up))
        expected = reference_projection(data, scales, w.down, w.down_scale)
        ffn_check = numerical_error(
            unpack_bf16(self.device, self.ffn.buffers[0]),
            expected,
            tolerance="mtp_hidden",
        )
        residual_check = numerical_error(
            unpack_bf16(self.device, self.ffn.buffers[1]),
            summed,
            tolerance="mtp_residual",
        )
        residual_store_exact = bool(
            np.array_equal(
                self.device.download(self.ffn.buffers[1], np.uint16),
                bf16_bits(
                    unpack_bf16(self.device, self.fa_case.operation.buffers[0])
                    + unpack_bf16(self.device, self.fa_case.operation.buffers[1])
                ),
            )
        )
        self.sample_case.x, self.sample_case.residual = expected, summed
        self.sample_case.inputs = self.ffn.buffers[:2]
        sample_check = self.sample_case.check(
            execute=False, logits_tolerance="mtp_logits"
        )
        return {
            "passed": input_check["passed"]
            and attention_check["passed"]
            and ffn_check["passed"]
            and residual_check["passed"]
            and residual_store_exact
            and sample_check["passed"],
            "input": input_check,
            "attention": attention_check,
            "ffn": ffn_check,
            "ffn_residual": residual_check,
            "ffn_residual_store_exact": residual_store_exact,
            "sample": sample_check,
        }
