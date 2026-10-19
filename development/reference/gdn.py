from dataclasses import replace
from itertools import pairwise

import numpy as np
import tvm_ffi
from oh_my_vllm.kernels.gdn import FUNCTIONS, GDN, Schedule, Weights, prepare_weights
from oh_my_vllm.kernels.specification import NUMERICAL_TOLERANCES

from development.native_build import build

from .ffn import reference_projection
from .reference import (
    bf16,
    bf16_bits,
    numerical_error,
    quantize,
    residual_rms,
    sigmoid,
    silu,
)


def decode_bf16(value):
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def summarize_rows(checks, rows):
    result = {
        "passed": all(check["passed"] for check in checks),
        "max_abs": max(check["max_abs"] for check in checks),
        "worst_chunk_nrmse": max(check["nrmse"] for check in checks),
        "reference_peak": max(check["reference_peak"] for check in checks),
        "max_tolerance_ratio": max(check["max_tolerance_ratio"] for check in checks),
        "tolerance_name": checks[0]["tolerance_name"],
        "required_atol": max(check["required_atol"] for check in checks),
        "rows_checked": rows,
    }
    violations = [item for check in checks for item in check.get("violations", ())]
    if violations:
        result["violations"] = violations
    return result


class GDNWeights:
    def __init__(self, device, checkpoint):
        build("gdn", FUNCTIONS)
        self.checkpoint = checkpoint
        prefix = "model.language_model.layers.0."
        attention = prefix + "linear_attn."
        self.norm = checkpoint.tensor(prefix + "input_layernorm.weight")
        self.qkvz = np.concatenate(
            [
                checkpoint.tensor(attention + f"in_proj_{name}.weight")
                for name in ("qkv", "z")
            ]
        )
        self.qkvz_scale = np.concatenate(
            [
                checkpoint.tensor(attention + f"in_proj_{name}.weight_scale_inv")
                for name in ("qkv", "z")
            ]
        )
        self.ba = np.concatenate(
            [
                checkpoint.tensor(attention + f"in_proj_{name}.weight")
                for name in ("b", "a")
            ]
        )
        self.convolution = checkpoint.tensor(attention + "conv1d.weight").reshape(
            10240, 4
        )
        self.a_log = checkpoint.tensor(attention + "A_log").astype(np.float32)
        self.dt_bias = checkpoint.tensor(attention + "dt_bias").astype(np.float32)
        self.head_norm = checkpoint.tensor(attention + "norm.weight").astype(np.float32)
        self.output = checkpoint.tensor(attention + "out_proj.weight")
        self.output_scale = checkpoint.tensor(attention + "out_proj.weight_scale_inv")
        self.device_weights = Weights(
            device.upload(self.norm),
            device.upload(self.qkvz, "float8_e4m3fn"),
            device.upload(self.qkvz_scale),
            device.upload(bf16_bits(self.ba), "bfloat16"),
            device.upload(bf16_bits(self.convolution), "bfloat16"),
            device.upload(self.a_log),
            device.upload(self.dt_bias),
            device.upload(self.head_norm),
            device.upload(self.output, "float8_e4m3fn"),
            device.upload(self.output_scale),
        )
        self.device_weights = prepare_weights(self.device_weights, device.empty)
        self.embedding = None
        self.device_embedding = None

    def with_embedding(self, device):
        if self.embedding is None:
            self.embedding = self.checkpoint.tensor(
                "model.language_model.embed_tokens.weight"
            )
            self.device_embedding = device.upload(bf16_bits(self.embedding), "bfloat16")
        return replace(self.device_weights, embedding=self.device_embedding)


def make_schedule(shape):
    if shape.rows != shape.batch * shape.queries:
        raise ValueError("GDN workload requires balanced sequence rows")
    starts = tuple(index * shape.queries for index in range(shape.batch + 1))
    verify = shape.queries <= 5
    reads = tuple(range(shape.batch))
    next_slot = shape.batch
    checkpoints, checkpoint_slots = [], []
    writes = [-1] * shape.rows
    for first, last in pairwise(starts):
        selected = (
            range(first, last)
            if verify
            else [
                row
                for row in range(first, last)
                if (shape.position + row - first + 1) % 784 == 0 or row == last - 1
            ]
        )
        for row in selected:
            if verify:
                writes[row] = next_slot
            else:
                checkpoints.append(row)
                checkpoint_slots.append(next_slot)
            next_slot += 1
    return Schedule(
        starts,
        reads,
        tuple(writes),
        tuple(checkpoints),
        tuple(checkpoint_slots),
        (shape.position,) * shape.batch,
        next_slot + 2,
        verify,
    )


class GDNCase:
    def __init__(self, device, weights, shape):
        build("gdn", FUNCTIONS)
        self.device, self.weights, self.shape = device, weights, shape
        self.schedule = make_schedule(shape)
        rng = np.random.default_rng(2200 + shape.rows)
        self.x = bf16(rng.normal(0, 0.2, (shape.rows, 5120))).astype(np.float32)
        self.residual = bf16(rng.normal(0, 1, (shape.rows, 5120))).astype(np.float32)
        self.tokens = np.arange(shape.rows, dtype=np.int64) % 248320
        self.media_rows = np.linspace(
            0, shape.rows - 1, shape.media_rows, dtype=np.int32
        )
        self.media = bf16(rng.normal(0, 0.2, (shape.media_rows, 5120))).astype(
            np.float32
        )
        device_weights = (
            weights.with_embedding(device)
            if shape.variant in {"embedding", "embedding_media"}
            else weights.device_weights
        )
        slots = self.schedule.slots
        self.initial_state = np.full(
            (slots, 48, 128, 128), bf16_bits(-0.125), dtype=np.uint16
        )
        self.initial_convolution = np.full(
            (slots, 10240, 3), bf16_bits(0.25), dtype=np.uint16
        )
        for slot in self.schedule.read_slots:
            self.initial_state[slot] = bf16_bits(rng.normal(0, 0.01, (48, 128, 128)))
            self.initial_convolution[slot] = bf16_bits(rng.normal(0, 0.1, (10240, 3)))
        self.state = device.upload(self.initial_state, "bfloat16")
        self.convolution_state = device.upload(self.initial_convolution, "bfloat16")
        self.inputs = tuple(
            device.upload(bf16_bits(value), "bfloat16")
            for value in (self.x, self.residual)
        )
        self.tokens_device = device.upload(self.tokens)
        self.media_device = device.upload(bf16_bits(self.media), "bfloat16")
        self.media_rows_device = device.upload(self.media_rows)
        self.operation = GDN(device_weights, self.schedule, device.empty, device.upload)

    def graph_lease(self):
        return self.operation.graph_lease()

    def close(self):
        self.operation.close()

    def reset(self):
        for output, initial in (
            (self.state, self.initial_state),
            (self.convolution_state, self.initial_convolution),
        ):
            self.device.module.copy_bytes(output, tvm_ffi.from_dlpack(initial))

    def run(self):
        return self.operation(
            *self.inputs,
            self.state,
            self.convolution_state,
            self.tokens_device,
            self.media_device,
            self.media_rows_device,
        )

    def download_bf16(self, value):
        return decode_bf16(self.device.download(value, np.uint16))

    def check(self, *, execute=True):
        if execute:
            self.reset()
            self.run()
        output, summed = (
            self.download_bf16(value) for value in self.operation.buffers[:2]
        )
        rows = self.shape.rows
        input_tolerance = "mxfp8_gdn_input" if rows > 128 else "bf16"
        output_tolerance = "mxfp8_gdn_output" if rows > 128 else "bf16"
        if not rows:
            unchanged = bool(
                np.array_equal(
                    self.device.download(self.state, np.uint16), self.initial_state
                )
                and np.array_equal(
                    self.device.download(self.convolution_state, np.uint16),
                    self.initial_convolution,
                )
            )
            return {
                "passed": unchanged,
                "empty_rows": True,
                "unchanged_slots": unchanged,
            }
        indices = np.unique([0, rows // 2, rows - 1])
        inputs = self.x
        if self.shape.variant in {"embedding", "embedding_media"}:
            inputs = self.weights.embedding[self.tokens].copy()
            inputs[self.media_rows] = self.media
        expected_sum = bf16(inputs + self.residual).astype(np.float32)
        residual_check = numerical_error(summed, expected_sum)
        residual_check["passed"] = bool(np.array_equal(summed, expected_sum))
        projection = self.download_bf16(self.operation.buffers[6])
        ba = self.download_bf16(self.operation.buffers[7])
        actual_normalized = self.download_bf16(self.operation.buffers[3])
        actual_data = self.device.download(self.operation.buffers[4], np.uint8)
        actual_scales = self.device.download(self.operation.buffers[5], np.float32)
        expected_projection = np.empty_like(projection)
        expected_ba = np.empty_like(ba)
        projection_checks, ba_checks, norm_checks, upstream_checks = [], [], [], []
        quantization = {"passed": True, "codes_exact": True, "scales_exact": True}
        recomputed_rows = 0
        for first in range(0, rows, 256):
            last = min(rows, first + 256)
            normalized = residual_rms(
                inputs[first:last], self.residual[first:last], self.weights.norm
            )
            data, scales = quantize(normalized)
            projected = reference_projection(
                data, scales, self.weights.qkvz, self.weights.qkvz_scale
            )
            projected_ba = bf16(
                normalized @ np.asarray(self.weights.ba, dtype=np.float64).T
            )
            expected_projection[first:last] = projected
            expected_ba[first:last] = projected_ba
            norm_checks.append(
                numerical_error(actual_normalized[first:last], normalized)
            )
            rounded_data, rounded_scales = quantize(actual_normalized[first:last])
            quantization["codes_exact"] &= bool(
                np.array_equal(rounded_data, actual_data[first:last])
            )
            quantization["scales_exact"] &= bool(
                np.array_equal(rounded_scales, actual_scales[first:last])
            )
            upstream = numerical_error(
                projection[first:last], projected, tolerance=input_tolerance
            )
            if not upstream["passed"]:
                errors = np.abs(projection[first:last] - projected)
                limits = NUMERICAL_TOLERANCES[input_tolerance]
                local_rows, columns = np.nonzero(
                    errors > limits["atol"] + limits["rtol"] * np.abs(projected)
                )
                upstream["violations"] = [
                    {
                        "row": first + int(row),
                        "column": int(column),
                        "actual": float(projection[first + row, column]),
                        "expected": float(projected[row, column]),
                    }
                    for row, column in zip(local_rows, columns, strict=True)
                ]
            upstream_checks.append(upstream)
            different = np.flatnonzero(
                np.any(data != actual_data[first:last], axis=1)
                | np.any(scales != actual_scales[first:last], axis=1)
            )
            same_input_projection = projected
            if len(different):
                same_input_projection = projected.copy()
                same_input_projection[different] = reference_projection(
                    actual_data[first:last][different],
                    actual_scales[first:last][different],
                    self.weights.qkvz,
                    self.weights.qkvz_scale,
                )
                recomputed_rows += len(different)
            projection_checks.append(
                numerical_error(
                    projection[first:last],
                    same_input_projection,
                    tolerance=input_tolerance,
                )
            )
            ba_checks.append(numerical_error(ba[first:last], projected_ba))
        projection_check = summarize_rows(projection_checks, rows)
        upstream_check = summarize_rows(upstream_checks, rows)
        norm_check = summarize_rows(norm_checks, rows)
        ba_check = summarize_rows(ba_checks, rows)
        quantization["passed"] = (
            quantization["codes_exact"] and quantization["scales_exact"]
        )
        actual_states = self.device.download(self.state, np.uint16)
        actual_conv = self.device.download(self.convolution_state, np.uint16)
        writes = self.schedule.validate()
        untouched = sorted(
            set(range(self.schedule.slots)) - {slot for slot in writes if slot >= 0}
        )
        guards = bool(
            np.array_equal(actual_states[untouched], self.initial_state[untouched])
            and np.array_equal(
                actual_conv[untouched], self.initial_convolution[untouched]
            )
        )
        reference_output, state_checks, convolution_checks = self.reference_scan(
            expected_projection,
            expected_ba,
            actual_states,
            actual_conv,
            indices,
            writes,
        )
        quantized, output_scales = quantize(reference_output)
        expected_output = reference_projection(
            quantized, output_scales, self.weights.output, self.weights.output_scale
        )
        output_check = numerical_error(
            output[indices], expected_output, tolerance=output_tolerance
        )
        finite = bool(np.all(np.isfinite(output)))
        checks = [
            residual_check,
            norm_check,
            quantization,
            projection_check,
            ba_check,
            output_check,
            *state_checks,
            *convolution_checks,
        ]
        return {
            "passed": finite and guards and all(check["passed"] for check in checks),
            "all_rows_finite": finite,
            "unchanged_slots": guards,
            "projection_tolerance_scope": (
                "all rows and columns, original checkpoint and legacy FP8 inputs; "
                "CPU FP64 GEMM"
                if rows > 128
                else "all rows and columns, same rounded FP8 inputs, CPU FP64 GEMM"
            ),
            "upstream_projection_diagnostic_scope": (
                "CPU-normalized raw temporary drift; not whole-input equality"
            ),
            "state_tolerance_scope": "all private writes, independent CPU FP64",
            "output_tolerance_scope": (
                "sampled rows, all columns; CPU FP64 recurrence and output projection"
            ),
            "rows_checked": indices.tolist(),
            "residual": residual_check,
            "input_projection": projection_check,
            "input_projection_upstream_drift": upstream_check,
            "projection_rows_recomputed": recomputed_rows,
            "normalization": norm_check,
            "activation_quantization": quantization,
            "ba_projection": ba_check,
            "output": output_check,
            "state_writes": state_checks,
            "convolution_writes": convolution_checks,
        }

    def reference_scan(
        self, projection, ba, actual_states, actual_conv, indices, writes
    ):
        weight = self.weights
        outputs = np.empty((len(indices), 48, 128), dtype=np.float64)
        selected = {int(row): index for index, row in enumerate(indices)}
        state_checks, convolution_checks = [], []
        state_tolerance = "mxfp8_gdn_state" if self.shape.rows > 128 else "state"
        mapping = np.arange(48) // 3
        for sequence, (first, last) in enumerate(pairwise(self.schedule.sequence_rows)):
            read = self.schedule.read_slots[sequence]
            state = decode_bf16(self.initial_state[read]).astype(np.float64)
            history = decode_bf16(self.initial_convolution[read]).astype(np.float64)
            for row in range(first, last):
                current = projection[row, :10240].astype(np.float64)
                window = np.concatenate((history, current[:, None]), axis=-1)
                convolution = bf16(silu(np.sum(window * weight.convolution, axis=-1)))
                history = window[:, 1:]
                q, k, value = np.split(convolution, (2048, 4096))
                q, k = (item.reshape(16, 128) for item in (q, k))
                q, k = (
                    item / np.sqrt(np.sum(item * item, axis=-1, keepdims=True) + 1e-6)
                    for item in (q, k)
                )
                if self.schedule.verify:
                    q, k = (
                        item.astype(np.float32).astype(np.float64) for item in (q, k)
                    )
                else:
                    q, k = (bf16(item) for item in (q, k))
                q, k = q[mapping], k[mapping]
                a = ba[row, 48:].astype(np.float64) + weight.dt_bias
                decay = -np.exp(weight.a_log.astype(np.float64)) * np.logaddexp(0, a)
                beta = sigmoid(ba[row, :48].astype(np.float64))
                state *= np.exp(decay)[:, None, None]
                recalled = np.einsum("hvk,hk->hv", state, k)
                delta = beta[:, None] * (value.reshape(48, 128) - recalled)
                state += delta[:, :, None] * k[:, None, :]
                if writes[row] >= 0:
                    slot = writes[row]
                    check = numerical_error(
                        decode_bf16(actual_states[slot]),
                        bf16(state),
                        state=True,
                        state_tolerance=state_tolerance,
                    )
                    check.update(row=row, slot=slot)
                    state_checks.append(check)
                    check = numerical_error(
                        decode_bf16(actual_conv[slot]),
                        history,
                        state=True,
                        state_tolerance=state_tolerance,
                    )
                    check.update(row=row, slot=slot)
                    convolution_checks.append(check)
                if row in selected:
                    output = bf16(np.einsum("hvk,hk->hv", state, q) / np.sqrt(128))
                    output = output / np.sqrt(
                        np.mean(output * output, axis=-1, keepdims=True) + 1e-6
                    )
                    output = output * weight.head_norm
                    gate = projection[row, 10240:].reshape(48, 128)
                    outputs[selected[row]] = bf16(
                        output * silu(gate.astype(np.float64))
                    )
        return outputs.reshape(len(indices), 6144), state_checks, convolution_checks
