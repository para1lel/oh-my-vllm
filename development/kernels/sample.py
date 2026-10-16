import hashlib

import numpy as np
import tvm_ffi
from oh_my_vllm.kernels.sample import MTPAccept, Sample, Weights

from .build import build
from .reference import bf16, bf16_bits, numerical_error, residual_rms

FUNCTIONS = ("sample_workspace", "sample", "mtp_accept")
VOCABULARY = 248320


def philox_word(seed, offset):
    block, lane = divmod(int(offset), 4)
    values = [block & 0xFFFFFFFF, block >> 32, 0, 0]
    key = [int(seed) & 0xFFFFFFFF, int(seed) >> 32]
    for _ in range(10):
        a, b = 0xD2511F53 * values[0], 0xCD9E8D57 * values[2]
        values = [
            ((b >> 32) ^ values[1] ^ key[0]) & 0xFFFFFFFF,
            b & 0xFFFFFFFF,
            ((a >> 32) ^ values[3] ^ key[1]) & 0xFFFFFFFF,
            a & 0xFFFFFFFF,
        ]
        key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF, (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return values[lane]


def philox_uniform(seed, offset):
    word = philox_word(seed, offset)
    return float(np.float32(word) * np.float32(2.0**-32) + np.float32(2.0**-33))


def philox_uniform64(seed, offset):
    bits = philox_word(seed, offset) ^ (philox_word(seed, int(offset) + 1) << 21)
    return bits * 2.0**-53 + 2.0**-54


def probabilities(logits, history, mask, parameters):
    result = np.zeros_like(logits, dtype=np.float64)
    for row, p in enumerate(parameters):
        score = np.asarray(logits[row], dtype=np.float64).copy()
        prompt, generated = history[row]
        repeated = (prompt + generated) > 0
        score[repeated] = np.where(
            score[repeated] > 0, score[repeated] / p[6], score[repeated] * p[6]
        )
        score -= generated * p[4] + (generated > 0) * p[5]
        score[mask[row] == 0] = -np.inf
        if not np.isfinite(score).any():
            continue
        if p[0] == 0:
            result[row, np.argmax(score)] = 1
            continue
        score /= p[0]
        order = np.argsort(-score, kind="stable")
        values = np.exp(score[order] - np.max(score))
        topk = int(p[1]) if p[1] > 0 else score.size
        threshold = score[order[min(topk, score.size) - 1]]
        values[(score[order] < threshold) | (values < p[3])] = 0
        before = np.cumsum(values) - values
        values[before >= float(p[2]) * np.sum(values)] = 0
        result[row, order] = values / np.sum(values)
    return result


def selected_tokens(logits, history, mask, parameters, probability, rng):
    tokens = []
    for row, p in enumerate(parameters):
        score = np.asarray(logits[row], dtype=np.float64).copy()
        prompt, generated = history[row]
        repeated = (prompt + generated) > 0
        score[repeated] = np.where(
            score[repeated] > 0, score[repeated] / p[6], score[repeated] * p[6]
        )
        score -= generated * p[4] + (generated > 0) * p[5]
        score[mask[row] == 0] = -np.inf
        order = np.argsort(-score, kind="stable")
        if p[0] == 0:
            tokens.append(order[0])
            continue
        draw = philox_uniform(*rng[row])
        cumulative = np.cumsum(probability[row, order], dtype=np.float64)
        index = np.searchsorted(cumulative, draw * cumulative[-1], side="left")
        tokens.append(order[min(index, len(order) - 1)])
    return np.asarray(tokens, dtype=np.int64)


def acceptance_reference(target, proposal, drafts, rng, private_slots):
    batch = drafts.shape[0]
    target = target.reshape(batch, 5, -1)
    kept = np.full((batch, 5), -1, dtype=np.int64)
    counts, slots = np.empty(batch, np.int32), np.empty(batch, np.int32)
    counters = rng.copy()
    for request in range(batch):
        accepted, draws = 0, 0
        for index in range(4):
            token = drafts[request, index]
            if token == -1:
                if np.any(drafts[request, index:] != -1):
                    raise ValueError("inactive drafts must be a suffix")
                break
            if token < 0 or token >= target.shape[-1]:
                raise ValueError("invalid draft token")
            p = float(target[request, index, token])
            q = float(proposal[request, index, token])
            draw = philox_uniform64(rng[request, 0], int(rng[request, 1]) + 2 * draws)
            draws += 1
            if q == 0 or float(np.float32(p) / np.float32(q)) < draw:
                break
            kept[request, index] = token
            accepted += 1
        mass = target[request, accepted].astype(np.float32)
        if accepted < 4 and drafts[request, accepted] >= 0:
            mass = np.maximum(mass - proposal[request, accepted], 0)
        cumulative = np.cumsum(mass)
        draw = philox_uniform64(rng[request, 0], int(rng[request, 1]) + 2 * draws)
        kept[request, accepted] = min(
            np.searchsorted(cumulative, draw * float(cumulative[-1]), side="left"),
            len(mass) - 1,
        )
        counters[request, 1] += 2 * (draws + 1)
        counts[request] = accepted + 1
        slots[request] = private_slots[request, accepted]
    return kept, counts, slots, counters


def projection(value, weight):
    output = np.empty((value.shape[0], VOCABULARY), dtype=np.float64)
    for first in range(0, VOCABULARY, 2048):
        last = min(first + 2048, VOCABULARY)
        output[:, first:last] = bf16(
            np.asarray(value, dtype=np.float64)
            @ np.asarray(weight[first:last], dtype=np.float64).T
        )
    return output


def unpack(device, value):
    return (device.download(value, np.uint16).astype(np.uint32) << 16).view(np.float32)


class SampleWeights:
    def __init__(self, device, checkpoint, *, norm="model.language_model.norm.weight"):
        self.norm = checkpoint.tensor(norm)
        self.projection = checkpoint.tensor("lm_head.weight")
        self.device_weights = Weights(
            device.upload(self.norm),
            device.upload(bf16_bits(self.projection), "bfloat16"),
        )


class SampleCase:
    def __init__(self, device, weights, shape):
        build("sample", FUNCTIONS)
        self.device, self.weights, self.shape = device, weights, shape
        rng = np.random.default_rng(shape.rows + 101)
        self.x = bf16(rng.normal(0, 0.2, (shape.rows, 5120)))
        self.residual = bf16(rng.normal(0, 1, (shape.rows, 5120)))
        self.inputs = tuple(
            device.upload(bf16_bits(value), "bfloat16")
            for value in (self.x, self.residual)
        )
        self.history = np.zeros((shape.rows, 2, VOCABULARY), dtype=np.int32)
        self.mask = np.ones((shape.rows, VOCABULARY), dtype=np.uint8)
        for row in range(shape.rows):
            self.history[row, 0, rng.integers(0, VOCABULARY, 32)] = 2
            self.history[row, 1, rng.integers(0, VOCABULARY, 16)] = 1
            self.mask[row, rng.integers(0, VOCABULARY, 128)] = 0
        self.parameters = np.tile(
            np.array([0.7, 40, 0.95, 0, 0.2, 0.5, 1.1], dtype=np.float32),
            (shape.rows, 1),
        )
        self.rng_initial = np.column_stack(
            [
                np.arange(shape.rows, dtype=np.uint64) + 17,
                np.zeros(shape.rows, np.uint64),
            ]
        )
        self.rng = device.upload(self.rng_initial)
        self.sampling_arguments = (
            device.upload(self.history),
            device.upload(self.mask),
            device.upload(self.parameters),
            self.rng,
        )
        self.operation = Sample(weights.device_weights, shape.rows, device.empty)
        self.accept = None
        if shape.variant == "verify":
            self.accept = MTPAccept(shape.batch, device.empty)
            self.proposal = np.zeros((shape.batch, 4, VOCABULARY), dtype=np.float32)
            self.drafts = np.zeros((shape.batch, 4), dtype=np.int64)
            self.accept_rng_initial = np.column_stack(
                [
                    np.arange(shape.batch, dtype=np.uint64) + 119,
                    np.zeros(shape.batch, np.uint64),
                ]
            )
            self.accept_rng = device.upload(self.accept_rng_initial)
            self.private_slots = np.arange(shape.batch * 5, dtype=np.int32).reshape(
                shape.batch, 5
            )
            self.accept_arguments = (
                device.upload(self.proposal),
                device.upload(self.drafts),
                self.accept_rng,
                device.upload(self.private_slots),
                shape.batch * 5 + 1,
            )
        self.expected_identity = None

    def reset(self):
        self.device.module.copy_bytes(self.rng, tvm_ffi.from_dlpack(self.rng_initial))
        if self.accept is not None:
            self.device.module.copy_bytes(
                self.accept_rng, tvm_ffi.from_dlpack(self.accept_rng_initial)
            )

    def expected(self):
        identity = hashlib.sha256(
            self.x.tobytes()
            + self.residual.tobytes()
            + self.history.tobytes()
            + self.mask.tobytes()
            + self.parameters.tobytes()
        ).digest()
        if identity != self.expected_identity:
            normalized = residual_rms(self.x, self.residual, self.weights.norm)
            self.expected_logits = projection(normalized, self.weights.projection)
            self.expected_probability = probabilities(
                self.expected_logits, self.history, self.mask, self.parameters
            )
            self.expected_identity = identity
            if self.accept is not None:
                for request in range(self.shape.batch):
                    for index in range(4):
                        row = request * 5 + index
                        distribution = self.expected_probability[row]
                        token = (
                            int(np.argmax(distribution))
                            if index < 3
                            else int(np.flatnonzero(self.mask[row] == 0)[0])
                        )
                        self.drafts[request, index] = token
                        if index < 3:
                            self.proposal[request, index] = distribution
                        else:
                            self.proposal[request, index] = 0
                            self.proposal[request, index, token] = 1
                for array, tensor in zip(
                    (self.proposal, self.drafts), self.accept_arguments[:2], strict=True
                ):
                    self.device.module.copy_bytes(tensor, tvm_ffi.from_dlpack(array))
        return self.expected_probability

    def run(self):
        output = self.operation(*self.inputs, *self.sampling_arguments)
        if self.accept is not None:
            self.accept(output[2], *self.accept_arguments)
        return output

    def check(self, *, execute=True, logits_tolerance="bf16"):
        expected = self.expected()
        if execute:
            self.run()
        actual_logits = unpack(self.device, self.operation.buffers[1])
        actual = self.device.download(self.operation.buffers[6], np.float32)
        tokens = self.device.download(self.operation.buffers[7], np.int64)
        selected = self.device.download(self.operation.buffers[8], np.float32)
        logit_check = numerical_error(
            actual_logits, self.expected_logits, tolerance=logits_tolerance
        )
        probability_check = numerical_error(actual, expected)
        finite = bool(np.isfinite(actual).all() and np.all(actual >= 0))
        normalized = bool(np.allclose(actual.sum(-1), 1, atol=1e-5))
        valid_tokens = bool(np.all((tokens >= 0) & (tokens < VOCABULARY)))
        if valid_tokens:
            rows = np.arange(self.shape.rows)
            valid_tokens = bool(
                np.all(self.mask[rows, tokens]) and np.all(actual[rows, tokens] > 0)
            )
            valid_tokens &= bool(np.allclose(selected, actual[rows, tokens], atol=1e-6))
            valid_tokens &= bool(
                np.array_equal(
                    tokens,
                    selected_tokens(
                        actual_logits,
                        self.history,
                        self.mask,
                        self.parameters,
                        actual,
                        self.rng_initial,
                    ),
                )
            )
        counters = self.device.download(self.rng, np.uint64)
        expected_rng = self.rng_initial.copy()
        expected_rng[:, 1] += (self.parameters[:, 0] > 0).astype(np.uint64)
        counter_check = bool(np.array_equal(counters, expected_rng))
        report = {
            "passed": finite
            and normalized
            and valid_tokens
            and counter_check
            and logit_check["passed"]
            and probability_check["passed"],
            "logits": logit_check,
            "probabilities": probability_check,
            "all_rows_finite": finite,
            "normalized": normalized,
            "valid_selected_tokens": valid_tokens,
            "rng_counter": counter_check,
            "output_tolerance_scope": "all vocabulary columns and all rows",
        }
        if self.accept is not None:
            kept, counts, slots = (
                self.device.download(tensor, dtype)
                for tensor, dtype in zip(
                    self.accept.buffers, (np.int64, np.int32, np.int32), strict=True
                )
            )
            prefix = bool(np.all((counts >= 1) & (counts <= 5)))
            expected_kept, expected_counts, expected_slots, expected_rng = (
                acceptance_reference(
                    actual,
                    self.proposal,
                    self.drafts,
                    self.accept_rng_initial,
                    self.private_slots,
                )
            )
            prefix &= bool(
                np.array_equal(kept, expected_kept)
                and np.array_equal(counts, expected_counts)
                and np.array_equal(slots, expected_slots)
                and np.array_equal(
                    self.device.download(self.accept_rng, np.uint64), expected_rng
                )
            )
            for request, count in enumerate(counts):
                if count < 1 or count > 5:
                    continue
                prefix &= bool(
                    np.array_equal(
                        kept[request, : count - 1], self.drafts[request, : count - 1]
                    )
                )
                prefix &= bool(slots[request] == self.private_slots[request, count - 1])
                prefix &= bool(np.all(kept[request, count:] == -1))
            report["acceptance_state_selection"] = prefix
            report["passed"] &= prefix
        report["greedy"] = self.check_greedy(actual_logits)
        report["passed"] &= report["greedy"]["passed"]
        return report

    def greedy_result(self, expected, rng):
        probability = self.device.download(self.operation.buffers[6], np.float32)
        tokens = self.device.download(self.operation.buffers[7], np.int64)
        selected = self.device.download(self.operation.buffers[8], np.float32)
        counters = self.device.download(self.rng, np.uint64)
        valid = expected.any(axis=1)
        expected_tokens = np.where(valid, expected.argmax(axis=1), -1)
        checks = {
            "probabilities_exact": bool(np.array_equal(probability, expected)),
            "tokens_exact": bool(np.array_equal(tokens, expected_tokens)),
            "selected_probability_exact": bool(np.array_equal(selected, valid)),
            "rng_unchanged": bool(np.array_equal(counters, rng)),
        }
        return {"passed": all(checks.values()), **checks}

    def check_greedy(self, logits):
        parameters = self.parameters.copy()
        parameters[:, 0] = 0
        tied_parameters = parameters.copy()
        tied_parameters[:, 4:6] = 0
        tied_parameters[:, 6] = 1
        tied_mask = np.zeros_like(self.mask)
        penalty_history = np.zeros_like(self.history)
        for row, values in enumerate(logits):
            finite, counts = np.unique(
                values[np.isfinite(values) & (values != 0)], return_counts=True
            )
            duplicated = finite[counts > 1]
            if not len(duplicated):
                raise ValueError(
                    "greedy tie check needs duplicate finite nonzero BF16 logits"
                )
            tied_ids = np.flatnonzero(values == duplicated[0])[:2]
            tied_mask[row, tied_ids] = 1
            penalty_history[row, 1, tied_ids[0]] = 1
        penalty_parameters = tied_parameters.copy()
        penalty_parameters[:, 4:7] = (0.2, 0.5, 1.1)
        d = self.device
        saved_rng = d.download(self.rng, np.uint64)
        saved_invalid = self.operation.allow_invalid
        reports = {}

        def run():
            return self.operation(*self.inputs, *self.sampling_arguments, greedy=True)

        try:
            cases = (
                (
                    "penalties_and_mask",
                    self.history,
                    self.mask,
                    parameters,
                    saved_invalid,
                ),
                ("tied_logits", self.history, tied_mask, tied_parameters, False),
                (
                    "penalized_tie",
                    penalty_history,
                    tied_mask,
                    penalty_parameters,
                    False,
                ),
                (
                    "empty_mask",
                    self.history,
                    np.zeros_like(self.mask),
                    parameters,
                    True,
                ),
            )
            for name, history, mask, current_parameters, allow_invalid in cases:
                self.operation.allow_invalid = allow_invalid
                for destination, source in (
                    (self.sampling_arguments[0], history),
                    (self.sampling_arguments[1], mask),
                    (self.sampling_arguments[2], current_parameters),
                ):
                    d.module.copy_bytes(destination, tvm_ffi.from_dlpack(source))
                expected = probabilities(logits, history, mask, current_parameters)
                run()
                reports[name] = {"eager": self.greedy_result(expected, saved_rng)}
                with d.graph(run) as replay:
                    replay()
                    reports[name]["graph"] = self.greedy_result(expected, saved_rng)
        finally:
            self.operation.allow_invalid = saved_invalid
            for destination, source in (
                (self.sampling_arguments[0], self.history),
                (self.sampling_arguments[1], self.mask),
                (self.sampling_arguments[2], self.parameters),
                (self.rng, saved_rng),
            ):
                d.module.copy_bytes(destination, tvm_ffi.from_dlpack(source))
        return {
            "passed": all(
                result["passed"]
                for modes in reports.values()
                for result in modes.values()
            ),
            "cases": reports,
            "output_tolerance_scope": "all vocabulary columns and all rows",
        }
