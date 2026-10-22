"""Compare captured draft sampling with sequential sampling on a CUDA device."""

import unittest
from contextlib import ExitStack

import numpy as np
from oh_my_vllm.operators.output.api import Settings
from oh_my_vllm.operators.output.plan import make_plan
from oh_my_vllm.operators.output.workspace import prepare
from oh_my_vllm.operators.runtime.device import Device


class DraftSampling(unittest.TestCase):
    def test_captured_history_probabilities_and_rng_match_sequential_sampling(self):
        device = Device()
        rows, vocabulary = 4, 248320
        parameters = np.asarray(
            [
                [1, 20, 0.95, 0, 0, 0.5, 1],
                [0, 3, 0.8, 0, 0, 1.25, 1],
                [0.7, 5, 0.6, 0.1, 0, -0.5, 1],
                [1.2, -1, 0.95, 0, 0, 2, 1],
            ],
            np.float32,
        )
        history = np.zeros((rows, vocabulary // 8), np.uint8)
        mask = np.zeros_like(history)
        candidates = (0, 1, 7, 8, 255, 256, vocabulary - 1)
        for token in candidates:
            mask[:, token // 8] |= 1 << (token % 8)
        gpu_history = device.upload(history)
        gpu_rng = device.empty((rows, 2), "uint64")
        settings = Settings(gpu_history, device.upload(mask), device.upload(parameters))
        stages = [prepare(make_plan(rows=rows), device, ()) for _ in range(4)]
        baseline = prepare(make_plan(rows=rows), device, ())
        captured = device.empty((4, rows), "int64")
        token_rows = [device.view(captured, (rows,), step * rows) for step in range(4)]
        logits = []
        for step, stage in enumerate(stages):
            values = np.full((rows, vocabulary), -20, np.float32)
            values[:, candidates] = np.asarray(candidates) % 5 + step * 0.25
            encoded = (values.view(np.uint32) >> 16).astype(np.uint16)
            logits.append(encoded)
            device.module.copy_bytes(stage.outputs.logits, device.upload(encoded))

        def run():
            for step, stage in enumerate(stages):
                stage.draw(settings, gpu_rng)
                stage.advance_presence(gpu_history)
                device.module.copy_bytes(token_rows[step], stage.outputs.tokens)

        try:
            edge_tokens = np.asarray([0, 7, 8, vocabulary - 1], np.int64)
            device.module.copy_bytes(
                stages[0].outputs.tokens, device.upload(edge_tokens)
            )
            stages[0].advance_presence(gpu_history)
            expected_history = history.copy()
            expected_history[0, 0] = 1
            expected_history[1, 0] = 128
            expected_history[2, 1] = 1
            expected_history[3, -1] = 128
            np.testing.assert_array_equal(
                device.download(gpu_history, np.uint8), expected_history
            )
            with ExitStack() as resources:
                for stage in stages:
                    resources.enter_context(stage.graph_lease())
                replay = resources.enter_context(device.graph(run))
                for iteration in range(3):
                    history.fill(0)
                    history[:, iteration // 8] |= 1 << iteration
                    offset = (1 << 64) - 3 if iteration == 2 else 17 + iteration * 4
                    random = np.asarray(
                        [[42 + row, offset] for row in range(rows)],
                        np.uint64,
                    )
                    device.module.copy_bytes(gpu_history, device.upload(history))
                    device.module.copy_bytes(gpu_rng, device.upload(random))
                    replay()
                    actual_tokens = device.download(captured, np.int64)
                    actual_rng = device.download(gpu_rng, np.uint64)
                    actual_history = device.download(gpu_history, np.uint8)
                    for step, stage in enumerate(stages):
                        device.module.copy_bytes(gpu_history, device.upload(history))
                        device.module.copy_bytes(gpu_rng, device.upload(random))
                        device.module.copy_bytes(
                            baseline.outputs.logits, device.upload(logits[step])
                        )
                        baseline.draw(settings, gpu_rng)
                        expected = device.download(baseline.outputs.tokens, np.int64)
                        np.testing.assert_array_equal(actual_tokens[step], expected)
                        np.testing.assert_array_equal(
                            device.download(stage.outputs.probabilities, np.float32),
                            device.download(baseline.outputs.probabilities, np.float32),
                        )
                        for row, token in enumerate(expected):
                            history[row, token // 8] |= 1 << (token % 8)
                        random[:, 1] += (parameters[:, 0] > 0).astype(np.uint64)
                    np.testing.assert_array_equal(actual_rng, random)
                    np.testing.assert_array_equal(actual_history, history)
            with self.assertRaises(RuntimeError):
                stages[0].advance_presence(device.empty((rows, 8), "uint8"))
        finally:
            device.synchronize()
            for stage in (*stages, baseline):
                stage.close()


if __name__ == "__main__":
    unittest.main()
