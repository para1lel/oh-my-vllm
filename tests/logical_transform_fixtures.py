import hashlib
import os
from pathlib import Path

import torch
from oh_my_vllm.operators.runtime.native import load
from oh_my_vllm.operators.transform.api import FUNCTIONS


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


def engram_inputs(rows):
    return [
        torch.zeros((rows, 4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.zeros((rows, 25600), dtype=torch.bfloat16, device="cuda"),
        torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.empty(0, dtype=torch.uint8, device="cuda"),
        torch.full((rows, 4, 5120), -7, dtype=torch.bfloat16, device="cuda"),
    ]


class LogicalTransformContractsFixture:
    @classmethod
    def setUpClass(cls):
        cls.module = load("transform", FUNCTIONS)[0]


class LogicalTransformSourceFixturesFixture:
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import load_file

        root = Path(os.environ["OH_MY_VLLM_TRANSFORM_REFERENCE"])
        cls.module = load("transform", FUNCTIONS)[0]
        cls.fixtures = {}
        paths = (
            (
                "engram",
                "target-engram/arithmetic-reference.safetensors",
                "efd4f32c39e44460a5675828bdbdd3b85b1d383288afa64f6b820071a60f5679",
            ),
            (
                "rms",
                "target-verification/rms-reference.safetensors",
                "f57626ec73d2f025da82200ee5dbabbec0fafa1a3ba9146bdc933af3f0e212f9",
            ),
            (
                "synthetic",
                "target-native-transform/synthetic-reference.safetensors",
                "435c6f78fbf0c223df7a5ddbb427ce88e79f4410d948ead17a5aba8aa0d60802",
            ),
        )
        for label, relative, expected in paths:
            path = root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("independent source fixture identity differs")
            cls.fixtures[label] = load_file(str(path), device="cuda")
