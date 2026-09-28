"""Framework-owned sampling configuration and grammar-mask result."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    ignore_eos: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        # A temperature in (0, 1e-5) causes NaN in softmax; clamp to greedy.
        if 0 < self.temperature < 1e-5:
            object.__setattr__(self, "temperature", 0.0)
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        if type(self.top_k) is not int or self.top_k < -1:
            raise ValueError("top_k must be -1, 0 or a positive integer")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if any(
            not -2 <= p <= 2 for p in (self.frequency_penalty, self.presence_penalty)
        ):
            raise ValueError("frequency/presence penalties must be in [-2,2]")
        if (
            not math.isfinite(self.repetition_penalty)
            or not 1e-5 <= self.repetition_penalty <= 1e5
        ):
            raise ValueError("repetition_penalty must be in [1e-5,1e5]")


@dataclass
class GrammarOutput:
    structured_output_request_ids: list[str]
    grammar_bitmask: np.ndarray
