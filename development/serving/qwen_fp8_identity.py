"""Bind archived Qwen calculations to the repository code that determines them."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_FILES = tuple(
    sorted(
        [
            f"development/serving/{name}.py"
            for name in (
                "accounting",
                "benchmark_case",
                "benchmark_options",
                "benchmark_trace",
                "qwen_fp8_acceptance",
                "qwen_fp8_evidence",
                "qwen_fp8_identity",
                "qwen_fp8_math",
                "qwen_fp8_roofline",
                "qwen_fp8_schedule",
                "qwen_fp8_trace",
                "workload",
            )
        ]
        + [
            "python/oh_my_vllm/models/qwen38/cache_budget.py",
            "python/oh_my_vllm/models/qwen38/tokenizer.py",
            "python/oh_my_vllm/operators/runtime/storage.py",
        ]
    )
)


def calculator_identity():
    files = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in CALCULATOR_FILES
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return dict(files=files, sha256=hashlib.sha256(encoded).hexdigest())
