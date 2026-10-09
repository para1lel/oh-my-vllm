"""One allocator variable prevents legacy precedence and preserves emptiness."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "new,legacy,expected",
    [
        (None, None, "graph_capture_record_stream_reuse:True"),
        ("", None, ""),
        (None, "", ""),
        (None, "max_split_size_mb:64", "max_split_size_mb:64"),
        (
            "graph_capture_record_stream_reuse:True",
            "graph_capture_record_stream_reuse:False",
            "graph_capture_record_stream_reuse:True",
        ),
    ],
)
def test_explicit_allocator_config_precedence(new, legacy, expected):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    for name, value in (
        ("PYTORCH_ALLOC_CONF", new),
        ("PYTORCH_CUDA_ALLOC_CONF", legacy),
    ):
        environment.pop(name, None)
        if value is not None:
            environment[name] = value
    output = subprocess.check_output(
        [
            str(root / "scripts/with-env.sh"),
            "bash",
            "-c",
            'printf "%s|%s" "$PYTORCH_ALLOC_CONF" "${PYTORCH_CUDA_ALLOC_CONF+x}"',
        ],
        env=environment,
        text=True,
    )
    assert output == expected + "|"
