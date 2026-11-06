import argparse
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


REQUEST_SHAPES = [("ordinary", batch, 32768) for batch in (1, 2, 4)] + [
    ("mtp4", batch, length) for batch in (1, 2, 4) for length in (32768, 65536, 131072)
]


OUTPUT_LENGTHS = (64, 8192)


CONFIGURATIONS = [
    (*shape, output) for shape in REQUEST_SHAPES for output in OUTPUT_LENGTHS
]


def select_configurations(selected, outputs):
    if (
        not outputs
        or len(set(outputs)) != len(outputs)
        or any(value < 1 for value in outputs)
    ):
        raise ValueError("use distinct positive output token counts")
    selected = list(selected or [])
    if len(set(selected)) != len(selected):
        raise ValueError("use distinct case selections")
    configurations = [
        (*shape, output)
        for shape in REQUEST_SHAPES
        for output in outputs
        if not selected
        or ":".join(map(str, shape)) in selected
        or ":".join(map(str, (*shape, output))) in selected
    ]
    matched = {
        ":".join(map(str, case))
        for configuration in configurations
        for case in (configuration[:3], configuration)
    }
    if set(selected) - matched:
        raise ValueError("each case must match the selected output token counts")
    return configurations


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prompt-seed", default=uuid.uuid4().hex)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    parser.add_argument(
        "--case",
        choices=[
            ":".join(map(str, case)) for case in (*REQUEST_SHAPES, *CONFIGURATIONS)
        ],
        action="append",
    )
    parser.add_argument("--output-tokens", type=int, nargs="+", default=OUTPUT_LENGTHS)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int, default=128)
    arguments = parser.parse_args()
    return parser, arguments
