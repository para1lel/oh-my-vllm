"""Run actual tokenizer -> Rust scheduler -> GPUWorker -> decoded text smoke."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default=os.environ.get("OH_MY_VLLM_MODEL"),
    required=not bool(os.environ.get("OH_MY_VLLM_MODEL")),
    help="checkpoint path; defaults to OH_MY_VLLM_MODEL",
)
parser.add_argument("--max-tokens", type=int, default=64)
parser.add_argument("--prompt", default="请用中文简短介绍北京。")
parser.add_argument("--socket", required=True)
parser.add_argument("--context-repeats", type=int, default=0)
parser.add_argument("--num-speculative-tokens", type=int, default=0)
parser.add_argument("--speculative-mode", choices=("none", "mtp", "dspark"))
parser.add_argument("--draft-model", default=os.environ.get("OH_MY_VLLM_DRAFT_MODEL"))
parser.add_argument("--dspark-confidence-threshold", type=float, default=0.2)
parser.add_argument("--prefix-hit", action="store_true")
parser.add_argument("--binary", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = args.model
tokenizer = AutoTokenizer.from_pretrained(model)
prompt = (
    "背景材料:北京拥有悠久的历史和丰富的文化。" * args.context_repeats + args.prompt
)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=False,
)
result = subprocess.run(
    [
        str(args.binary or root / "target/release/oh-my-vllm-zmq-worker"),
        "--model",
        model,
        "--socket",
        args.socket,
        "--max-model-len",
        "8192",
        "--num-gpu-blocks",
        "128",
        "--num-speculative-tokens",
        str(args.num_speculative_tokens),
        *(
            ["--speculative-mode", args.speculative_mode]
            if args.speculative_mode
            else []
        ),
        *(["--draft-model", args.draft_model] if args.draft_model else []),
        "--dspark-confidence-threshold",
        str(args.dspark_confidence_threshold),
        "run",
        "--tokens",
        *map(str, tokens),
        "--max-tokens",
        str(args.max_tokens),
        *(["--prefix-hit"] if args.prefix_hit else []),
    ],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
)
line = next(
    line for line in result.stdout.splitlines() if line.startswith("output token ids:")
)
output = json.loads(line.split(":", 1)[1])
assert len(output) == args.max_tokens, (len(output), args.max_tokens)
print(
    json.dumps(
        {
            "prompt": args.prompt,
            "output_tokens": len(output),
            "text": tokenizer.decode(output, skip_special_tokens=True),
        },
        ensure_ascii=False,
    )
)
