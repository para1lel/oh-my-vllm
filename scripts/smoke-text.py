"""Run actual tokenizer -> Rust scheduler -> GPUWorker -> decoded text smoke."""

import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--max-tokens", type=int, default=64)
parser.add_argument("--prompt", default="请用中文简短介绍北京。")
parser.add_argument("--socket", required=True)
parser.add_argument("--context-repeats", type=int, default=0)
parser.add_argument("--num-speculative-tokens", type=int, default=0)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = "/data0/shared/Qwen3.8-27B-FP8"
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
        str(root / "target/release/oh-my-vllm-zmq-worker"),
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
        "run",
        "--tokens",
        *map(str, tokens),
        "--max-tokens",
        str(args.max_tokens),
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
