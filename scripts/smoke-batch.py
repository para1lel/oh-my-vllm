"""Check request isolation with real Chinese text, arrivals and recompute preemption."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
)
parser.add_argument("--socket", required=True)
parser.add_argument("--scheduler-blocks", type=int, required=True)
parser.add_argument("--num-speculative-tokens", type=int, default=0)
parser.add_argument("--prefix-hit", action="store_true")
parser.add_argument("--max-tokens", type=int, default=1024)
args = parser.parse_args()
model = "/data0/shared/Qwen3.8-27B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model)
cities = [("北京", "Beijing"), ("东京", "Tokyo")]
prompts = []
questions = []
for city, english in cities:
    question = (
        f"\n请用中文写一篇至少四千字的文章,详细介绍{city}的历史、地理、交通、文化和景点。"
        f"每段都以“{city}:”开头,不要介绍另一个城市。请直接开始正文。"
    )
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": "__CONTENT_SLOT__"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    before, marker, after = template.partition("__CONTENT_SLOT__")
    assert marker
    head = tokenizer.encode(before, add_special_tokens=False)
    tail = tokenizer.encode(question + after, add_special_tokens=False)
    filler = tokenizer.encode(
        f"Information about {english}. ", add_special_tokens=False
    )
    count = 2048 - len(head) - len(tail)
    assert count > 0
    prompts.append(head + (filler * (count // len(filler) + 1))[:count] + tail)
    questions.append(question.strip())

with tempfile.TemporaryDirectory(prefix="oh-my-vllm-text-batch-") as directory:
    prompt_file = Path(directory) / "prompts.tokens"
    prompt_file.write_text("\n".join(" ".join(map(str, ids)) for ids in prompts) + "\n")
    command = [
        str(args.binary),
        "--socket",
        args.socket,
        "--model",
        model,
        "--max-model-len",
        "8192",
        "--num-gpu-blocks",
        "256",
        "--scheduler-blocks",
        str(args.scheduler_blocks),
        "--num-speculative-tokens",
        str(args.num_speculative_tokens),
        "run",
        "--prompt-file",
        str(prompt_file),
        "--arrival-interval",
        "3",
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.prefix_hit:
        command.append("--prefix-hit")
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)


def row(marker):
    return json.loads(
        next(
            line[len(marker) :]
            for line in result.stdout.splitlines()
            if line.startswith(marker)
        )
    )


outputs = row("output batches: ")
stats = row("batch stats: ")
assert len(outputs) == 2 and all(len(ids) == args.max_tokens for ids in outputs)
assert stats["preemptions"] > 0, stats
if args.prefix_hit:
    assert stats["initial_prefix_hit_tokens"] > 0, stats
if args.num_speculative_tokens:
    assert stats["accepted_draft_tokens"] > 0, stats
requests = []
for index, (ids, question) in enumerate(zip(outputs, questions, strict=True)):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    expected = cities[index][0]
    other = cities[1 - index][0]
    assert expected in text[:256] and other not in text[:256], text[:256]
    # Keep tail evidence: a correct first token cannot validate recompute recovery.
    assert expected in text[-512:] and other not in text[-512:], text[-512:]
    requests.append({"question": question, "output_tokens": len(ids), "text": text})
print(json.dumps({"stats": stats, "requests": requests}, ensure_ascii=False))
