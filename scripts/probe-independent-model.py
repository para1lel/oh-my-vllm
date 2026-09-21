#!/usr/bin/env python3
"""Short eager numerical smoke for the standalone model; not a throughput test.

This diagnostic does not start a service or replace the Rust scheduler. Run under
with-gpu.sh and with-env.sh using the target oh-my-vllm Python environment.
"""

import argparse
import importlib.util
import json
import sys

import torch
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import Batch, Qwen
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data0/shared/Qwen3.8-27B-FP8")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if importlib.util.find_spec("vllm") is not None:
        raise RuntimeError("standalone probe requires an environment without vLLM")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "请用中文简短介绍北京。"}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) + args.max_tokens > 784:
        raise ValueError("this short numerical probe uses one attention page")
    with torch.inference_mode():
        model = Qwen(args.model)
        caches = []
        for kind in model.kinds:
            if kind == "full_attention":
                cache = torch.zeros(
                    2, 2, 784, 4, 256, dtype=torch.bfloat16, device="cuda"
                )
            else:
                cache = (
                    torch.zeros(3, 10240, 3, dtype=torch.bfloat16, device="cuda"),
                    torch.zeros(3, 48, 128, 128, dtype=torch.float32, device="cuda"),
                )
            caches.append(cache)
        plan = PagedAttention()
        computed, source, generated = 0, 0, []
        tokens = ids
        for step in range(args.max_tokens):
            count = len(tokens)
            destination = 1 if source != 1 else 2
            starts = torch.tensor([0, count], dtype=torch.int32)
            plan.plan(
                starts,
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([computed + count], dtype=torch.int32),
                24,
                4,
                256,
            )
            writes = torch.full((count,), -1, dtype=torch.int32, device="cuda")
            writes[-1] = destination
            batch = Batch(
                positions=torch.arange(computed, computed + count, device="cuda"),
                starts=starts.cuda(),
                sequence_ids=torch.zeros(count, dtype=torch.int32, device="cuda"),
                state_reads=torch.tensor([source], dtype=torch.int32, device="cuda"),
                state_writes=writes,
                final_state_writes=torch.tensor(
                    [destination], dtype=torch.int64, device="cuda"
                ),
                fa_slots=torch.arange(
                    computed + 784, computed + count + 784, device="cuda"
                ),
                attention=plan,
                prefill_sequences=int(step == 0),
                prefill_tokens=count if step == 0 else 0,
            )
            hidden = model.forward(torch.tensor(tokens, device="cuda"), batch, caches)
            if not torch.isfinite(hidden).all():
                raise RuntimeError(f"nonfinite hidden state at step {step}")
            token = int(model.logits(hidden[-1:]).argmax(-1))
            generated.append(token)
            print(
                json.dumps(
                    {"step": step, "token": token, "text": tokenizer.decode(generated)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            computed += count
            tokens, source = [token], destination
            if token in {248044, 248046}:
                break
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)


if __name__ == "__main__":
    main()
