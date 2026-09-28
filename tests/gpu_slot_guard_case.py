"""Subprocess helper: an invalid CUDA slot can poison its device context."""

import sys

import torch


def main() -> None:
    entry = sys.argv[1]
    cache = torch.zeros(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([len(cache) * 784], device="cuda", dtype=torch.int64)
    try:
        if entry == "grouped_decode":
            from oh_my_vllm.kernels.decode_attention import decode

            query = torch.zeros(6, 24, 256, device="cuda", dtype=torch.bfloat16)
            tables = torch.zeros(6, 1, device="cuda", dtype=torch.int32)
            lengths = torch.arange(2, 8, device="cuda", dtype=torch.int32)
            starts = torch.tensor([0, 6], device="cuda", dtype=torch.int32)
            decode(
                query, cache, tables, lengths, first=1, max_tokens=784, starts=starts
            )
        elif entry == "append":
            from oh_my_vllm.kernels.attention import append

            k = torch.zeros(1, 4, 256, device="cuda", dtype=torch.bfloat16)
            append(cache, k, k, slots)
        elif entry == "prepare_attention":
            from oh_my_vllm.kernels.attention_prepare import prepare_attention

            packed = torch.zeros(1, 14336, device="cuda", dtype=torch.bfloat16)
            weight = torch.ones(256, device="cuda", dtype=torch.float32)
            position = torch.zeros(1, device="cuda", dtype=torch.int64)
            prepare_attention(packed, weight, weight, position, cache, slots)
        else:
            raise ValueError(entry)
        torch.cuda.synchronize()
    except (RuntimeError, torch.AcceleratorError) as error:
        message = str(error).lower()
        if any(
            word in message
            for word in (
                "assert",
                "trap",
                "illegal instruction",
                "cuda kv append launch failed",
                "cuda attention partial launch failed",
                "cuda attention merge launch failed",
                "unspecified launch failure",
                "grouped decode supports 1..5",
            )
        ):
            return
        raise
    raise AssertionError(f"{entry} silently accepted an out-of-range slot")


if __name__ == "__main__":
    main()
