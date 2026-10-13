"""Subprocess helper: an invalid CUDA slot can poison its device context."""

import sys

import torch


def main() -> None:
    entry = sys.argv[1]
    cache = torch.zeros(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([len(cache) * 784], device="cuda", dtype=torch.int64)
    try:
        if entry == "gdn_prefill":
            from oh_my_vllm.kernels.gdn import prefill

            length, invalid = int(sys.argv[2]), sys.argv[3]
            offsets = {
                "origin": [1, 1, length],
                "terminal": [0, 0, length + 1],
                "decreasing": [0, length, 0, length],
            }[invalid]
            q = torch.zeros(length, 1, 128, device="cuda", dtype=torch.bfloat16)
            gate = torch.zeros(length, 1, device="cuda")
            state = torch.zeros(len(offsets) - 1, 1, 128, 128, device="cuda")
            starts = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            prefill(q, q, q, gate, gate, state, starts)
        elif entry == "grouped_offsets":
            from oh_my_vllm.kernels.decode_attention import decode

            offsets = {
                "negative": [-1, 1],
                "terminal": [0, 2],
                "origin": [1, 1],
            }[sys.argv[2]]
            query = torch.zeros(1, 24, 256, device="cuda", dtype=torch.bfloat16)
            tables = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
            lengths = torch.ones(1, device="cuda", dtype=torch.int32)
            starts = torch.tensor(offsets, device="cuda", dtype=torch.int64)
            decode(query, cache, tables, lengths, max_tokens=784, starts=starts)
        elif entry == "grouped_decode":
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
