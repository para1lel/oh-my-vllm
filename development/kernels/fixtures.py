"""Repeatable full-operation fixtures for the static maximum workload cases."""

import importlib

import torch

from development.kernels.timing import reference_operator

ENTRIES = {
    "prepare_attention": ("attention_prepare", "prepare_attention"),
    "norm": ("normalization", "rms_norm"),
    "add_norm": ("normalization", "add_rms_norm"),
    "gated_norm": ("normalization", "rms_norm"),
    "norm_rope": ("normalization", "rms_rotary"),
    "quant": ("fp8", "quantize"),
    "silu_quant": ("fp8", "quantize"),
    "gates": ("elementwise", "delta_gates"),
    "qk": ("gdn", "normalize_qk"),
    "recurrent": ("gdn", "recurrent"),
    "convolution": ("convolution", "causal_conv"),
    "append": ("attention", "append"),
    "attention": ("decode_attention", "decode"),
}


def fixture(config, *, seed=784):
    """Share immutable inputs; allocate isolated mutable destination pools.

    Sources never alias written destinations, so warmup/capture/replay preserve
    identical effective inputs. Production-required snapshots stay in the call.
    A local CUDA generator makes construction reproducible for every caller.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    operation = config["operation"]
    module, entry = ENTRIES[operation]
    reference = reference_operator(module, entry)
    candidate = getattr(importlib.import_module(f"oh_my_vllm.kernels.{module}"), entry)
    n = config.get("tokens", 0)
    kwargs = {}

    def random(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    def metadata(values, dtype=torch.int64):
        return torch.tensor(values, device="cuda", dtype=dtype)

    if operation == "prepare_attention":
        packed = random(n, 14336)
        qw, kw = random(256, dtype=torch.float32), random(256, dtype=torch.float32)
        dtype = getattr(torch, config["index_dtype"])
        positions = torch.arange(n, device="cuda", dtype=dtype)
        slots = positions + 784
        pool = torch.empty(1400, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
        other = torch.empty_like(pool)
        return lambda: reference(
            packed, qw, kw, positions, pool, slots
        ), lambda: candidate(packed, qw, kw, positions, other, slots)
    if operation in ("norm", "add_norm"):
        x, w = random(n, 5120), random(5120, dtype=torch.float32)
        args = (x, w) if operation == "norm" else (x, random(n, 5120), w)
    elif operation == "gated_norm":
        x = random(n, 48, 128)
        gate = random(n, 16384)[:, 10240:].view(n, 48, 128)
        args, kwargs = (x, random(128, dtype=torch.float32)), {"gate": gate}
    elif operation == "norm_rope":
        heads = config["heads"]
        packed = random(n, 14336)
        x = (
            packed[:, :12288].view(n, 24, 512)[..., :256]
            if heads == 24
            else packed[:, 12288:13312].view(n, 4, 256)
        )
        args = (x, random(256, dtype=torch.float32), torch.arange(n, device="cuda"))
    elif operation in ("quant", "silu_quant"):
        fused = operation == "silu_quant"
        args = (random(n, config["width"] * (2 if fused else 1)),)
        kwargs = dict(column_major=config["column"], silu_gate=fused)
    elif operation == "gates":
        args = (
            random(n, 96),
            random(48, dtype=torch.float32),
            random(48, dtype=torch.float32),
        )
    elif operation == "qk":
        packed = random(n, 10240)
        args = (
            packed[:, :2048].view(n, 16, 128),
            packed[:, 2048:4096].view(n, 16, 128),
        )
    elif operation in ("recurrent", "convolution"):
        counts = config["counts"]
        n, sequences = sum(counts), len(counts)
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + count)
        reads = metadata(list(range(sequences)))
        if operation == "convolution":
            x = random(n, 16384)[:, :10240]
            pool = random(128, 10240, 3)
            writes = [-1] * n
            next_slot = sequences
            for start, count in zip(starts[:-1], counts, strict=True):
                positions = (
                    [start + count - 1] if count > 5 else range(start, start + count)
                )
                for pos in positions:
                    writes[pos] = next_slot
                    next_slot += 1
            args = (
                x,
                random(10240, 4),
                pool,
                metadata([i for i, count in enumerate(counts) for _ in range(count)]),
                metadata(starts, torch.int32),
                reads,
                metadata(writes),
            )
            pool_index = 2
        else:
            packed = random(n, 10240)
            q, k, v = (
                packed[:, :2048].view(n, 16, 128),
                packed[:, 2048:4096].view(n, 16, 128),
                packed[:, 4096:].view(n, 48, 128),
            )
            pool = random(
                128, 48, 128, 128, dtype=getattr(torch, config["state_dtype"])
            )
            args = (
                q,
                k,
                v,
                -torch.rand(n, 48, device="cuda", generator=generator),
                torch.rand(n, 48, device="cuda", generator=generator),
                pool,
                metadata(starts, torch.int32),
                reads,
                metadata(list(range(sequences, sequences + n))),
            )
            pool_index = 5
        other = list(args)
        other[pool_index] = pool.clone()
        return lambda: reference(*args), lambda: candidate(*other)
    elif operation == "append":
        k, v = random(n, 4, 256), random(n, 4, 256)
        slots = torch.arange(784, 784 + n, device="cuda", dtype=torch.int64)
        pool = torch.empty(1400, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
        other = torch.empty_like(pool)
        return lambda: reference(pool, k, v, slots), lambda: candidate(
            other, k, v, slots
        )
    elif operation == "attention":
        batch, queries, length = config["batch"], config["queries"], config["length"]
        q = random(batch * queries, 24, 256)
        pool = random(1400, 2, 784, 4, 256)
        table = torch.zeros(
            batch, 335, device="cuda", dtype=getattr(torch, config["table_dtype"])
        )
        pages = (length + 783) // 784
        for row in range(batch):
            table[row, :pages] = torch.arange(
                1 + row * pages, 1 + (row + 1) * pages, device="cuda"
            )
        table = table.repeat_interleave(queries, dim=0)
        lengths = metadata(
            list(range(length - queries + 1, length + 1)) * batch,
            getattr(torch, config["length_dtype"]),
        )
        args = (q, pool, table, lengths)
        kwargs = dict(
            first=1,
            max_tokens=config["max_tokens"],
            starts=metadata(list(range(0, batch * queries + 1, queries)), torch.int32)
            if config["grouped"]
            else None,
        )
    else:
        raise ValueError(operation)
    return lambda: reference(*args, **kwargs), lambda: candidate(*args, **kwargs)
