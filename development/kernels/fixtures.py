"""Repeatable full-operation fixtures for the static maximum workload cases."""

import importlib

import torch

from development.kernels.timing import reference_operator

ENTRIES = {
    "prepare_attention": ("attention_prepare", "prepare_attention"),
    "prepare_context": ("partial_attention", "prepare_context"),
    "prepare_query": ("partial_attention", "prepare_query"),
    "norm": ("normalization", "rms_norm"),
    "add_norm": ("normalization", "add_rms_norm"),
    "add_norm_fp8_linear": ("fp8", "add_norm_linear"),
    "fp8_linear": ("fp8", "linear"),
    "gated_norm": ("normalization", "rms_norm"),
    "norm_rope": ("normalization", "rms_rotary"),
    "quant": ("fp8", "quantize"),
    "silu_quant": ("fp8", "quantize"),
    "gates": ("elementwise", "delta_gates"),
    "qk": ("gdn", "normalize_qk"),
    "gdn_prefill": ("gdn", "prefill"),
    "recurrent": ("gdn", "recurrent"),
    "convolution": ("convolution", "causal_conv"),
    "append": ("attention", "append"),
    "attention": ("decode_attention", "decode"),
    "dspark_rms_norm": ("dspark_attention", "rms_norm"),
    "dspark_norm_rope": ("dspark_attention", "normalize_rope"),
    "dspark_append": ("dspark_attention", "append"),
    "dspark_attention": ("dspark_attention", "attention"),
}


def fixture(
    config, *, seed=784, observe=False, shared_append_destination=False, qk_inputs=False
):
    """Share immutable inputs; allocate isolated mutable destination pools.

    Sources never alias written destinations, so warmup/capture/replay preserve
    identical effective inputs. Production-required snapshots stay in the call.
    A local CUDA generator makes construction reproducible for every caller.
    DSpark append timing can share a cache because immutable K/V and slots
    repeat the same writes. Numerical verification keeps independent caches.
    """
    if shared_append_destination and (
        observe or config["operation"] != "dspark_append"
    ):
        raise ValueError(
            "shared append destinations are only for timing after output verification"
        )
    if qk_inputs and (observe or config["operation"] != "qk"):
        raise ValueError("Q/K timing inputs require prior independent verification")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    operation = config["operation"]
    module, entry = ENTRIES[operation]
    reference = reference_operator(module, entry)
    candidate = getattr(importlib.import_module(f"oh_my_vllm.kernels.{module}"), entry)
    n = config.get("tokens", 0)
    kwargs = {}

    def package(reference_call, candidate_call, witnesses=()):
        if observe:
            return reference_call, candidate_call, witnesses
        if qk_inputs:
            return reference_call, candidate_call, args
        return reference_call, candidate_call

    def random(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    def metadata(values, dtype=torch.int64):
        return torch.tensor(values, device="cuda", dtype=dtype)

    if operation == "dspark_rms_norm":
        args = (random(n, 5120), random(5120))
    elif operation == "dspark_norm_rope":
        # The supplied frequency vector matches the production YaRN blend.
        # Construction remains outside both full-operation measurements.
        from oh_my_vllm.models.dspark import DSparkConfig, yarn_frequencies

        frequency, factor = yarn_frequencies(DSparkConfig(), "cuda")
        positions = torch.arange(n, device="cuda", dtype=torch.int64)
        if config["draft"]:
            positions = positions % 7
        positions = positions + config["position_base"]
        args = (
            random(n, config["heads"], 128),
            random(128, dtype=torch.float32),
            positions,
            frequency,
            factor,
        )
    elif operation == "dspark_append":
        key, value = random(n, 8, 128), random(n, 8, 128)
        slots = torch.arange(784, 784 + n, device="cuda", dtype=torch.int64)
        pool = torch.empty(
            (n + 1567) // 784, 2, 784, 8, 128, device="cuda", dtype=torch.bfloat16
        )
        other = pool if shared_append_destination else torch.empty_like(pool)
        return package(
            lambda: reference(pool, key, value, slots),
            lambda: candidate(other, key, value, slots),
            (
                (
                    "written_cache",
                    lambda: pool[slots // 784, :, slots % 784],
                    lambda: other[slots // 784, :, slots % 784],
                ),
            ),
        )
    elif operation == "dspark_attention":
        batch, length = config["batch"], config["length"]
        pages = (length + 783) // 784
        pool = random(1 + batch * pages, 2, 784, 8, 128)
        table = torch.arange(
            1, 1 + batch * pages, device="cuda", dtype=torch.int32
        ).view(batch, pages)
        contexts = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        args = (
            random(batch, 7, 32, 128),
            pool,
            table,
            contexts,
            random(batch, 7, 8, 128),
            random(batch, 7, 8, 128),
        )
    elif operation in ("prepare_attention", "prepare_context", "prepare_query"):
        packed = random(
            n,
            {
                "prepare_attention": 14336,
                "prepare_context": 2048,
                "prepare_query": 12288,
            }[operation],
        )
        qw, kw = random(256, dtype=torch.float32), random(256, dtype=torch.float32)
        dtype = getattr(torch, config["index_dtype"])
        positions = torch.arange(n, device="cuda", dtype=dtype)
        slots = positions + 784
        if operation == "prepare_query":
            return package(
                lambda: reference(packed, qw, positions),
                lambda: candidate(packed, qw, positions),
            )
        pool = torch.empty(1400, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
        other = torch.empty_like(pool)

        def run(function, destination):
            if operation == "prepare_context":
                return function(packed, kw, positions, destination, slots)
            return function(packed, qw, kw, positions, destination, slots)

        return package(
            lambda: run(reference, pool),
            lambda: run(candidate, other),
            (
                (
                    "written_key",
                    lambda: pool[slots // 784, 0, slots % 784],
                    lambda: other[slots // 784, 0, slots % 784],
                ),
                (
                    "written_value",
                    lambda: pool[slots // 784, 1, slots % 784],
                    lambda: other[slots // 784, 1, slots % 784],
                ),
            ),
        )
    elif operation == "fp8_linear":
        args = (
            random(n, config["width"] * (2 if config["silu"] else 1)),
            random(config["columns"], config["width"]).to(torch.float8_e4m3fn),
            torch.rand(
                config["columns"] // 128,
                config["width"] // 128,
                device="cuda",
                generator=generator,
            )
            * 0.01,
        )
        kwargs = dict(silu_gate=config["silu"])
    elif operation == "add_norm_fp8_linear":
        args = (
            random(n, 5120),
            random(n, 5120),
            random(5120, dtype=torch.float32),
            random(config["columns"], 5120).to(torch.float8_e4m3fn),
            torch.rand(config["columns"] // 128, 40, device="cuda", generator=generator)
            * 0.01,
        )
    elif operation in ("norm", "add_norm"):
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
    elif operation == "gdn_prefill":
        counts = config["counts"]
        n = sum(counts)
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        packed = random(n, 10240)
        args = (
            packed[:, :2048].view(n, 16, 128),
            packed[:, 2048:4096].view(n, 16, 128),
            packed[:, 4096:].view(n, 48, 128),
            -torch.rand(n, 48, device="cuda", generator=generator) * 0.1,
            torch.rand(n, 48, device="cuda", generator=generator),
            random(len(counts), 48, 128, 128, dtype=torch.float32) * 0.01,
            metadata(offsets, torch.int32),
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
                    [start + count - 1] if count > 8 else range(start, start + count)
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
        if observe:
            written = (
                metadata(sorted({slot for slot in writes if slot >= 0}))
                if operation == "convolution"
                else other[8]
            )
            witnesses = (
                (
                    "written_state",
                    lambda: pool.index_select(0, written),
                    lambda: other[pool_index].index_select(0, written),
                ),
            )
        else:
            witnesses = ()
        return package(lambda: reference(*args), lambda: candidate(*other), witnesses)
    elif operation == "append":
        k, v = random(n, 4, 256), random(n, 4, 256)
        slots = torch.arange(784, 784 + n, device="cuda", dtype=torch.int64)
        pool = torch.empty(1400, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
        other = torch.empty_like(pool)
        return package(
            lambda: reference(pool, k, v, slots),
            lambda: candidate(other, k, v, slots),
            (
                (
                    "written_cache",
                    lambda: pool[slots // 784, :, slots % 784],
                    lambda: other[slots // 784, :, slots % 784],
                ),
            ),
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
            first=config.get("first", 1),
            max_tokens=config["max_tokens"],
            starts=metadata(list(range(0, batch * queries + 1, queries)), torch.int32)
            if config["grouped"]
            else None,
        )
        if queries > 5:
            # The frozen grouped tile covers five target queries. Its original
            # independent-row path computes all eight without changing source.
            reference_kwargs = kwargs | {"starts": None}
            candidate_kwargs = kwargs | {"max_query_len": 8}
            return package(
                lambda: reference(*args, **reference_kwargs),
                lambda: candidate(*args, **candidate_kwargs),
            )
    else:
        raise ValueError(operation)
    return package(
        lambda: reference(*args, **kwargs), lambda: candidate(*args, **kwargs)
    )
