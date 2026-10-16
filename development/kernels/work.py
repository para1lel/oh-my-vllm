from .roofline import Work


def linear(rows, columns, width, *, fp8=False):
    weights = columns * width * (1 if fp8 else 2)
    scales = 4 * ((columns + 127) // 128) * ((width + 127) // 128) if fp8 else 0
    return Work(
        weights + scales,
        fp8_flops=2 * rows * columns * width if fp8 else 0,
        bf16_flops=0 if fp8 else 2 * rows * columns * width,
    )


def add(*items):
    return Work(
        *(sum(getattr(item, field) for item in items) for field in vars(Work(0)))
    )


def attention(
    shape, *, vision=False, queries=None, first_token=0, cached_queries=False
):
    rows, batch = shape.rows, shape.batch
    if vision:
        flops = 2 * rows * shape.frame_tokens * 1152
        return Work(0, bf16_flops=2 * flops)
    if queries is None:
        count = rows // batch
        queries = [(max(0, shape.context - count), count)] * batch
    pairs = sum(
        count * (start - first_token) + count * (count + 1) // 2
        for start, count in queries
    )
    return Work(
        sum(
            max(0, start + (count if cached_queries else 0) - first_token)
            for start, count in queries
        )
        * 4
        * 256
        * 4,
        bf16_flops=4 * pairs * 24 * 256,
    )


def ffn(rows):
    return add(linear(rows, 34816, 5120, fp8=True), linear(rows, 5120, 17408, fp8=True))


def work_for(shape, *, queries=None, state_writes=None):
    name, rows, batch = shape.operator, shape.rows, shape.batch
    if name == "gdn_attention":
        writes = state_writes
        if writes is None:
            writes = (
                rows if shape.queries <= 5 else batch * ((shape.queries + 783) // 784)
            )
        state = (batch + writes) * (48 * 128 * 128 + 10240 * 3) * 2
        external = (6 if shape.variant.startswith("embedding") else 8) * rows * 5120
        return add(
            linear(rows, 16384, 5120, fp8=True),
            linear(rows, 96, 5120),
            linear(rows, 5120, 6144, fp8=True),
            Work(
                external + state + 4 * (5120 + 128 + 96) + 10240 * 4 * 2,
                fp32_flops=rows * 48 * (7 * 128 * 128 + 3 * 128),
            ),
        )
    if name == "fa_attention":
        return add(
            linear(rows, 14336, 5120, fp8=True),
            linear(rows, 5120, 6144, fp8=True),
            attention(shape, queries=queries),
            Work(8 * rows * 5120 + rows * 4096 + 4 * (5120 + 512)),
        )
    if name == "ffn":
        return add(ffn(rows), Work(8 * rows * 5120 + 4 * 5120))
    if name == "sample":
        verification = shape.variant == "verify"
        return add(
            linear(rows, 248320, 5120),
            Work(
                4 * rows * 5120
                + 4 * 5120
                + rows * 12
                + (batch * (248320 * 4 + 4 * 12 + 16 + 8) if verification else 0),
                fp32_flops=2 * batch * 248320 if verification else 0,
                fp64_flops=11 * batch if verification else 0,
            ),
        )
    if name == "mtp_context":
        return add(
            linear(rows, 5120, 10240),
            linear(rows, 2048, 5120, fp8=True),
            Work(
                4 * rows * 5120 + rows * 4096 + 4 * (3 * 5120 + 256),
                fp32_flops=rows * (8 * 5120 + 8),
            ),
        )
    if name == "mtp_draft":
        full = shape.variant == "full"
        return add(
            linear(rows, 5120, 10240) if full else Work(0),
            linear(rows, 14336 if full else 12288, 5120, fp8=True),
            linear(rows, 5120, 6144, fp8=True),
            ffn(rows),
            linear(rows, 248320, 5120),
            attention(shape, queries=queries, first_token=1, cached_queries=not full),
            Work(
                (6 if full else 4) * rows * 5120
                + 4 * rows * 248320
                + (rows * 4096 if full else 0)
                + 4 * ((6 if full else 4) * 5120 + 512),
                fp32_flops=rows * (8 * 5120 + 8) if full else 0,
            ),
        )
    if name == "vision_block":
        final = shape.variant == "merge"
        first = shape.variant == "input"
        external = (4 if first or final else 8) * rows * 1152
        result = add(
            linear(rows, 3456, 1152),
            linear(rows, 1152, 1152),
            linear(rows, 4304, 1152),
            linear(rows, 1152, 4304),
            attention(shape, vision=True),
            Work(external + 2 * (4 * 1152 + 3456 + 1152 + 4304 + 1152)),
        )
        if first:
            # One source frame is a conservative bound for repeated or video frames.
            result = add(
                result,
                linear(rows, 1152, 1536),
                Work(rows * 768 + 1152 * 4),
            )
        if final:
            result = add(
                result,
                linear(rows // 4, 4608, 4608),
                linear(rows // 4, 5120, 4608),
                Work(1152 * 4 + (4608 + 5120) * 2 + rows // 4 * 5120 * 2),
            )
        return result
    raise ValueError(f"unknown serving operator: {name}")
