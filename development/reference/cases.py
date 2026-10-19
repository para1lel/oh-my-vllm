from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Shape:
    operator: str
    rows: int
    batch: int
    queries: int
    context: int = 0
    frame_tokens: int = 0
    variant: str = "default"
    media_rows: int = 0
    position: int = 0


def numerical_cases():
    shapes = [
        Shape("ffn", rows, batch, 1)
        for rows, batch in (
            (1, 1),
            (4, 4),
            (5, 1),
            (20, 4),
            (65, 1),
            (129, 1),
            (144, 1),
        )
    ]
    shapes.extend(
        Shape("gdn_attention", rows, batch, queries)
        for rows, batch, queries in ((1, 1, 1), (4, 4, 1), (5, 1, 5), (20, 4, 5))
    )
    shapes.extend(
        (
            Shape("gdn_attention", 1, 1, 1, variant="embedding"),
            Shape("gdn_attention", 65, 1, 65, variant="embedding_media", media_rows=2),
            Shape("gdn_attention", 65, 1, 65),
            Shape("gdn_attention", 129, 1, 129, position=770),
            Shape("gdn_attention", 0, 1, 0),
        )
    )
    shapes.extend(
        Shape("fa_attention", rows, batch, queries, 256)
        for rows, batch, queries in (
            (1, 1, 1),
            (4, 4, 1),
            (5, 1, 5),
            (20, 4, 5),
            (33, 1, 33),
        )
    )
    shapes.append(Shape("fa_attention", 5, 1, 5, 786))
    shapes.extend(
        Shape("sample", rows, batch, 1, variant=variant)
        for rows, batch, variant in (
            (1, 1, "sampling"),
            (4, 4, "sampling"),
            (5, 1, "verify"),
            (20, 4, "verify"),
        )
    )
    shapes.extend(
        Shape("mtp_context", rows, batch, 1, variant="context")
        for rows, batch in ((4, 1), (16, 4))
    )
    shapes.extend(
        Shape("mtp_draft", rows, batch, 1, 256, variant=variant)
        for rows, batch in ((1, 1), (4, 4))
        for variant in ("full", "query_only")
    )
    shapes.extend(
        Shape("vision_block", 392, 1, 1, frame_tokens=196, variant=variant)
        for variant in ("input", "default", "merge")
    )
    return tuple(shapes)
