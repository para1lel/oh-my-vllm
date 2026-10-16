from collections import defaultdict
from dataclasses import dataclass
from math import ceil

from oh_my_vllm.kernels.specification import MODEL


@dataclass(frozen=True)
class Profile:
    name: str
    probability: float
    batch: int
    text_tokens: int
    output_tokens: int
    mtp: bool = False
    height: int = 0
    width: int = 0
    frames: int = 0

    @property
    def patches_per_frame(self):
        return self.height * self.width // MODEL["patch"] ** 2

    @property
    def frame_groups(self):
        return ceil(self.frames / MODEL["temporal_patch"])

    @property
    def vision_tokens(self):
        return self.patches_per_frame * self.frame_groups

    @property
    def input_tokens(self):
        return self.text_tokens + self.vision_tokens // MODEL["spatial_merge"] ** 2


PROFILES = (
    Profile("text_short", 0.20, 1, 784, 128),
    Profile("text_batch", 0.20, 4, 32768, 256),
    Profile("text_long_mtp", 0.20, 1, 131072, 512, True),
    Profile("image", 0.15, 1, 1024, 256, height=448, width=448, frames=1),
    Profile("image_batch_mtp", 0.15, 4, 4096, 256, True, 896, 896, 1),
    Profile("video_mtp", 0.10, 1, 1024, 128, True, 448, 448, 8),
)

CONTEXT_SAMPLES = ((0.5, 1.0),)
REPORT_VERSION = 2
NUMERICAL_POLICY = "small_shapes_v2"


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


@dataclass(frozen=True)
class Case:
    shape: Shape
    calls: float


def numerical_cases():
    shapes = [
        Shape("ffn", rows, batch, 1)
        for rows, batch in ((1, 1), (4, 4), (5, 1), (20, 4), (65, 1))
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
            Shape("gdn_attention", 65, 1, 65, position=770),
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


def stage_cases(profiles=PROFILES):
    counts = defaultdict(float)

    def add(
        name,
        rows,
        batch,
        queries,
        context,
        frequency,
        frame_tokens=0,
        variant="default",
        media_rows=0,
    ):
        if frequency > 0:
            if name != "gqa_attention_output":
                context = 0
            if name not in {"gdn_input", "gdn_prefill", "gdn_recurrent"}:
                queries = 1
            shape = Shape(
                name, rows, batch, queries, context, frame_tokens, variant, media_rows
            )
            counts[shape] += frequency

    def target(profile, queries, context, frequency, prefill=False):
        rows = profile.batch * queries
        for name, layers in (
            ("gdn_input", 47),
            ("gdn_prefill" if prefill else "gdn_recurrent", 48),
            ("gdn_output", 48),
            ("gqa_input", 16),
            ("gqa_attention_output", 16),
            ("ffn_gate_up", 64),
            ("ffn_down", 64),
        ):
            add(name, rows, profile.batch, queries, context, layers * frequency)
        media_rows = profile.batch * profile.vision_tokens // 4 if prefill else 0
        add(
            "gdn_input",
            rows,
            profile.batch,
            queries,
            context,
            frequency,
            variant="embedding_media" if media_rows else "embedding",
            media_rows=media_rows,
        )
        if not prefill:
            add(
                "output_sample",
                rows,
                profile.batch,
                queries,
                0,
                frequency,
                variant="verify" if profile.mtp else "sampling",
            )

    for profile in profiles:
        if profile.probability <= 0 or profile.batch < 1:
            raise ValueError("profile probability and batch must be positive")
        if profile.input_tokens + profile.output_tokens > MODEL["context_tokens"]:
            raise ValueError("profile exceeds the context limit")
        per_request = MODEL["step_tokens"] // (profile.batch * MODEL["block_size"])
        chunk = max(1, per_request) * MODEL["block_size"]
        full, tail = divmod(profile.input_tokens, chunk)
        if full:
            context = min(profile.input_tokens, chunk * (full + 1) // 2)
            target(profile, chunk, context, full * profile.probability, True)
        if tail:
            target(profile, tail, profile.input_tokens, profile.probability, True)
        add(
            "output_sample",
            profile.batch,
            profile.batch,
            1,
            0,
            profile.probability,
            variant="sampling",
        )
        queries = 5 if profile.mtp else 1
        kept_per_step = 4 if profile.mtp else 1
        steps = ceil(profile.output_tokens / kept_per_step)
        for fraction, share in CONTEXT_SAMPLES:
            context = profile.input_tokens + int(profile.output_tokens * fraction)
            target(profile, queries, context, profile.probability * steps * share)
        if profile.mtp:
            frequency = profile.probability * steps
            add("mtp_accept_commit", profile.batch, profile.batch, 5, 0, frequency)
            for queries, count in ((chunk, full), (tail, int(tail > 0))):
                if count:
                    for name in ("mtp_input", "gqa_input"):
                        add(
                            name,
                            profile.batch * queries,
                            profile.batch,
                            queries,
                            0,
                            profile.probability * count,
                            variant="context" if name == "mtp_input" else "kv_only",
                        )
            for name in ("mtp_input", "gqa_input"):
                add(
                    name,
                    profile.batch * kept_per_step,
                    profile.batch,
                    kept_per_step,
                    0,
                    profile.probability * max(0, steps - 1),
                    variant="context" if name == "mtp_input" else "kv_only",
                )
            for name, layers in (
                ("gqa_attention_output", 1),
                ("ffn_gate_up", 1),
                ("ffn_down", 1),
                ("output_sample", 1),
            ):
                for fraction, share in CONTEXT_SAMPLES:
                    context = profile.input_tokens + int(
                        profile.output_tokens * fraction
                    )
                    add(
                        name,
                        profile.batch,
                        profile.batch,
                        1,
                        context if name == "gqa_attention_output" else 0,
                        4 * layers * frequency * share,
                        variant="mtp",
                    )
            add("mtp_input", profile.batch, profile.batch, 1, 0, 3 * frequency)
            add(
                "gqa_input",
                profile.batch,
                profile.batch,
                1,
                0,
                3 * frequency,
                variant="mtp",
            )
            add(
                "gqa_input",
                profile.batch,
                profile.batch,
                1,
                0,
                frequency,
                variant="mtp_query",
            )
        if profile.vision_tokens:
            frequency = profile.probability
            rows = profile.batch * profile.vision_tokens
            for name, layers, divisor in (
                ("vision_patch_input", 1, 1),
                ("vision_qkv", 27, 1),
                ("vision_attention_output", 27, 1),
                ("vision_ffn_up", 27, 1),
                ("vision_ffn_down", 27, 1),
                ("vision_merge_up", 1, 4),
                ("vision_merge_down", 1, 4),
            ):
                add(
                    name,
                    rows // divisor,
                    profile.batch,
                    profile.patches_per_frame,
                    0,
                    frequency * layers,
                    profile.patches_per_frame,
                )
    return tuple(Case(shape, calls) for shape, calls in sorted(counts.items()))


SERVING_OPERATORS = {
    "gdn_attention": ("gdn_input", "gdn_recurrent|gdn_prefill", "gdn_output"),
    "fa_attention": ("gqa_input", "gqa_attention_output"),
    "ffn": ("ffn_gate_up", "ffn_down"),
    "sample": ("output_sample", "mtp_accept_commit"),
    "mtp_context": ("mtp_input", "gqa_input:kv_only"),
    "mtp_draft": (
        "mtp_input?",
        "gqa_input",
        "gqa_attention_output",
        "ffn_gate_up",
        "ffn_down",
        "output_sample",
    ),
    "vision_block": (
        "vision_patch_input?",
        "vision_qkv",
        "vision_attention_output",
        "vision_ffn_up",
        "vision_ffn_down",
        "vision_merge_up?",
        "vision_merge_down?",
    ),
}


def modeled_cases(profiles=PROFILES):
    counts = defaultdict(float)
    stages = stage_cases(profiles)
    for case in stages:
        shape = case.shape
        name = shape.operator
        if name == "gdn_input":
            operator = "gdn_attention"
        elif name == "gqa_input" and shape.variant == "default":
            operator = "fa_attention"
        elif name == "ffn_gate_up" and shape.variant == "default":
            operator = "ffn"
        elif name == "output_sample" and shape.variant != "mtp":
            operator = "sample"
        elif name == "mtp_input" and shape.variant == "context":
            operator = "mtp_context"
        elif name == "gqa_attention_output" and shape.variant == "mtp":
            for variant, share in (("full", 0.75), ("query_only", 0.25)):
                counts[
                    Shape(
                        "mtp_draft",
                        shape.rows,
                        shape.batch,
                        1,
                        shape.context,
                        variant=variant,
                    )
                ] += case.calls * share
            continue
        elif name == "vision_qkv":
            for variant, layers in (("input", 1), ("default", 25), ("merge", 1)):
                counts[
                    Shape(
                        "vision_block",
                        shape.rows,
                        shape.batch,
                        1,
                        frame_tokens=shape.frame_tokens,
                        variant=variant,
                    )
                ] += case.calls * layers / 27
            continue
        else:
            continue
        if operator == "fa_attention":
            matching = [
                item
                for item in stages
                if item.shape.operator == "gqa_attention_output"
                and item.shape.rows == shape.rows
                and item.shape.batch == shape.batch
                and item.shape.variant == shape.variant
            ]
            for attention in matching:
                counts[
                    Shape(operator, shape.rows, shape.batch, 1, attention.shape.context)
                ] += attention.calls
            continue
        counts[
            Shape(
                operator,
                shape.rows,
                shape.batch,
                shape.queries,
                shape.context,
                shape.frame_tokens,
                shape.variant,
                shape.media_rows,
            )
        ] += case.calls
    return tuple(Case(shape, calls) for shape, calls in sorted(counts.items()))
