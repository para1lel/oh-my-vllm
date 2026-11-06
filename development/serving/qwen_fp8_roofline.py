"""Qwen FP8 components and strict archived-HTTP resource accounting.

resource_floor retains the partial components-v1 scope and explicit BA precision.
lower_bound validates loaded identities, execution, parser acceptance and HTTP
usage before canonical scheduling. Missing evidence blocks a result. AgentX
prefix and tree accounting require evidence this cold-HTTP path does not accept.
FP8 stores KV; QK/PV use BF16. Logical cache traffic alone earns no HBM credit.
Unknown instruction rates are excluded. Full HTTP time remains the denominator.

resource_floor 保留 components-v1 的局部范围和显式 BA 精度。
lower_bound 先核验已加载身份、执行、解析器端点及 HTTP usage, 再做语义调度。
缺少必要证据时阻塞结果。此冷 HTTP 路径不接受缺少完整证据的 AgentX 前缀与树。
FP8 存储 KV, QK/PV 使用 BF16。逻辑缓存流量不直接获得 HBM credit。
未知指令率不计分, 完整 HTTP 时间保留为分母。
"""

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType

from .qwen_fp8_math import TARGET_MATRICES, resource_reference
from .qwen_fp8_schedule import canonical_passes
from .qwen_fp8_trace import VERSION as HTTP_VERSION
from .qwen_fp8_trace import Blocked, digest, validate

VERSION = "qwen-fp8-components-v1"
PAGE_TOKENS = 128
CHECKPOINT_TOKENS = 8192
PAGES_PER_CHECKPOINT = 64
RATES = MappingProxyType(
    dict(hbm_bytes=7.7e12, fp8_flops=4.5e15, bf16_flops=2.25e15, fp32_flops=75e12)
)
OMISSIONS = (
    "HTTP canonical budget scheduler and source/trace/usage identity validation",
    "AgentX independent KV/GDN prefix hits, session ownership and tree closure",
    "MTP proposal/verification-only work, acceptance and sampling",
    "GDN recurrence/WY, norms, gates, RoPE, embedding and hidden checkpoints",
    "GDN BA projections without explicit modeled-step precision",
    "All weight/weight-scale reads and output vocabulary projections",
    "FP8 absmax, scale construction, casts, rounding and scale application",
    "Temporary buffers, metadata/token transfers, ZMQ/HTTP and cache residency",
)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self):
        _integer(self.start, "start")
        _integer(self.end, "end", self.start + 1)

    @property
    def tokens(self):
        return self.end - self.start


def checkpoint_writes(span):
    """Count interval checkpoints and its latest tail once. 检查点与末尾去重。"""
    if not isinstance(span, Span):
        raise ValueError("checkpoint writes require a Span")
    return (
        span.end // CHECKPOINT_TOKENS
        - span.start // CHECKPOINT_TOKENS
        + int(span.end % CHECKPOINT_TOKENS != 0)
    )


def kv_storage(tokens, *, layers=16):
    _integer(tokens, "tokens")
    if type(layers) is not int or layers not in (1, 16, 17):
        raise ValueError("KV layers must be 1, 16 or 17")
    codes = tokens * layers * 2 * 4 * 256
    scales = tokens * layers * 2 * 4 * 4
    return dict(codes=codes, scales=scales, total=codes + scales)


def pool_storage(checkpoints, *, mode):
    _integer(checkpoints, "checkpoints")
    if mode not in ("ordinary", "mtp4"):
        raise ValueError("mode must be ordinary or mtp4")
    pages = PAGES_PER_CHECKPOINT * checkpoints
    kv = kv_storage(pages * PAGE_TOKENS, layers=16 + int(mode == "mtp4"))
    return dict(
        pages=pages,
        checkpoints=checkpoints,
        kv=kv,
        gdn_bytes=checkpoints * 48 * (48 * 128 * 128 * 4 + 10240 * 3 * 2),
        scope="Pool payload only; private/tail ownership, hidden and scratch omitted",
    )


@dataclass(frozen=True)
class Components:
    kv_read_codes: int = 0
    kv_read_scales: int = 0
    kv_write_codes: int = 0
    kv_write_scales: int = 0
    gdn_read_recurrent: int = 0
    gdn_read_convolution: int = 0
    gdn_write_recurrent: int = 0
    gdn_write_convolution: int = 0
    fp8_flops: int = 0
    bf16_flops: int = 0
    fp32_flops: int = 0

    @property
    def necessary_bytes(self):
        return sum(value for name, value in asdict(self).items() if "flops" not in name)


def _sum(items):
    totals = dict.fromkeys(Components.__dataclass_fields__, 0)
    for item in items:
        for name, value in asdict(item).items():
            totals[name] += value
    return Components(**totals)


def target_components(span, *, ba_precision=None):
    """Accepted work; omit BA if its precision is unknown. BA 精度未知时不计。"""
    if not isinstance(span, Span):
        raise ValueError("target components require a Span")
    if ba_precision not in (None, "bf16", "fp32"):
        raise ValueError("BA precision must be bf16, fp32 or None")
    ba = 2 * span.tokens * 96 * 5120 * 48
    read = kv_storage(span.start)
    write = kv_storage(span.tokens)
    saves = checkpoint_writes(span)
    pairs = span.tokens * span.start + span.tokens * (span.tokens + 1) // 2
    recurrent = 48 * 48 * 128 * 128 * 4
    convolution = 48 * 10240 * 3 * 2
    projection = sum(
        layers * columns * width for layers, columns, width in TARGET_MATRICES
    )
    return Components(
        kv_read_codes=read["codes"],
        kv_read_scales=read["scales"],
        kv_write_codes=write["codes"],
        kv_write_scales=write["scales"],
        gdn_read_recurrent=int(span.start > 0) * recurrent,
        gdn_read_convolution=int(span.start > 0) * convolution,
        gdn_write_recurrent=saves * recurrent,
        gdn_write_convolution=saves * convolution,
        fp8_flops=2 * span.tokens * projection,
        bf16_flops=4 * pairs * 24 * 256 * 16 + (ba if ba_precision == "bf16" else 0),
        fp32_flops=3 * pairs * 24 * 16
        + span.tokens * 7 * 10240 * 48
        + (ba if ba_precision == "fp32" else 0),
    )


def mtp_context_components(span):
    if not isinstance(span, Span) or span.start < 1:
        raise ValueError("MTP context positions start at one")
    write = kv_storage(span.tokens, layers=1)
    return Components(
        kv_write_codes=write["codes"],
        kv_write_scales=write["scales"],
        fp8_flops=2 * span.tokens * 2048 * 5120,
        bf16_flops=2 * span.tokens * 5120 * 10240,
    )


def _request_spans(request, mode):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    prompt = _integer(request.get("input_tokens"), "input_tokens", 1)
    output = _integer(request.get("output_tokens"), "output_tokens", 1)
    if (
        request.get("cached_prefix_tokens") != 0
        or type(request.get("cached_prefix_tokens")) is not int
    ):
        raise ValueError(
            "HTTP component reference requires an explicit uncached prefix"
        )
    steps = request.get("accepted_decode_tokens")
    if not isinstance(steps, list):
        raise ValueError("accepted_decode_tokens must be a list")
    limit = 5 if mode == "mtp4" else 1
    for count in steps:
        if _integer(count, "accepted decode tokens", 1) > limit:
            raise ValueError("accepted decode step exceeds mode capacity")
    if sum(steps) != output - 1:
        raise ValueError(
            "accepted steps must match HTTP output usage after first sample"
        )
    spans, end = [Span(0, prompt)], prompt
    for count in steps:
        spans.append(Span(end, end + count))
        end += count
    return spans, Span(end, end + 1)


def _floor(components):
    return max(
        components.necessary_bytes / RATES["hbm_bytes"],
        components.fp8_flops / RATES["fp8_flops"]
        + components.bf16_flops / RATES["bf16_flops"],
        components.fp32_flops / RATES["fp32_flops"],
    )


def resource_floor(workload):
    """Return an aggregate component floor, without serial kernel timing.

    返回聚合资源下界,不叠加尚未冻结的融合 kernel 时间。
    accepted_decode_tokens counts emitted tokens after the prefill sample.
    Exact-end target computation consumes the final emitted token once.
    MTP context persists positions 1..exact_end-1; draft/private state is excluded.
    Optional target_ba_precisions lists prefill, decode steps, then exact-end.
    Each value is bf16, fp32 or None, supplied from the modeled dispatch.
    A missing list omits all BA. A None entry omits that step's BA.
    Accepted counts do not determine batch dispatch.
    target_ba_precisions 依次给出 prefill、decode 步骤和 exact-end 的 BA 精度。
    每项为 bf16、fp32 或 None,由建模分派提供。
    未提供列表时省略全部 BA,None 项省略对应步骤的 BA。
    已接受数量不能决定批次分派。
    """
    if not isinstance(workload, dict) or workload.get("workload") != "serving-tools-v1":
        raise ValueError("workload must bind serving-tools-v1")
    mode = workload.get("mode")
    if mode not in ("ordinary", "mtp4"):
        raise ValueError("mode must be ordinary or mtp4")
    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a nonempty list")
    regular, closing, contexts, identities = [], [], [], set()
    for ordinal, request in enumerate(requests):
        spans, tail = _request_spans(request, mode)
        identity = _integer(request.get("id"), "id", 1)
        if (
            identity in identities
            or type(request.get("ordinal")) is not int
            or request["ordinal"] != ordinal
        ):
            raise ValueError(
                "request IDs must be distinct and ordinals contiguous in client order"
            )
        response = request.get("response_id")
        if (
            not isinstance(response, str)
            or not response.startswith("chatcmpl_")
            or response.rsplit("_", 1)[-1] != str(identity)
        ):
            raise ValueError("response ID must bind HTTP and trace IDs")
        identities.add(identity)
        precisions = request.get("target_ba_precisions", [None] * (len(spans) + 1))
        if not isinstance(precisions, list) or len(precisions) != len(spans) + 1:
            raise ValueError("target_ba_precisions must cover all steps and exact-end")
        regular.extend(
            target_components(span, ba_precision=precision)
            for span, precision in zip(spans, precisions[:-1], strict=True)
        )
        closing.append(target_components(tail, ba_precision=precisions[-1]))
        if mode == "mtp4" and tail.end > 1:
            contexts.append(mtp_context_components(Span(1, tail.end)))
    groups = dict(
        target=_sum(regular),
        compute_only_exact_end=_sum(closing),
        mtp_context=_sum(contexts),
    )
    total = _sum(groups.values())
    return dict(
        accounting=VERSION,
        components={name: asdict(value) for name, value in groups.items()},
        total=asdict(total),
        necessary_bytes=total.necessary_bytes,
        modeled_resource_floor_seconds=_floor(total),
        complete_acceptance_calculator=False,
        omissions=list(OMISSIONS),
        traffic_assumption="Compulsory bytes cross HBM once; residency is unproved",
        work_contract="Accepted steps and one additional exact-end target token",
    )


def meets_efficiency_target(reference_seconds, http_seconds):
    """Use the inclusive 40% gate. 使用包含 40% 的门槛。"""
    for value in (reference_seconds, http_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("reference and HTTP seconds must be finite and positive")
    return reference_seconds / http_seconds >= 0.40


def lower_bound(events, manifest, artifacts, responses, *, tokenizer=None):
    """Use archived Rust-validated execution and full HTTP evidence, or block."""
    try:
        states = validate(events, manifest, artifacts, responses, tokenizer)
        lengths = {
            (row.manifest["input_tokens"], row.manifest["output_tokens"])
            for row in states.values()
        }
        prompts = {32768} if manifest["mode"] == "ordinary" else {32768, 65536, 131072}
        if (
            len(states) not in (1, 2, 4)
            or len(lengths) != 1
            or any(
                prompt not in prompts or output not in (64, 8192)
                for prompt, output in lengths
            )
        ):
            raise Blocked("request shape is outside the frozen 24-case HTTP matrix")
        passes = canonical_passes(states, manifest["mode"])
        calculation = resource_reference(passes, target_components, artifacts["device"])
        timer = manifest["timer"]
        return dict(
            accounting=HTTP_VERSION,
            status="calculated",
            **calculation,
            http_seconds=timer["finished"] - timer["started"],
            efficiency=calculation["seconds"] / (timer["finished"] - timer["started"]),
            manifest_id=digest(manifest),
            trace_id=digest(events),
            acceptance={
                key: dict(
                    prompt=state.manifest["input_tokens"],
                    output=state.manifest["output_tokens"],
                    accepted_end=state.accepted_end,
                )
                for key, state in states.items()
            },
        )
    except (Blocked, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(
            accounting=HTTP_VERSION,
            status="blocked",
            reasons=[str(error)],
            seconds=None,
            efficiency=None,
        )
