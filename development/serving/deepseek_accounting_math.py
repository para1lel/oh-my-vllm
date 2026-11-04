"""DeepSeek target dimensions, resource arithmetic, and communication formulas."""

import hashlib
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise

COLLECTIVE_PROFILE_VERSION = "idle-unlocked-nccl-profile-v1"


DEP4_PROXY_RULE_VERSION = "dep4-bf16-all-to-all-proxy-v1"


WORLD_SIZE = 4


STRATEGIES = ("tp4", "tep4", "dep4")


WORK_FIELDS = (
    "hbm_bytes",
    "fp8_flops",
    "bf16_flops",
    "fp32_flops",
    "nvlink_bytes",
    "host_bytes",
    "collective_seconds",
)


RATE_PROFILE = (
    ("hbm_bytes", 7.7e12, "HBM3e, 7.7 TB/s per GPU"),
    ("fp8_flops", 4.5e15, "B200 dense FP8 tensor"),
    ("bf16_flops", 2.25e15, "B200 dense BF16 tensor"),
    ("fp32_flops", 75e12, "B200 FP32 vector"),
    ("nvlink_bytes", 0.9e12, "NVLink 5, 900 GB/s per direction"),
    ("host_bytes", 63.02e9, "PCIe 5.0 x16, 63.02 GB/s per direction"),
)


RATES = {name: value for name, value, _source in RATE_PROFILE}


COLLECTIVE_LATENCY_PROFILE_US = {
    ("all_gather", "bf16"): (
        (8, 13.359500095248222),
        (32, 13.451749924570322),
        (128, 13.5777504183352),
        (512, 13.680499978363514),
        (2048, 14.009749982506037),
        (4096, 14.61374992504716),
        (8192, 15.947000123560429),
        (10240, 16.81974995881319),
        (16384, 16.027499921619892),
        (18432, 16.96349959820509),
        (20480, 15.82574937492609),
        (32768, 16.093000769615173),
        (40960, 18.23775004595518),
        (64640, 19.26224958151579),
        (65536, 18.423750065267086),
        (131072, 18.516999669373035),
        (262144, 20.439750514924526),
        (524288, 24.333500303328037),
        (2097152, 56.8195004016161),
    ),
    ("all_gather", "fp32"): (
        (16, 13.288749847561121),
        (64, 13.543750159442425),
        (256, 13.587249908596277),
        (1024, 13.808250427246094),
        (4096, 14.561749994754791),
        (8192, 15.895500779151917),
        (16384, 16.026999801397324),
        (20480, 16.021749936044216),
        (32768, 16.04249980300665),
        (36864, 18.112500198185444),
        (40960, 18.060999922454357),
        (65536, 18.36750004440546),
        (81920, 18.187000416219234),
        (129280, 19.330999813973904),
        (131072, 18.598999828100204),
        (262144, 20.155000500380993),
        (524288, 24.270249530673027),
        (1048576, 31.94749914109707),
        (4194304, 64.7365003824234),
    ),
    ("all_reduce", "bf16"): (
        (8, 15.61524998396635),
        (32, 15.942499972879887),
        (128, 17.07924995571375),
        (512, 17.328000627458096),
        (2048, 17.605249769985676),
        (4096, 17.742750234901905),
        (8192, 17.996249720454216),
        (10240, 18.906750716269016),
        (16384, 18.584500066936016),
        (18432, 19.86700017005205),
        (20480, 20.27474995702505),
        (32768, 19.787750206887722),
        (40960, 21.18975017219782),
        (64640, 21.61649987101555),
        (65536, 19.99824959784746),
        (131072, 20.244000479578972),
        (262144, 22.998749278485775),
        (524288, 23.364750668406487),
        (2097152, 30.089249834418297),
    ),
    ("all_reduce", "fp32"): (
        (16, 15.52075007930398),
        (64, 16.957500018179417),
        (256, 17.20150001347065),
        (1024, 17.42975041270256),
        (4096, 17.74074975401163),
        (8192, 18.02500057965517),
        (16384, 18.37250031530857),
        (20480, 20.078249275684357),
        (32768, 19.728999584913254),
        (36864, 20.818249322474003),
        (40960, 21.060249768197536),
        (65536, 19.8127506300807),
        (81920, 19.733999855816364),
        (129280, 22.23525010049343),
        (131072, 20.187249407172203),
        (262144, 22.318500094115734),
        (524288, 23.900000378489494),
        (1048576, 25.10949969291687),
        (4194304, 40.68350046873093),
    ),
    ("all_to_all", "bf16"): (
        (8, 12.810999993234873),
        (32, 12.592750135809183),
        (128, 12.644750066101551),
        (512, 12.660750187933445),
        (2048, 12.813749723136425),
        (4096, 12.95475009828806),
        (8192, 12.971249874681234),
        (10240, 13.433999847620726),
        (16384, 13.425000011920929),
        (18432, 13.475500047206879),
        (20480, 13.78625025972724),
        (32768, 14.311749953776598),
        (40960, 14.595000073313713),
        (64640, 15.806750394403934),
        (65536, 15.83200041204691),
        (131072, 18.860249780118465),
        (262144, 21.20125014334917),
        (524288, 21.765999495983124),
        (2097152, 25.57575050741434),
    ),
}


COLLECTIVE_PROFILE_RAW_SHA256 = (
    "c6b55ee5354c7ef6fb58606cca9268fcef90bc535935d4f07750029a81506e49"
)


def rate_profile_digest():
    text = "\n".join(f"{name}={value!r}" for name, value, _source in RATE_PROFILE)
    text += f"\ncollective_profile={collective_profile_digest()}"
    text += f"\ncollective_profile_version={COLLECTIVE_PROFILE_VERSION}"
    text += f"\ndep4_proxy_rule={DEP4_PROXY_RULE_VERSION}"
    return hashlib.sha256(text.encode()).hexdigest()


def collective_profile_digest():
    content = repr(COLLECTIVE_LATENCY_PROFILE_US).encode()
    raw_digest = bytes.fromhex(COLLECTIVE_PROFILE_RAW_SHA256)
    return hashlib.sha256(raw_digest + content).hexdigest()


@dataclass(frozen=True, slots=True)
class Work:
    """Resource quantities for validated target work."""

    hbm_bytes: float = 0.0
    fp8_flops: float = 0.0
    bf16_flops: float = 0.0
    fp32_flops: float = 0.0
    nvlink_bytes: float = 0.0
    host_bytes: float = 0.0
    collective_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Group:
    """One serial kernel group."""

    name: str
    work: Work
    repetitions: int = 1


@dataclass(frozen=True, slots=True)
class CriticalPath:
    """Serial group sum after a per-group max over ranks."""

    seconds: float


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """The credited operations of one backbone layer."""

    layer_id: int
    compress_ratio: int
    owns_compressed_kv: bool
    owns_index_keys: bool
    uses_candidate_mask: bool
    has_engram: bool


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Fixed DeepSeek-V4.1-Flash dimensions from the checkpoint configuration."""

    vocab_size: int = 129280
    hidden_size: int = 5120
    num_layers: int = 40
    max_context_tokens: int = 1048576
    num_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    moe_intermediate: int = 2304
    routed_experts: int = 384
    experts_per_token: int = 6
    shared_experts: int = 1
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0, *([2] * 18), *([1] * 20))
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_positions: int = 2048 * 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    engram_layers: tuple[int, ...] = (1, 14)
    engram_columns: int = 24
    engram_head_dim: int = 256
    engram_scale_bytes: int = 8
    draft_tokens: int = 5

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


DEEPSEEK_V41_MODEL = ModelSpec()


ENGRAM_BYTES_PER_ROW = DEEPSEEK_V41_MODEL.engram_head_dim + (
    DEEPSEEK_V41_MODEL.engram_scale_bytes
)


def _checked_positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checked_nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def add_work(*items):
    """Add resource vectors. Empty input returns a zero vector."""
    return Work(
        **{
            name: math.fsum(getattr(item, name) for item in items)
            for name in WORK_FIELDS
        }
    )


def scale_work(work, count):
    """Scale one logical work vector by a positive serial count."""
    count = _checked_positive_int(count, "work scale")
    return Work(**{name: getattr(work, name) * count for name in WORK_FIELDS})


def group_seconds(group):
    """Return one group's maximum resource time, multiplied by repetitions."""
    work = scale_accumulation_work(group.work)
    tensor_seconds = math.fsum(
        getattr(work, name) / RATES[name] for name in ("fp8_flops", "bf16_flops")
    )
    seconds = max(
        work.hbm_bytes / RATES["hbm_bytes"],
        tensor_seconds,
        work.fp32_flops / RATES["fp32_flops"],
        work.nvlink_bytes / RATES["nvlink_bytes"],
        work.host_bytes / RATES["host_bytes"],
        work.collective_seconds,
    )
    return group.repetitions * seconds


def serial_seconds(groups):
    """Return the sum of serial groups for one rank."""
    return math.fsum(group_seconds(group) for group in groups)


def critical_seconds(rank_groups):
    """Take a per-group maximum over ranks, then sum the serial groups."""
    ranks = list(rank_groups)
    if not ranks:
        raise ValueError("critical path requires at least one rank")
    length = len(ranks[0])
    if any(len(groups) != length for groups in ranks):
        raise ValueError("each rank must supply the same number of serial groups")
    names = tuple(group.name for group in ranks[0])
    for groups in ranks:
        if tuple(group.name for group in groups) != names:
            raise ValueError("rank group names and order must match")
    total = math.fsum(
        max(group_seconds(groups[index]) for groups in ranks) for index in range(length)
    )
    if not math.isfinite(total) or total < 0:
        raise ValueError("critical path seconds must be finite and nonnegative")
    return CriticalPath(total)


def validate_execution_strategy(requested, effective):
    """Reject a strategy label that the executed strategy does not satisfy."""
    if requested not in STRATEGIES:
        raise ValueError(f"unknown requested strategy: {requested}")
    if effective not in STRATEGIES:
        raise ValueError(f"unknown effective strategy: {effective}")
    if requested == "dep4" and effective != "dep4":
        raise ValueError("DEP4 requires a real DEP4 executor")
    if requested != "dep4" and effective != requested:
        raise ValueError("effective strategy differs from the requested strategy")


def layer_table(model=DEEPSEEK_V41_MODEL):
    """Return the per-layer operation table."""
    table = []
    for layer_id, ratio in enumerate(model.compress_ratios[: model.num_layers]):
        is_index_source = layer_id in model.index_source_layers
        table.append(
            LayerSpec(
                layer_id=layer_id,
                compress_ratio=ratio,
                owns_compressed_kv=layer_id in model.kv_source_layers,
                owns_index_keys=is_index_source,
                uses_candidate_mask=(
                    is_index_source and layer_id > model.candidate_source_layer
                ),
                has_engram=layer_id in model.engram_layers,
            )
        )
    return tuple(table)


def _local_shape(out_features, in_features, shard_axis, shards):
    if type(shards) is not int or shards < 1:
        raise ValueError("shards must be a positive integer")
    if shards == 1:
        if shard_axis is not None:
            raise ValueError("shard axis requires more than one shard")
        return out_features, in_features
    if shard_axis not in (0, 1):
        raise ValueError("shard axis must be zero or one")
    axis_dimension = out_features if shard_axis == 0 else in_features
    if axis_dimension % shards:
        raise ValueError("shards must divide the sharded dimension")
    return (
        out_features // shards if shard_axis == 0 else out_features,
        in_features // shards if shard_axis == 1 else in_features,
    )


def _weight_bytes(storage, out_features, in_features):
    if storage == "fp8":
        blocks = ((out_features + 31) // 32) * ((in_features + 31) // 32)
        return out_features * in_features + blocks
    if storage == "fp4":
        if in_features % 32:
            raise ValueError("FP4 storage requires K aligned to 32")
        return out_features * in_features // 2 + out_features * (in_features // 32)
    if storage == "bf16":
        return 2 * out_features * in_features
    if storage == "fp32":
        return 4 * out_features * in_features
    raise ValueError(f"unknown storage format: {storage}")


def _activation_element_bytes(compute):
    if compute in ("fp8", "fp4"):
        return 1, 2
    if compute == "bf16":
        return 2, 2
    if compute == "fp32":
        return 4, 4
    raise ValueError(f"unknown compute format: {compute}")


def _projection_parts(
    rows,
    out_features,
    in_features,
    *,
    storage,
    compute=None,
    shard_axis=None,
    shards=1,
    include_weights=True,
):
    compute = compute or ("fp8" if storage == "fp4" else storage)
    rows = _checked_positive_int(rows, "rows")
    out_features = _checked_positive_int(out_features, "out_features")
    in_features = _checked_positive_int(in_features, "in_features")
    local_out, local_in = _local_shape(out_features, in_features, shard_axis, shards)
    weight = _weight_bytes(storage, local_out, local_in) if include_weights else 0
    input_bytes, output_bytes = _activation_element_bytes(compute)
    activation = rows * local_in * input_bytes + rows * local_out * output_bytes
    flops = 2 * rows * local_out * local_in
    return weight, activation, flops


def projection_work(
    rows,
    out_features,
    in_features,
    *,
    storage,
    compute=None,
    shard_axis=None,
    shards=1,
    include_weights=True,
):
    """Return weight, activation, and useful MMA work for one projection pass.

    FP4 storage uses an FP8 MMA, because the model contract changes FP4
    weights to FP8 before the dot product. Use `compute` to price a BF16
    weight that the model promotes to FP32.
    """
    compute = compute or ("fp8" if storage == "fp4" else storage)
    weight, activation, flops = _projection_parts(
        rows,
        out_features,
        in_features,
        storage=storage,
        compute=compute,
        shard_axis=shard_axis,
        shards=shards,
        include_weights=include_weights,
    )
    if compute not in ("fp8", "bf16", "fp32"):
        raise ValueError(f"unknown compute format: {compute}")
    return Work(
        hbm_bytes=weight + activation,
        **{f"{compute}_flops": flops},
    )


def _prefix_window(length, window):
    if length <= 0:
        return 0
    if length <= window:
        return length * (length + 1) // 2
    return window * (window + 1) // 2 + (length - window) * window


def window_pairs(start, rows, window=128):
    """Return sum of min(window, p+1) for p in [start, start+rows)."""
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    window = _checked_positive_int(window, "window")
    return _prefix_window(start + rows, window) - _prefix_window(start, window)


def _prefix_floor(ratio, limit):
    if limit <= 0:
        return 0
    quotient, remainder = divmod(limit, ratio)
    return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _prefix_capped(ratio, cap, limit):
    if limit <= 0:
        return 0
    cutoff = cap * ratio
    if limit <= cutoff:
        return _prefix_floor(ratio, limit)
    return _prefix_floor(ratio, cutoff) + (limit - cutoff) * cap


def compressed_pairs(start, rows, ratio, topk=512):
    """Return attention top-k pairs. This count is not the indexer count."""
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    ratio = _checked_positive_int(ratio, "ratio")
    topk = _checked_positive_int(topk, "topk")
    return _prefix_capped(ratio, topk, start + rows) - _prefix_capped(
        ratio, topk, start
    )


def index_score_pairs(start, rows, ratio, *, candidate_limit=None):
    """Return compressed positions an index layer scores.

    Without a candidate limit the layer scores all reachable positions.
    With a limit, the layer scores at most that many candidates per query.
    """
    start = _checked_nonnegative_int(start, "start")
    rows = _checked_positive_int(rows, "rows")
    ratio = _checked_positive_int(ratio, "ratio")
    if candidate_limit is None:
        return _prefix_floor(ratio, start + rows) - _prefix_floor(ratio, start)
    candidate_limit = _checked_positive_int(candidate_limit, "candidate_limit")
    return _prefix_capped(ratio, candidate_limit, start + rows) - _prefix_capped(
        ratio, candidate_limit, start
    )


def collective_latency_seconds(kind, dtype, profile_input_bytes):
    """Return an idle, unlocked latency from the measured profile."""
    profile_input_bytes = _checked_nonnegative_int(
        profile_input_bytes, "collective profile input"
    )
    points_us = COLLECTIVE_LATENCY_PROFILE_US.get((kind, dtype))
    if points_us is None:
        raise ValueError(f"no measured latency profile for {kind}/{dtype}")
    if profile_input_bytes == 0:
        return 0.0
    points = tuple((size, latency_us * 1e-6) for size, latency_us in points_us)
    monotonic_points = []
    highest_latency = 0.0
    for size, latency in points:
        highest_latency = max(highest_latency, latency)
        monotonic_points.append((size, highest_latency))
    if profile_input_bytes <= monotonic_points[0][0]:
        return monotonic_points[0][1]
    if profile_input_bytes >= monotonic_points[-1][0]:
        return monotonic_points[-1][1]
    for left, right in pairwise(monotonic_points):
        if left[0] <= profile_input_bytes <= right[0]:
            fraction = (profile_input_bytes - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    raise ValueError(f"no latency interpolation interval for {kind}/{dtype}")


def all_reduce_wire_bytes(world_size, bytes_per_rank):
    """Ring all-reduce bytes sent per rank for N input bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    bytes_per_rank = _checked_nonnegative_int(bytes_per_rank, "bytes_per_rank")
    return 2 * (world_size - 1) * bytes_per_rank / world_size


def all_gather_wire_bytes(world_size, output_bytes_per_rank):
    """Ring all-gather bytes sent per rank for N output bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    output_bytes_per_rank = _checked_nonnegative_int(
        output_bytes_per_rank, "output_bytes_per_rank"
    )
    return (world_size - 1) * output_bytes_per_rank / world_size


def reduce_scatter_wire_bytes(world_size, output_bytes_per_rank):
    """Ring reduce-scatter bytes sent per rank for M output bytes per rank."""
    world_size = _checked_positive_int(world_size, "world_size")
    output_bytes_per_rank = _checked_nonnegative_int(
        output_bytes_per_rank, "output_bytes_per_rank"
    )
    return (world_size - 1) * output_bytes_per_rank


def route_wire_pressure_bytes(route_rows, rank, element_bytes):
    """Return max(sent, received) wire bytes for one rank.

    Self routes use no wire. Each NVLink direction has its own 900 GB/s,
    so a sender and a receiver load different directions. The larger count
    sets the time for this rank.
    """
    if not route_rows:
        raise ValueError("route matrix must not be empty")
    world_size = len(route_rows)
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise ValueError("rank is outside the route matrix")
    element_bytes = _checked_positive_int(element_bytes, "element_bytes")
    for row in route_rows:
        if len(row) != world_size:
            raise ValueError("route matrix must be square")
        for count in row:
            _checked_nonnegative_int(count, "route count")
    sent = sum(
        route_rows[rank][destination]
        for destination in range(world_size)
        if destination != rank
    )
    received = sum(
        route_rows[source][rank] for source in range(world_size) if source != rank
    )
    return max(sent, received) * element_bytes


def _tensor_shards(strategy):
    return WORLD_SIZE if strategy in ("tp4", "tep4") else 1


def _expert_shards(strategy):
    return WORLD_SIZE if strategy == "tp4" else 1


def attention_input_work(rows, model, strategy, *, include_weights=True):
    """Return wq_a, wq_b, and wkv work for the credited rows."""
    shards = _tensor_shards(strategy)
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.q_lora_rank,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.num_heads * model.head_dim,
                model.q_lora_rank,
                storage="fp8",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.head_dim,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def attention_output_work(rows, model, strategy, *, include_weights=True):
    """Return wo_a and wo_b work for the credited rows."""
    shards = _tensor_shards(strategy)
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.o_groups * model.o_lora_rank,
                model.num_heads * model.head_dim // model.o_groups,
                storage="bf16",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.hidden_size,
                model.o_groups * model.o_lora_rank,
                storage="fp8",
                shard_axis=1 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
        )
    return work


def routed_expert_work(
    assignments,
    touched_experts,
    model,
    strategy,
    *,
    include_weights=True,
):
    """Return routed FP4 expert work for the local assignments.

    TP4 keeps every assignment on each rank and shards output channels.
    TEP4 and DEP4 keep the full expert on its owner rank.
    """
    assignments = _checked_nonnegative_int(assignments, "assignments")
    touched_experts = _checked_nonnegative_int(touched_experts, "touched_experts")
    shards = _expert_shards(strategy)
    rows = max(assignments, 1)
    up = _projection_parts(
        rows,
        model.moe_intermediate,
        model.hidden_size,
        storage="fp4",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
    )
    down = _projection_parts(
        rows,
        model.hidden_size,
        model.moe_intermediate,
        storage="fp4",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
    )
    weight = touched_experts * (2 * up[0] + down[0]) if include_weights else 0
    activation = 0 if not assignments else 2 * up[1] + down[1]
    flops = 0 if not assignments else 2 * up[2] + down[2]
    return Work(hbm_bytes=weight + activation, fp8_flops=flops)


def shared_expert_work(rows, model, strategy, *, include_weights=True):
    """Return the replicated shared expert work."""
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.moe_intermediate,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.moe_intermediate,
                model.hidden_size,
                storage="fp8",
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.hidden_size,
                model.moe_intermediate,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def router_work(rows, model, *, include_weights=True):
    """Return the FP32 router projection work."""
    work = Work()
    for _layer in layer_table(model):
        work = add_work(
            work,
            projection_work(
                rows,
                model.routed_experts,
                model.hidden_size,
                storage="bf16",
                compute="fp32",
                include_weights=include_weights,
            ),
        )
    return work


def hyper_connection_work(rows, model, *, include_weights=True):
    """Return Hyper-Connection mixes in FP32."""
    hc_width = model.hc_mult * model.hidden_size
    flops = (
        rows
        * model.num_layers
        * (2 * 2 * model.hc_mix * hc_width + model.hc_sinkhorn_iters * model.hc_mult**3)
    )
    weights = 0
    if include_weights:
        weights = model.num_layers * 2 * _weight_bytes("fp32", model.hc_mix, hc_width)
    activations = (
        rows
        * model.num_layers
        * (2 * hc_width * 4 + 2 * model.hc_mix * 4 + 2 * model.hc_mult**2 * 4)
    )
    return Work(hbm_bytes=weights + activations, fp32_flops=flops)


def compressor_work(rows, model, *, include_weights=True):
    """Return compressor projections for the layers that own compressed KV."""
    work = Work()
    for layer in layer_table(model):
        if not layer.owns_compressed_kv:
            continue
        if layer.compress_ratio > 1:
            work = add_work(
                work,
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="fp32",
                    compute="fp32",
                    include_weights=include_weights,
                ),
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="fp32",
                    compute="fp32",
                    include_weights=include_weights,
                ),
            )
        else:
            work = add_work(
                work,
                projection_work(
                    rows,
                    model.head_dim,
                    model.hidden_size,
                    storage="bf16",
                    compute="bf16",
                    include_weights=include_weights,
                ),
            )
    return work


def engram_projection_work(rows, model, *, include_weights=True):
    """Return the Engram value projection for each Engram layer."""
    work = Work()
    for _layer in layer_table(model):
        if not _layer.has_engram:
            continue
        work = add_work(
            work,
            projection_work(
                rows,
                model.hidden_size * (model.hc_mult + 1),
                model.engram_columns * model.engram_head_dim,
                storage="fp8",
                include_weights=include_weights,
            ),
        )
    return work


def engram_gate_work(rows, model):
    """Return the FP32 Engram gate on the residual stream."""
    flops = rows * len(model.engram_layers) * 6 * model.hc_mult * model.hidden_size
    return Work(fp32_flops=flops)


def output_head_work(logit_rows, model, strategy, *, include_weights=True):
    """Return the FP32 output head for the credited logit rows."""
    shards = _tensor_shards(strategy)
    return projection_work(
        logit_rows,
        model.vocab_size,
        model.hidden_size,
        storage="fp32",
        compute="fp32",
        shard_axis=0 if shards > 1 else None,
        shards=shards,
        include_weights=include_weights,
    )


def embedding_work(rows, model, strategy):
    """Return the sharded embedding lookup work.

    The selected-row read is per call. The embedding table is not a dense
    weight read, so a later prefill chunk keeps its own row reads.
    """
    shards = _tensor_shards(strategy)
    selected = min(rows, model.vocab_size)
    local_rows = (selected + shards - 1) // shards
    weight = local_rows * model.hidden_size * 2
    activation = rows * model.hidden_size * 2
    return Work(hbm_bytes=weight + activation)


def scale_accumulation_work(work):
    """One FP32 FMA per output and K32 group is the minimum scale accumulation."""
    if not work.fp8_flops:
        return work
    return replace(work, fp32_flops=work.fp32_flops + work.fp8_flops / 32)


@lru_cache(maxsize=128)
def layer_model(model, layer):
    return replace(
        model,
        num_layers=1,
        compress_ratios=(layer.compress_ratio,),
        kv_source_layers=(0,) if layer.owns_compressed_kv else (),
        index_source_layers=(0,) if layer.owns_index_keys else (),
        engram_layers=(0,) if layer.has_engram else (),
    )


@lru_cache(maxsize=512)
def index_projection_work(
    rows, new_keys, model, strategy, include_weights, key_weights=None
):
    shards = 1 if strategy == "dep4" else WORLD_SIZE
    common = dict(
        shard_axis=0 if shards > 1 else None,
        shards=shards,
        include_weights=include_weights,
    )
    work = add_work(
        projection_work(
            rows,
            model.index_heads * model.index_head_dim,
            model.q_lora_rank,
            storage="fp8",
            **common,
        ),
        projection_work(
            rows, model.index_heads, model.hidden_size, storage="bf16", **common
        ),
    )
    if new_keys:
        work = add_work(
            work,
            projection_work(
                new_keys,
                model.index_head_dim,
                model.head_dim,
                storage="bf16",
                include_weights=include_weights if key_weights is None else key_weights,
            ),
        )
    return work


def dep4_all_to_all_seconds(wire_bytes, source_dtype):
    if source_dtype not in ("bf16", "fp32"):
        raise ValueError("DEP4 proxy supports BF16 and FP32 route values")
    profile_input_bytes = math.ceil(wire_bytes * WORLD_SIZE / (WORLD_SIZE - 1))
    latency = collective_latency_seconds("all_to_all", "bf16", profile_input_bytes)
    return max(latency, wire_bytes / RATES["nvlink_bytes"])


def collective_work(kind, dtype, profile_input_bytes, output_bytes_per_rank):
    if kind == "all_gather":
        wire = all_gather_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    elif kind == "all_reduce":
        wire = all_reduce_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    else:
        raise ValueError(f"unknown collective kind: {kind}")
    latency = collective_latency_seconds(kind, dtype, profile_input_bytes)
    return Work(
        nvlink_bytes=wire,
        collective_seconds=max(latency, wire / RATES["nvlink_bytes"]),
    )


@lru_cache(maxsize=512)
def local_work(rows, model, strategy, include_weights):
    return (
        (
            "attention_input",
            attention_input_work(
                rows, model, strategy, include_weights=include_weights
            ),
        ),
        (
            "attention_output",
            attention_output_work(
                rows, model, strategy, include_weights=include_weights
            ),
        ),
        (
            "shared_expert",
            shared_expert_work(rows, model, strategy, include_weights=include_weights),
        ),
        ("router", router_work(rows, model, include_weights=include_weights)),
        (
            "hyper_connection",
            hyper_connection_work(rows, model, include_weights=include_weights),
        ),
        ("compressor", compressor_work(rows, model, include_weights=include_weights)),
        (
            "engram_projection",
            engram_projection_work(rows, model, include_weights=include_weights),
        ),
        ("engram_gate", engram_gate_work(rows, model)),
    )


def engram_lookup_work(rows, layer, model, *, shards=1):
    # Disjoint hash columns prove this floor without token/hash identities.
    unique = model.engram_columns if rows and layer.has_engram else 0
    packed = model.engram_head_dim + model.engram_scale_bytes
    return Work(
        host_bytes=unique * packed / shards,
        hbm_bytes=unique * (packed + 2 * model.engram_head_dim) / shards,
    )


def candidate_pairs(length, ratio, limit):
    """Count valid keys when the latest eight-position block must be selected."""
    block = 8
    if limit < block or limit % block:
        raise ValueError("candidate limit must contain complete blocks")
    groups, remainder = divmod(length, ratio)
    if not groups:
        return 0

    def prefix(count):
        if count <= limit:
            return count * (count + 1) // 2
        tail = count - limit
        cycles, partial = divmod(tail, block)
        return (
            limit * (limit + 1) // 2
            + tail * (limit - block)
            + cycles * block * (block + 1) // 2
            + partial * (partial + 1) // 2
        )

    last = min(groups, limit - block + (groups - 1) % block + 1)
    return ratio * prefix(groups - 1) + (remainder + 1) * last
