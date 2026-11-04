"""DeepSeek-V4.1-Flash HTTP reference calculation `deepseek-http-reference-v2`.

The reference is a lower bound on the time to make the same accepted output.
It prices exact resource work from a frozen execution trace. It adds the
serial groups after the maximum of four ranks for each group. The work does
not include queueing, HTTP, scheduling, padding, rejected drafts, rejected
verification rows, per-chunk weight reads, or temporary copies.

The module also makes the ideal-acceptance plan of an approved HTTP case.
That plan is not a measurement. Use a trace for an efficiency result.
"""

import hashlib
import math
from dataclasses import dataclass, replace
from itertools import pairwise

VERSION = "deepseek-http-reference-v2"
LEGACY_VERSION = "deepseek-http-reference-v1"
COLLECTIVE_PROFILE_VERSION = "idle-unlocked-nccl-profile-v1"
DEP4_PROXY_RULE_VERSION = "dep4-bf16-all-to-all-proxy-v1"
WIRE_MODEL_VERSION = "legacy-wire-v1"
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

# Rate profile. Each value is per rank. A change to one value makes a new version.
RATE_PROFILE = (
    ("hbm_bytes", 7.7e12, "HBM3e, 7.7 TB/s per GPU"),
    ("fp8_flops", 4.5e15, "B200 dense FP8 tensor"),
    ("bf16_flops", 2.25e15, "B200 dense BF16 tensor"),
    ("fp32_flops", 75e12, "B200 FP32 vector"),
    ("nvlink_bytes", 0.9e12, "NVLink 5, 900 GB/s per direction"),
    ("host_bytes", 63.02e9, "PCIe 5.0 x16, 63.02 GB/s per direction"),
)
RATES = {name: value for name, value, _source in RATE_PROFILE}

# Rank medians in microseconds. Generated from the hashed raw profile.
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


def collective_profile_digest():
    content = repr(COLLECTIVE_LATENCY_PROFILE_US).encode()
    raw_digest = bytes.fromhex(COLLECTIVE_PROFILE_RAW_SHA256)
    return hashlib.sha256(raw_digest + content).hexdigest()


def communication_model_metadata(strategy, communication_model):
    version = communication_version(communication_model)
    if strategy == "dep4" and communication_model == "lower_bound":
        rule_version = DEP4_PROXY_RULE_VERSION
    elif communication_model == "wire":
        rule_version = WIRE_MODEL_VERSION
    else:
        rule_version = COLLECTIVE_PROFILE_VERSION
    return {
        "accounting_version": version,
        "communication_model": communication_model,
        "profile_digest": (
            collective_profile_digest()
            if communication_model == "lower_bound"
            else None
        ),
        "communication_rule_version": rule_version,
        "dep4_communication_modeled": strategy == "dep4",
        "dep4_communication_measured": False,
    }


class BindingError(ValueError):
    """The accounting identity is absent or different."""


class TraceError(ValueError):
    """A call record does not satisfy the trace contract."""


@dataclass(frozen=True, slots=True)
class Work:
    """Immutable resource vector. All values are finite and nonnegative."""

    hbm_bytes: float = 0.0
    fp8_flops: float = 0.0
    bf16_flops: float = 0.0
    fp32_flops: float = 0.0
    nvlink_bytes: float = 0.0
    host_bytes: float = 0.0
    collective_seconds: float = 0.0

    def __post_init__(self):
        for name in WORK_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) not in (int, float):
                raise TypeError(f"{name} must be an int or float")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Group:
    """One serial kernel group."""

    name: str
    work: Work
    repetitions: int = 1

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise ValueError("group name must be a nonempty string")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("group repetitions must be a positive integer")
        if not isinstance(self.work, Work):
            raise TypeError("group work must be a Work value")


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

# Engram host-to-device row: FP8 values plus one E8M0 scale per 32 values.
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
    if any(not isinstance(item, Work) for item in items):
        raise TypeError("add_work requires Work values")
    return Work(
        **{
            name: math.fsum(getattr(item, name) for item in items)
            for name in WORK_FIELDS
        }
    )


def scale_work(work, count):
    """Scale one logical work vector by a positive serial count."""
    if not isinstance(work, Work):
        raise TypeError("scale_work requires a Work value")
    count = _checked_positive_int(count, "work scale")
    return Work(**{name: getattr(work, name) * count for name in WORK_FIELDS})


def _checked_rate(name):
    rate = RATES.get(name)
    if rate is None or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"resource has no reviewed positive rate: {name}")
    return rate


def group_seconds(group):
    """Return one group's maximum resource time, multiplied by repetitions."""
    if not isinstance(group, Group):
        raise TypeError("group_seconds requires a Group value")
    work = group.work
    tensor_seconds = math.fsum(
        getattr(work, name) / _checked_rate(name)
        for name in ("fp8_flops", "bf16_flops")
    )
    seconds = max(
        work.hbm_bytes / _checked_rate("hbm_bytes"),
        tensor_seconds,
        work.fp32_flops / _checked_rate("fp32_flops"),
        work.nvlink_bytes / _checked_rate("nvlink_bytes"),
        work.host_bytes / _checked_rate("host_bytes"),
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


def rate_profile_digest(version=VERSION):
    """Return the identity of a rate and latency profile."""
    text = "\n".join(f"{name}={value!r}" for name, value, _source in RATE_PROFILE)
    if version == LEGACY_VERSION:
        return hashlib.sha256(text.encode()).hexdigest()
    if version != VERSION:
        raise ValueError(f"unknown accounting version: {version}")
    text += f"\ncollective_profile={collective_profile_digest()}"
    text += f"\ncollective_profile_version={COLLECTIVE_PROFILE_VERSION}"
    text += f"\ndep4_proxy_rule={DEP4_PROXY_RULE_VERSION}"
    return hashlib.sha256(text.encode()).hexdigest()


def communication_version(communication_model):
    if communication_model == "wire":
        return LEGACY_VERSION
    if communication_model == "lower_bound":
        return VERSION
    raise ValueError("communication_model must be 'wire' or 'lower_bound'")


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


def sparse_pairs(start, rows, ratio, *, window=128, topk=512):
    """Return exact selected attention slots, without padding."""
    return window_pairs(start, rows, window) + compressed_pairs(
        start, rows, ratio, topk
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


def _expected_distinct(bins, draws):
    if draws <= 0:
        return 0
    return min(bins, round(bins * (1 - (1 - 1 / bins) ** draws)))


def _route_row(assignments):
    base, remainder = divmod(assignments, WORLD_SIZE)
    return tuple(base + (destination < remainder) for destination in range(WORLD_SIZE))


def _route_matrix(plans, model):
    """Return the per-layer 4 by 4 dispatch matrix for all ranks."""
    return tuple(_route_row(plan.rows * model.experts_per_token) for plan in plans)


def _partition_rows(total, rank):
    base, remainder = divmod(total, WORLD_SIZE)
    return base + int(rank < remainder)


def _layer_model(model):
    return layer_table(model)


def _dense_shape(model, strategy):
    return _tensor_shards(strategy)


def attention_input_work(rows, model, strategy, *, include_weights=True):
    """Return wq_a, wq_b, and wkv work for the credited rows."""
    shards = _dense_shape(model, strategy)
    work = Work()
    for _layer in _layer_model(model):
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
    shards = _dense_shape(model, strategy)
    work = Work()
    for _layer in _layer_model(model):
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
    for _layer in _layer_model(model):
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
    for _layer in _layer_model(model):
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
    for layer in _layer_model(model):
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
    for _layer in _layer_model(model):
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


def indexer_work(rows, pairs, model, strategy, *, include_weights=True):
    """Return indexer projections and BF16 score work for one call."""
    shards = _dense_shape(model, strategy)
    work = Work(bf16_flops=2 * pairs * model.index_heads * model.index_head_dim)
    for layer in _layer_model(model):
        if not layer.owns_index_keys:
            continue
        work = add_work(
            work,
            projection_work(
                rows,
                model.index_heads * model.index_head_dim,
                model.q_lora_rank,
                storage="fp8",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
            projection_work(
                rows,
                model.index_heads,
                model.hidden_size,
                storage="bf16",
                shard_axis=0 if shards > 1 else None,
                shards=shards,
                include_weights=include_weights,
            ),
        )
        if layer.owns_compressed_kv:
            compressed_rows = rows // layer.compress_ratio
            if compressed_rows:
                work = add_work(
                    work,
                    projection_work(
                        compressed_rows,
                        model.index_head_dim,
                        model.head_dim,
                        storage="bf16",
                        include_weights=include_weights,
                    ),
                )
    return work


def sparse_attention_work(pairs, model):
    """Return BF16 sparse attention QK and PV work."""
    return Work(bf16_flops=4 * pairs * model.num_heads * model.head_dim)


def output_head_work(logit_rows, model, strategy, *, include_weights=True):
    """Return the FP32 output head for the credited logit rows."""
    shards = _dense_shape(model, strategy)
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
    shards = _dense_shape(model, strategy)
    selected = min(rows, model.vocab_size)
    local_rows = (selected + shards - 1) // shards
    weight = local_rows * model.hidden_size * 2
    activation = rows * model.hidden_size * 2
    return Work(hbm_bytes=weight + activation)


def _routed_assignments_per_layer(plan, model, strategy):
    if strategy == "tp4":
        return plan.global_rows * model.experts_per_token
    if strategy == "tep4":
        return (plan.global_rows * model.experts_per_token) // WORLD_SIZE
    return plan.rows * model.experts_per_token


def _routed_assignments(plan, model, strategy):
    return _routed_assignments_per_layer(plan, model, strategy) * model.num_layers


def _touched_experts(plan, model, strategy):
    """Return local expert weights read. Each layer owns different tensors."""
    assignments = _routed_assignments_per_layer(plan, model, strategy)
    bins = (
        model.routed_experts
        if strategy == "tp4"
        else model.routed_experts // WORLD_SIZE
    )
    return model.num_layers * _expected_distinct(bins, assignments)


def _index_score_layer_width(plan, layer):
    end_pos = plan.context_end + 1
    return end_pos // layer.compress_ratio if layer.compress_ratio > 1 else end_pos


def _index_score_collectives(plan, model):
    collectives = []
    for layer in _layer_model(model):
        if not layer.owns_index_keys:
            continue
        width = _index_score_layer_width(plan, layer)
        if width and plan.rows:
            collectives.append((plan.rows * width * 4, 1))
    return tuple(collectives)


def _dep4_all_to_all_seconds(wire_bytes, source_dtype):
    if source_dtype not in ("bf16", "fp32"):
        raise ValueError("DEP4 proxy supports BF16 and FP32 route values")
    profile_input_bytes = math.ceil(wire_bytes * WORLD_SIZE / (WORLD_SIZE - 1))
    latency = collective_latency_seconds("all_to_all", "bf16", profile_input_bytes)
    return max(latency, wire_bytes / RATES["nvlink_bytes"])


def _collective_work(
    kind,
    dtype,
    profile_input_bytes,
    output_bytes_per_rank,
    count,
    communication_model,
):
    if communication_model not in ("lower_bound", "wire"):
        raise ValueError("unknown communication model")
    if type(count) is not int or count < 1:
        raise ValueError("collective count must be positive")
    profile_input_bytes = _checked_nonnegative_int(
        profile_input_bytes, "collective profile input"
    )
    output_bytes_per_rank = _checked_nonnegative_int(
        output_bytes_per_rank, "collective output bytes"
    )
    if kind == "all_gather":
        wire = all_gather_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    elif kind == "all_reduce":
        if profile_input_bytes != output_bytes_per_rank:
            raise ValueError("all-reduce profile and output bytes must match")
        wire = all_reduce_wire_bytes(WORLD_SIZE, output_bytes_per_rank)
    elif kind == "all_to_all":
        wire = output_bytes_per_rank
    else:
        raise ValueError(f"unknown collective kind: {kind}")
    latency = 0.0
    if communication_model == "lower_bound":
        measured_latency = collective_latency_seconds(kind, dtype, profile_input_bytes)
        wire_seconds = wire / RATES["nvlink_bytes"]
        latency = count * max(measured_latency, wire_seconds)
    return Work(nvlink_bytes=count * wire, collective_seconds=latency)


def _wire_communication_groups(plan, model, strategy):
    """Return the communication groups for one rank and one call."""
    groups = []
    if strategy in ("tp4", "tep4"):
        rows = plan.global_rows
        dense = Work()
        dense = add_work(
            dense,
            Work(
                nvlink_bytes=all_reduce_wire_bytes(
                    WORLD_SIZE, rows * model.hidden_size * 4
                )
                * model.num_layers
            ),
        )
        dense = add_work(
            dense,
            Work(
                nvlink_bytes=all_reduce_wire_bytes(
                    WORLD_SIZE, rows * model.hidden_size * 2
                )
            ),
        )
        dense = add_work(
            dense,
            Work(
                nvlink_bytes=all_gather_wire_bytes(
                    WORLD_SIZE,
                    plan.logit_rows * model.vocab_size * 4,
                )
            ),
        )
        groups.append(Group("communication.dense", dense))
        if strategy == "tp4":
            assignments = _routed_assignments(plan, model, strategy)
            routed = Work(
                nvlink_bytes=assignments
                * all_gather_wire_bytes(WORLD_SIZE, model.moe_intermediate * 2)
                + rows
                * model.num_layers
                * all_gather_wire_bytes(WORLD_SIZE, model.hidden_size * 4)
            )
        else:
            routed = Work(
                nvlink_bytes=rows
                * model.num_layers
                * all_reduce_wire_bytes(WORLD_SIZE, model.hidden_size * 4)
            )
        groups.append(Group("communication.routed", routed))
        groups.append(
            Group(
                "communication.engram",
                Work(
                    nvlink_bytes=all_reduce_wire_bytes(
                        WORLD_SIZE,
                        plan.rows
                        * len(model.engram_layers)
                        * model.engram_columns
                        * model.engram_head_dim
                        * 2,
                    )
                ),
            )
        )
    else:
        groups.append(
            Group(
                "communication.dispatch",
                Work(
                    nvlink_bytes=route_wire_pressure_bytes(
                        plan.route_matrix, plan.rank, model.hidden_size * 2
                    )
                    * model.num_layers
                ),
            )
        )
        groups.append(
            Group(
                "communication.combine",
                Work(
                    nvlink_bytes=route_wire_pressure_bytes(
                        plan.route_matrix, plan.rank, model.hidden_size * 4
                    )
                    * model.num_layers
                ),
            )
        )
    return groups


def communication_groups(plan, model, strategy, communication_model="lower_bound"):
    """Return serial communication groups for the selected lower-bound model."""
    communication_version(communication_model)
    if communication_model == "wire":
        return _wire_communication_groups(plan, model, strategy)
    groups = []
    if strategy in ("tp4", "tep4"):
        rows = plan.global_rows
        dense = add_work(
            _collective_work(
                "all_reduce",
                "fp32",
                rows * model.hidden_size * 4,
                rows * model.hidden_size * 4,
                model.num_layers,
                communication_model,
            ),
            _collective_work(
                "all_reduce",
                "bf16",
                rows * model.hidden_size * 2,
                rows * model.hidden_size * 2,
                1,
                communication_model,
            ),
            _collective_work(
                "all_gather",
                "fp32",
                plan.logit_rows * model.vocab_size,
                plan.logit_rows * model.vocab_size * WORLD_SIZE,
                1,
                communication_model,
            ),
        )
        groups.append(Group("communication.dense", dense))
        if strategy == "tp4":
            routed = add_work(
                _collective_work(
                    "all_gather",
                    "bf16",
                    rows
                    * model.experts_per_token
                    * (model.moe_intermediate // WORLD_SIZE)
                    * 2,
                    rows * model.experts_per_token * model.moe_intermediate * 2,
                    model.num_layers,
                    communication_model,
                ),
                _collective_work(
                    "all_gather",
                    "fp32",
                    rows * model.hidden_size,
                    rows * model.hidden_size * WORLD_SIZE,
                    model.num_layers,
                    communication_model,
                ),
            )
        else:
            routed = _collective_work(
                "all_reduce",
                "fp32",
                rows * model.hidden_size * 4,
                rows * model.hidden_size * 4,
                model.num_layers,
                communication_model,
            )
        for score_input_bytes, count in _index_score_collectives(plan, model):
            routed = add_work(
                routed,
                _collective_work(
                    "all_reduce",
                    "fp32",
                    score_input_bytes,
                    score_input_bytes,
                    count,
                    communication_model,
                ),
            )
        groups.append(Group("communication.routed", routed))
        engram = Work()
        if plan.rows and model.engram_layers:
            payload = plan.rows * model.engram_columns * model.engram_head_dim * 2
            engram = _collective_work(
                "all_reduce",
                "bf16",
                payload,
                payload,
                len(model.engram_layers),
                communication_model,
            )
        groups.append(Group("communication.engram", engram))
    else:
        dispatch = route_wire_pressure_bytes(
            plan.route_matrix, plan.rank, model.hidden_size * 2
        )
        combine = route_wire_pressure_bytes(
            plan.route_matrix, plan.rank, model.hidden_size * 4
        )
        modeled_dispatch = communication_model == "lower_bound" and dispatch > 0
        groups.append(
            Group(
                "communication.dispatch",
                Work(
                    nvlink_bytes=dispatch * model.num_layers,
                    collective_seconds=(
                        model.num_layers * _dep4_all_to_all_seconds(dispatch, "bf16")
                        if modeled_dispatch
                        else 0.0
                    ),
                ),
            )
        )
        modeled_combine = communication_model == "lower_bound" and combine > 0
        groups.append(
            Group(
                "communication.combine",
                Work(
                    nvlink_bytes=combine * model.num_layers,
                    collective_seconds=(
                        model.num_layers * _dep4_all_to_all_seconds(combine, "fp32")
                        if modeled_combine
                        else 0.0
                    ),
                ),
            )
        )
    return groups


@dataclass(frozen=True, slots=True)
class CallPlan:
    """One rank's credited work in one target call."""

    rank: int
    phase: str
    requests: int
    rows: int
    global_rows: int
    logit_rows: int
    context_end: int
    accepted: int
    proposed: int
    sparse_pairs: int
    index_pairs: int
    engram_unique_rows: int
    touched_experts: int
    route_matrix: tuple[tuple[int, ...], ...]


def _group_names(strategy):
    names = [
        "embedding",
        "routed_experts",
        "shared_expert",
        "attention_input",
        "attention_output",
        "sparse_attention",
        "indexer",
        "router",
        "hyper_connection",
        "compressor",
        "engram_projection",
        "engram_gate",
        "output_head",
        "engram_h2d",
    ]
    if strategy in ("tp4", "tep4"):
        names += ["communication.dense", "communication.routed", "communication.engram"]
    else:
        names += ["communication.dispatch", "communication.combine"]
    return tuple(names)


def call_groups(
    plan,
    model=DEEPSEEK_V41_MODEL,
    strategy="tp4",
    *,
    include_weights=True,
    communication_model="lower_bound",
):
    """Return the serial groups for one rank and one call."""
    if plan.phase not in ("prefill", "decode"):
        raise ValueError("call phase must be prefill or decode")
    if plan.rows == 0 and plan.logit_rows == 0:
        return tuple(Group(name, Work()) for name in _group_names(strategy))
    groups = [
        Group(
            "embedding",
            embedding_work(plan.rows, model, strategy),
        )
    ]
    groups.extend(
        [
            Group(
                "routed_experts",
                routed_expert_work(
                    _routed_assignments(plan, model, strategy),
                    plan.touched_experts,
                    model,
                    strategy,
                    include_weights=include_weights,
                ),
            ),
            Group(
                "shared_expert",
                shared_expert_work(
                    plan.rows, model, strategy, include_weights=include_weights
                ),
            ),
            Group(
                "attention_input",
                attention_input_work(
                    plan.rows, model, strategy, include_weights=include_weights
                ),
            ),
            Group(
                "attention_output",
                attention_output_work(
                    plan.rows, model, strategy, include_weights=include_weights
                ),
            ),
            Group("sparse_attention", sparse_attention_work(plan.sparse_pairs, model)),
            Group(
                "indexer",
                indexer_work(
                    plan.rows,
                    plan.index_pairs,
                    model,
                    strategy,
                    include_weights=include_weights,
                ),
            ),
            Group(
                "router",
                router_work(plan.rows, model, include_weights=include_weights),
            ),
            Group(
                "hyper_connection",
                hyper_connection_work(
                    plan.rows, model, include_weights=include_weights
                ),
            ),
            Group(
                "compressor",
                compressor_work(plan.rows, model, include_weights=include_weights),
            ),
            Group(
                "engram_projection",
                engram_projection_work(
                    plan.rows, model, include_weights=include_weights
                ),
            ),
            Group("engram_gate", engram_gate_work(plan.rows, model)),
            Group(
                "output_head",
                output_head_work(
                    max(plan.logit_rows, 1),
                    model,
                    strategy,
                    include_weights=include_weights,
                ),
            ),
            Group(
                "engram_h2d",
                Work(
                    host_bytes=plan.engram_unique_rows
                    * (model.engram_head_dim + model.engram_scale_bytes)
                ),
            ),
            *communication_groups(plan, model, strategy, communication_model),
        ]
    )
    if not plan.logit_rows:
        groups = [
            Group(group.name, Work()) if group.name == "output_head" else group
            for group in groups
        ]
    return tuple(groups)


def _layer_pairs(model, start, rows):
    total = model.num_layers * window_pairs(start, rows, model.window_size)
    for layer in _layer_model(model):
        if layer.compress_ratio > 0:
            total += compressed_pairs(
                start, rows, layer.compress_ratio, model.index_topk
            )
    return total


def _index_pairs(model, start, rows):
    total = 0
    for layer in _layer_model(model):
        if not layer.owns_index_keys:
            continue
        limit = model.candidate_positions if layer.uses_candidate_mask else None
        total += index_score_pairs(
            start, rows, layer.compress_ratio, candidate_limit=limit
        )
    return total


def _prefill_plan(rank, case, model, strategy):
    batch = case["batch"]
    tokens = batch * case["input_tokens"]
    owned = batch if strategy in ("tp4", "tep4") else _partition_rows(batch, rank)
    rows = tokens if strategy in ("tp4", "tep4") else owned * case["input_tokens"]
    pairs = batch * _layer_pairs(model, 0, case["input_tokens"])
    indexes = batch * _index_pairs(model, 0, case["input_tokens"])
    engram = (
        batch * case["input_tokens"] * len(model.engram_layers) * (model.engram_columns)
    )
    if strategy in ("tp4", "tep4"):
        sparse, indexes, engram = (
            pairs // WORLD_SIZE,
            indexes // WORLD_SIZE,
            engram // WORLD_SIZE,
        )
    else:
        sparse = pairs * owned // batch
        indexes = indexes * owned // batch
        engram = engram * owned // batch
    plan = CallPlan(
        rank=rank,
        phase="prefill",
        requests=owned,
        rows=rows,
        global_rows=tokens,
        logit_rows=owned,
        context_end=case["input_tokens"] - 1,
        accepted=0,
        proposed=0,
        sparse_pairs=sparse,
        index_pairs=indexes,
        engram_unique_rows=engram,
        touched_experts=0,
        route_matrix=((0, 0, 0, 0),) * WORLD_SIZE,
    )
    touched = _touched_experts(plan, model, strategy)
    return replace(plan, touched_experts=touched)


def _prefill_plans(case, model, strategy):
    """Return one plan for each rank with the shared dispatch matrix."""
    plans = [_prefill_plan(rank, case, model, strategy) for rank in range(WORLD_SIZE)]
    if strategy != "dep4":
        return plans
    matrix = _route_matrix(plans, model)
    return [replace(plan, route_matrix=matrix) for plan in plans]


def _decode_plans(case, model, strategy):
    """Yield one call plan per target call for each rank."""
    batch = case["batch"]
    drafts = 0 if case["mode"] == "ordinary" else model.draft_tokens
    start = case["input_tokens"]
    remaining = case["output_tokens"] - 1
    while remaining > 0:
        tokens = min(drafts + 1, remaining) if drafts else 1
        rows = batch * tokens
        pairs = batch * _layer_pairs(model, start, tokens)
        indexes = batch * _index_pairs(model, start, tokens)
        engram = batch * tokens * len(model.engram_layers) * model.engram_columns
        plans = []
        for rank in range(WORLD_SIZE):
            owned = (
                batch if strategy in ("tp4", "tep4") else _partition_rows(batch, rank)
            )
            local_rows = owned * tokens
            accepted = owned * (tokens - 1)
            if strategy in ("tp4", "tep4"):
                local_sparse = pairs // WORLD_SIZE
                local_indexes = indexes // WORLD_SIZE
                local_engram = engram // WORLD_SIZE
            else:
                local_sparse = pairs * owned // batch
                local_indexes = indexes * owned // batch
                local_engram = engram * owned // batch
            plan = CallPlan(
                rank=rank,
                phase="decode",
                requests=owned,
                rows=local_rows,
                global_rows=rows,
                logit_rows=local_rows,
                context_end=start + tokens - 1,
                accepted=accepted,
                proposed=accepted,
                sparse_pairs=local_sparse,
                index_pairs=local_indexes,
                engram_unique_rows=local_engram,
                touched_experts=0,
                route_matrix=((0, 0, 0, 0),) * WORLD_SIZE,
            )
            touched = _touched_experts(plan, model, strategy)
            plans.append(replace(plan, touched_experts=touched))
        if strategy == "dep4":
            matrix = _route_matrix(plans, model)
            plans = [replace(plan, route_matrix=matrix) for plan in plans]
        yield plans
        start += tokens
        remaining -= tokens


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """The reference seconds and calculation identity for one case or trace."""

    version: str
    seconds: float
    acceptance: str
    strategy: str
    calls: int
    communication_model: str
    profile_digest: str | None
    communication_rule_version: str
    communication_is_modeled: bool
    communication_is_measured: bool


def reference_case(
    case, *, strategy=None, model=DEEPSEEK_V41_MODEL, communication_model="lower_bound"
):
    """Return the ideal-acceptance reference for one approved HTTP case.

    This function credits every proposed draft. It does not replace a trace.
    """
    for field in ("mode", "batch", "input_tokens", "output_tokens"):
        if field not in case:
            raise ValueError(f"case requires {field}")
    if case["mode"] not in ("dspark5", "ordinary"):
        raise ValueError("case mode must be dspark5 or ordinary")
    strategy = strategy or case.get("strategy") or "tp4"
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    communication_version(communication_model)
    if case.get("strategy") is not None and case["strategy"] != strategy:
        raise ValueError("strategy differs from the case strategy")
    if case["mode"] == "ordinary" and strategy != "tp4":
        raise ValueError("ordinary cases use TP4")
    for field in ("batch", "input_tokens", "output_tokens"):
        _checked_positive_int(case[field], field)

    total = 0.0
    calls = 0
    prefill = _prefill_plans(case, model, strategy)
    groups = [
        call_groups(plan, model, strategy, communication_model=communication_model)
        for plan in prefill
    ]
    total += critical_seconds(groups).seconds
    calls += 1
    for plans in _decode_plans(case, model, strategy):
        groups = [
            call_groups(plan, model, strategy, communication_model=communication_model)
            for plan in plans
        ]
        total += critical_seconds(groups).seconds
        calls += 1
    return ReferenceResult(
        version=communication_version(communication_model),
        seconds=total,
        acceptance="ideal",
        strategy=strategy,
        calls=calls,
        communication_model=communication_model,
        profile_digest=(
            collective_profile_digest()
            if communication_model == "lower_bound"
            else None
        ),
        communication_rule_version=communication_model_metadata(
            strategy, communication_model
        )["communication_rule_version"],
        communication_is_modeled=strategy == "dep4",
        communication_is_measured=False,
    )


def efficiency_percent(reference_seconds, http_seconds):
    """Return the reference seconds as a percentage of the full HTTP seconds."""
    if not math.isfinite(http_seconds) or http_seconds <= 0:
        raise ValueError("full HTTP seconds must be finite and positive")
    if not math.isfinite(reference_seconds) or reference_seconds < 0:
        raise ValueError("reference seconds must be finite and nonnegative")
    return 100 * reference_seconds / http_seconds


BINDING_FIELDS = (
    "source",
    "native_modules",
    "checkpoint_config",
    "checkpoint_index",
    "tokenizer",
    "gpu_uuids",
    "strategy",
    "effective_strategy",
    "owner_plan",
    "sampling",
    "case_id",
    "rate_profile",
    "accepted",
    "proposed",
)


def build_binding(**values):
    """Build and check the accounting identity."""
    version = values.get("version", VERSION)
    values.setdefault("rate_profile", rate_profile_digest(version))
    binding = {"version": version, **values}
    validate_binding(binding)
    return binding


def validate_binding(binding, *, expected=None):
    """Reject an absent, incomplete, or changed accounting identity."""
    if type(binding) is not dict:
        raise BindingError("accounting identity must be an object")
    version = binding.get("version")
    if version not in (VERSION, LEGACY_VERSION):
        raise BindingError("accounting identity has a different version")
    if binding.get("rate_profile") != rate_profile_digest(version):
        raise BindingError("accounting rate profile differs")
    if binding.get("strategy") not in STRATEGIES:
        raise BindingError("accounting identity has an unknown strategy")
    if binding.get("effective_strategy") not in STRATEGIES:
        raise BindingError("accounting identity has an unknown effective strategy")
    try:
        validate_execution_strategy(binding["strategy"], binding["effective_strategy"])
    except ValueError as error:
        raise BindingError(str(error)) from error
    for name in BINDING_FIELDS:
        if name not in binding:
            raise BindingError(f"accounting identity lacks {name}")
        value = binding[name]
        if value is None or value == "" or value == [] or value == {}:
            raise BindingError(f"accounting identity has an empty {name}")
    if (
        type(binding["gpu_uuids"]) is not list
        or len(binding["gpu_uuids"]) != WORLD_SIZE
        or len(set(binding["gpu_uuids"])) != WORLD_SIZE
    ):
        raise BindingError("accounting identity requires four different GPU UUIDs")
    for name in ("accepted", "proposed"):
        _checked_nonnegative_int(binding[name], name)
    if binding["accepted"] > binding["proposed"]:
        raise BindingError("accepted tokens exceed proposed tokens")
    if expected is not None and binding != expected:
        raise BindingError("accounting identity changed")
    return binding


CALL_FIELDS = (
    "rank",
    "call",
    "phase",
    "requests",
    "rows",
    "global_rows",
    "logit_rows",
    "context_end",
    "accepted",
    "proposed",
    "sparse_pairs",
    "index_pairs",
    "engram_unique_rows",
    "touched_experts",
    "layer_ids",
    "route_matrix",
)


def _trace_plan(record, model, strategy):
    for name in CALL_FIELDS:
        if name not in record:
            raise TraceError(f"call record lacks {name}")
    rank = _checked_nonnegative_int(record["rank"], "rank")
    if rank >= WORLD_SIZE:
        raise TraceError("call rank is outside the world")
    if record["phase"] not in ("prefill", "decode"):
        raise TraceError("call phase must be prefill or decode")
    _checked_nonnegative_int(record["call"], "call")
    requests = _checked_nonnegative_int(record["requests"], "requests")
    rows = _checked_nonnegative_int(record["rows"], "rows")
    global_rows = _checked_nonnegative_int(record["global_rows"], "global_rows")
    logit_rows = _checked_nonnegative_int(record["logit_rows"], "logit_rows")
    context_end = _checked_nonnegative_int(record["context_end"], "context_end")
    accepted = _checked_nonnegative_int(record["accepted"], "accepted")
    proposed = _checked_nonnegative_int(record["proposed"], "proposed")
    sparse = _checked_nonnegative_int(record["sparse_pairs"], "sparse_pairs")
    indexes = _checked_nonnegative_int(record["index_pairs"], "index_pairs")
    engram = _checked_nonnegative_int(
        record["engram_unique_rows"], "engram_unique_rows"
    )
    touched = _checked_nonnegative_int(record["touched_experts"], "touched_experts")
    layer_ids = record["layer_ids"]
    if type(layer_ids) is not list and type(layer_ids) is not tuple:
        raise TraceError("layer_ids must be a list")
    if len(layer_ids) != model.num_layers or set(layer_ids) != set(
        range(model.num_layers)
    ):
        raise TraceError("layer set is incomplete")
    local_experts = (
        model.routed_experts
        if strategy == "tp4"
        else model.routed_experts // WORLD_SIZE
    )
    per_layer_assignments = rows * model.experts_per_token
    max_touched = model.num_layers * min(local_experts, per_layer_assignments)
    if touched > max_touched:
        raise TraceError("touched expert count exceeds the causal limit")
    if accepted > proposed:
        raise TraceError("accepted tokens exceed proposed tokens")
    if proposed > requests * model.draft_tokens:
        raise TraceError("proposed tokens exceed the causal draft limit")
    if record["phase"] == "decode":
        if rows != accepted + requests:
            raise TraceError("credited rows differ from accepted tokens plus requests")
        if logit_rows != rows:
            raise TraceError("decode logit rows differ from credited rows")
    elif accepted or proposed:
        raise TraceError("prefill call has invalid target counters")
    elif logit_rows > requests:
        raise TraceError("prefill logit rows exceed the request count")
    if rows == 0 and (requests or logit_rows):
        raise TraceError("zero-row call has a request or logit row")
    if context_end >= model.max_context_tokens:
        raise TraceError("context end exceeds the model context")
    if strategy != "dep4" and rows != global_rows:
        raise TraceError("replicated rank must process all global rows")
    share = 1 if strategy == "dep4" else WORLD_SIZE
    # Attention and index scoring use 16 of 64 and 8 of 32 heads on each
    # TP4/TEP4 rank, so the credited pair count is one quarter.
    if (
        sparse
        > rows * model.num_layers * (model.window_size + model.index_topk) // share
    ):
        raise TraceError("sparse pair count exceeds the causal limit")
    if indexes > rows * len(model.index_source_layers) * (context_end + 1) // share:
        raise TraceError("index pair count exceeds the causal limit")
    if engram > rows * len(model.engram_layers) * model.engram_columns:
        raise TraceError("Engram row count exceeds the causal lookup limit")
    route = record["route_matrix"]
    if (
        type(route) not in (list, tuple)
        or len(route) != WORLD_SIZE
        or any(
            type(row) not in (list, tuple) or len(row) != WORLD_SIZE for row in route
        )
    ):
        raise TraceError("route matrix must be four by four")
    for row in route:
        for count in row:
            _checked_nonnegative_int(count, "route count")
    route = tuple(tuple(row) for row in route)
    if strategy != "dep4" and any(count for row in route for count in row):
        raise TraceError("non-DEP4 trace has a route matrix")
    return CallPlan(
        rank=rank,
        phase=record["phase"],
        requests=requests,
        rows=rows,
        global_rows=global_rows,
        logit_rows=logit_rows,
        context_end=context_end,
        accepted=accepted,
        proposed=proposed,
        sparse_pairs=sparse,
        index_pairs=indexes,
        engram_unique_rows=engram,
        touched_experts=touched,
        route_matrix=route,
    )


def validate_trace(trace, *, strategy, model=DEEPSEEK_V41_MODEL, expected_binding=None):
    """Check the trace contract and return its call plans by call id."""
    if type(trace) is not dict or set(trace) != {"binding", "calls"}:
        raise TraceError("trace requires binding and calls")
    binding = validate_binding(trace["binding"], expected=expected_binding)
    if binding["strategy"] != strategy:
        raise TraceError("trace strategy differs from the requested strategy")
    records = trace["calls"]
    if type(records) is not list or not records:
        raise TraceError("trace requires call records")
    plans = {}
    for record in records:
        plan = _trace_plan(record, model, strategy)
        key = (record["call"], plan.rank)
        if key in plans:
            raise TraceError("trace repeats a rank call")
        plans[key] = plan
    call_ids = sorted({call for call, _rank in plans})
    for call in call_ids:
        present = {rank for (record_call, rank) in plans if record_call == call}
        if present != set(range(WORLD_SIZE)):
            raise TraceError("trace lacks a rank record for a call")
        call_plans = [plans[(call, rank)] for rank in range(WORLD_SIZE)]
        if len({plan.phase for plan in call_plans}) != 1:
            raise TraceError("rank records for a call have different phases")
        if strategy == "dep4":
            matrices = {plan.route_matrix for plan in call_plans}
            if len(matrices) != 1:
                raise TraceError("route matrix differs across ranks")
            route = call_plans[0].route_matrix
            for source in range(WORLD_SIZE):
                expected = call_plans[source].rows * model.experts_per_token
                if sum(route[source]) != expected:
                    raise TraceError(
                        "route row sum differs from the source assignments"
                    )
                if any(count > expected for count in route[source]):
                    raise TraceError("route cell exceeds the source assignments")
            rows = sum(plan.rows for plan in call_plans)
            if rows != call_plans[0].global_rows:
                raise TraceError("DEP4 rank rows differ from the global row count")
        else:
            signatures = {
                (
                    plan.rows,
                    plan.global_rows,
                    plan.logit_rows,
                    plan.context_end,
                    plan.accepted,
                    plan.proposed,
                )
                for plan in call_plans
            }
            if len(signatures) != 1:
                raise TraceError("replicated ranks have different call counters")
    if strategy in ("tp4", "tep4"):
        counted = [plan for (call, rank), plan in plans.items() if rank == 0]
    else:
        counted = list(plans.values())
    accepted = sum(plan.accepted for plan in counted)
    proposed = sum(plan.proposed for plan in counted)
    if accepted != binding["accepted"] or proposed != binding["proposed"]:
        raise TraceError("trace counters differ from the accounting identity")
    return plans, call_ids


def reference_trace(
    trace,
    *,
    strategy,
    model=DEEPSEEK_V41_MODEL,
    expected_binding=None,
    communication_model=None,
):
    """Return the reference seconds of a frozen execution trace."""
    plans, call_ids = validate_trace(
        trace,
        strategy=strategy,
        model=model,
        expected_binding=expected_binding,
    )
    bound_version = trace["binding"]["version"]
    if communication_model is None:
        communication_model = (
            "wire" if bound_version == LEGACY_VERSION else "lower_bound"
        )
    version = communication_version(communication_model)
    if version != bound_version:
        raise BindingError("communication model differs from the trace version")
    prefill_calls = [call for call in call_ids if plans[(call, 0)].phase == "prefill"]
    # The maximum across chunks is a safe lower bound for the union of the
    # touched expert tensors. A full union can only add weight reads.
    prefill_touched = (
        {
            rank: max(plans[(call, rank)].touched_experts for call in prefill_calls)
            for rank in range(WORLD_SIZE)
        }
        if prefill_calls
        else {}
    )
    first_weight_call = {
        rank: next(
            (call for call in prefill_calls if plans[(call, rank)].rows > 0),
            None,
        )
        for rank in range(WORLD_SIZE)
    }
    total = 0.0
    for call in call_ids:
        rank_plans = [plans[(call, rank)] for rank in range(WORLD_SIZE)]
        prefill = rank_plans[0].phase == "prefill"
        groups = []
        for plan in rank_plans:
            include_weights = not prefill or call == first_weight_call[plan.rank]
            if prefill and include_weights:
                plan = replace(plan, touched_experts=prefill_touched[plan.rank])
            groups.append(
                call_groups(
                    plan,
                    model,
                    strategy,
                    include_weights=include_weights,
                    communication_model=communication_model,
                )
            )
        total += critical_seconds(groups).seconds
    metadata = communication_model_metadata(strategy, communication_model)
    return ReferenceResult(
        version=version,
        seconds=total,
        acceptance="trace",
        strategy=strategy,
        calls=len(call_ids),
        communication_model=metadata["communication_model"],
        profile_digest=metadata["profile_digest"],
        communication_rule_version=metadata["communication_rule_version"],
        communication_is_modeled=metadata["dep4_communication_modeled"],
        communication_is_measured=metadata["dep4_communication_measured"],
    )


def ideal_trace(case, *, strategy=None, model=DEEPSEEK_V41_MODEL):
    """Make the ideal-acceptance trace of an approved HTTP case.

    The runtime does not record these counters yet. This function makes the
    ideal plan available to the reference checks and the report table.
    """
    strategy = strategy or case.get("strategy") or "tp4"
    prefill = _prefill_plans(case, model, strategy)
    calls = []
    accepted = proposed = 0
    call_id = 0
    for plan in prefill:
        calls.append(_record(plan, call_id, model))
    call_id += 1
    for plans in _decode_plans(case, model, strategy):
        for plan in plans:
            calls.append(_record(plan, call_id, model))
        if strategy in ("tp4", "tep4"):
            accepted += plans[0].accepted
            proposed += plans[0].proposed
        else:
            accepted += sum(plan.accepted for plan in plans)
            proposed += sum(plan.proposed for plan in plans)
        call_id += 1
    return {
        "binding": build_binding(
            version=VERSION,
            source="ideal",
            native_modules=["ideal"],
            checkpoint_config="ideal",
            checkpoint_index="ideal",
            tokenizer="ideal",
            gpu_uuids=[f"ideal{rank}" for rank in range(WORLD_SIZE)],
            strategy=strategy,
            effective_strategy=strategy,
            owner_plan="ideal",
            sampling=f"greedy-block-{model.draft_tokens}",
            case_id=":".join(
                str(case[name])
                for name in (
                    "mode",
                    "strategy",
                    "batch",
                    "input_tokens",
                    "output_tokens",
                )
                if name in case
            ),
            accepted=accepted,
            proposed=proposed,
        ),
        "calls": calls,
    }


def _record(plan, call, model):
    return {
        "rank": plan.rank,
        "call": call,
        "phase": plan.phase,
        "requests": plan.requests,
        "rows": plan.rows,
        "global_rows": plan.global_rows,
        "logit_rows": plan.logit_rows,
        "context_end": plan.context_end,
        "accepted": plan.accepted,
        "proposed": plan.proposed,
        "sparse_pairs": plan.sparse_pairs,
        "index_pairs": plan.index_pairs,
        "engram_unique_rows": plan.engram_unique_rows,
        "touched_experts": plan.touched_experts,
        "layer_ids": list(range(model.num_layers)),
        "route_matrix": [list(row) for row in plan.route_matrix],
    }


MISSING_RUNTIME_COUNTERS = (
    "execution trace records for ranks 1 to 3",
    "per-call case id and rank record in the execution trace",
    "decode accepted and proposed draft counters per request",
    "per-rank routed assignment count and touched local expert count",
    "per-rank sparse attention selected pair count",
    "per-rank indexer scored pair count",
    "per-rank unique Engram row count",
    "DEP4 dispatch and combine 4x4 route matrix",
    "native module and four GPU UUID identity in the trace",
)
