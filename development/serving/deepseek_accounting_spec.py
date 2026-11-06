import math
from dataclasses import dataclass, replace
from functools import lru_cache

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
