"""Strict TP4 up comparison over unchanged encoded bytes and all expert owners.

The reference invoke includes dynamic counts, padding, row layout, activation
scale packing, GEMM, and valid output recovery. Only fixed weight layout is
prepared once. This module makes no performance or arithmetic equivalence claim.
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


class ComparisonError(ValueError):
    def __init__(self, state, reason):
        super().__init__(reason)
        self.state = state


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TensorBytes:
    file: Path
    key: str
    row_range: tuple[int, int]
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    @classmethod
    def parse(cls, value):
        return cls(
            Path(value["file"]).resolve(),
            value["key"],
            tuple(value["row_range"]),
            tuple(value["shape"]),
            value["dtype"],
            value["sha256"],
        )

    def read(self):
        with self.file.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ComparisonError("missing", "incomplete safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= min(self.file.stat().st_size - 8, 100_000_000):
                raise ComparisonError("missing", "invalid safetensors header size")
            item = json.loads(stream.read(size))[self.key]
            shape = item["shape"]
            first, last = self.row_range
            unit = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "I32": 4}[
                self.dtype
            ]
            if (
                item["dtype"] != self.dtype
                or not 0 <= first <= last <= shape[0]
                or self.shape != (last - first, *shape[1:])
            ):
                raise ComparisonError(
                    "semantic_mismatch", f"tensor shape/dtype: {self.key}"
                )
            start, end = item["data_offsets"]
            row_bytes = math.prod(shape[1:]) * unit
            if (
                end - start != math.prod(shape) * unit
                or start < 0
                or 8 + size + end > self.file.stat().st_size
            ):
                raise ComparisonError("missing", "incomplete tensor storage")
            stream.seek(8 + size + start + first * row_bytes)
            value = stream.read((last - first) * row_bytes)
        if hashlib.sha256(value).hexdigest() != self.sha256:
            raise ComparisonError("missing", f"tensor digest mismatch: {self.key}")
        return value


@dataclass(frozen=True)
class ExpertWeights:
    expert: int
    weight: TensorBytes
    scales: TensorBytes


@dataclass(frozen=True)
class GroupedCase:
    case_id: str
    case_sha256: str
    groups: int
    outputs: int
    width: int
    counts: tuple[int, ...]
    activation: TensorBytes | None
    activation_scales: TensorBytes | None
    weights: tuple[ExpertWeights, ...]
    identity_json: str
    expected: TensorBytes | None
    generation_json: str | None

    @property
    def rows(self):
        return sum(self.counts)


def validate_case_header(value):
    payload = {key: item for key, item in value.items() if key != "case_sha256"}
    if canonical_hash(payload) != value["case_sha256"]:
        raise ComparisonError("missing", "case identity mismatch")
    if (value["groups"], value["outputs"], value["width"]) != (384, 576, 5120):
        raise ComparisonError(
            "unsupported", "only real tp4_up G384 N576 K5120 is implemented"
        )
    if (
        value["boundary"] != "strict_encoded_expert_major_to_bf16"
        or value["output_dtype"] != "BF16"
    ):
        raise ComparisonError("unsupported", "only strict BF16 output is implemented")
    counts = tuple(value["counts"])
    if (
        len(counts) != 384
        or any(type(x) is not int or not 0 <= x <= 32768 for x in counts)
        or not 0 < sum(counts) <= 32768
    ):
        raise ComparisonError("semantic_mismatch", "invalid complete counts")
    if (
        hashlib.sha256(struct.pack("<384i", *counts)).hexdigest()
        != value["counts_sha256"]
    ):
        raise ComparisonError("missing", "counts digest mismatch")
    identity = value["identity"]
    if (identity["strategy"], identity["world_size"], identity["projection"]) != (
        "tp4",
        4,
        "up",
    ):
        raise ComparisonError("unsupported", "case is not a TP4 up projection")
    if (
        identity["origin"] not in ("captured", "derived_from_capture", "constructed")
        or not identity["source_commit"]
        or not 0 <= identity["rank"] < 4
    ):
        raise ComparisonError("missing", "input provenance is incomplete")
    contract = value["contract"]
    if digest(contract["file"]) != contract["sha256"]:
        raise ComparisonError("missing", "numerical contract digest mismatch")
    if (contract["group_k"], contract["rule"], contract["accumulation"]) != (
        32,
        "full_output_bytes",
        "zero_raw_k32_then_mul_rn_fma_rn",
    ):
        raise ComparisonError("semantic_mismatch", "numerical contract differs")
    return counts, identity


def read_case(value):
    counts, identity = validate_case_header(value)
    weights = tuple(
        ExpertWeights(
            x["expert"], TensorBytes.parse(x["weight"]), TensorBytes.parse(x["scales"])
        )
        for x in value["weights"]
    )
    if tuple(x.expert for x in weights) != tuple(range(384)):
        raise ComparisonError(
            "missing", "all 384 original expert owners are required in order"
        )
    for expert in weights:
        prefix = f"layers.{identity['layer']}.ffn.experts.{expert.expert}.w3."
        if (
            expert.weight.key != prefix + "weight"
            or expert.scales.key != prefix + "scale"
            or expert.weight.row_range
            != (identity["rank"] * 576, (identity["rank"] + 1) * 576)
            or expert.scales.row_range != expert.weight.row_range
        ):
            raise ComparisonError(
                "semantic_mismatch", "expert checkpoint slice differs"
            )
    activation = (
        TensorBytes.parse(value["activation"]) if value.get("activation") else None
    )
    scales = (
        TensorBytes.parse(value["activation_scales"])
        if value.get("activation_scales")
        else None
    )
    expected = TensorBytes.parse(value["expected"]) if value.get("expected") else None
    return GroupedCase(
        value["case_id"],
        value["case_sha256"],
        384,
        576,
        5120,
        counts,
        activation,
        scales,
        weights,
        json.dumps(identity, sort_keys=True),
        expected,
        json.dumps(value["generation"], sort_keys=True)
        if value.get("generation")
        else None,
    )


def verify_generation(case, identity):
    if identity["origin"] == "constructed":
        if not case.case_id.startswith("constructed/"):
            raise ComparisonError(
                "semantic_mismatch", "constructed input needs its own case identity"
            )
        return
    if identity["origin"] != "derived_from_capture":
        return
    if (
        identity.get("input_state") != "derived_input_complete"
        or not case.generation_json
    ):
        raise ComparisonError(
            "missing", "derived input generation is pending or absent"
        )
    source = TensorBytes.parse(identity["bf16_source"])
    if source.dtype != "BF16" or source.shape != (case.rows, case.width):
        raise ComparisonError(
            "semantic_mismatch", "derived input source must be complete BF16"
        )
    source.read()
    binding = json.loads(case.generation_json)
    if digest(binding["file"]) != binding["sha256"]:
        raise ComparisonError("missing", "generation record digest mismatch")
    record = json.loads(Path(binding["file"]).read_text())
    if (
        record["status"] != "completed"
        or record["operation"] != "quantize"
        or canonical_hash(record["parameters"])
        != canonical_hash([8, 32, 1, True, False])
        or TensorBytes.parse(record["bf16_source"]) != source
        or TensorBytes.parse(record["counts_tensor"])
        != TensorBytes.parse(identity["counts_tensor"])
        or record["counts_sha256"]
        != hashlib.sha256(struct.pack("<384i", *case.counts)).hexdigest()
    ):
        raise ComparisonError(
            "semantic_mismatch", "generation inputs or quantize parameters differ"
        )
    for name, item in (
        ("activation", case.activation),
        ("activation_scales", case.activation_scales),
    ):
        if TensorBytes.parse(record["outputs"][name]) != item:
            raise ComparisonError(
                "semantic_mismatch", f"generation output differs: {name}"
            )
    quantizer = record["quantizer"]
    manifest = Path(quantizer["manifest"])
    if digest(manifest) != quantizer["manifest_sha256"]:
        raise ComparisonError("missing", "quantizer manifest digest mismatch")
    native = json.loads(manifest.read_text())
    if (
        native["source_id"] != quantizer["source_id"]
        or not native["source_id"]
        or native["library"] != quantizer["library"]
        or native["binary_id"] != quantizer["binary_sha256"]
        or digest(native["library"]) != native["binary_id"]
        or "quantize" not in native["functions"]
    ):
        raise ComparisonError("missing", "quantizer source or binary identity differs")


def verify_inputs(case):
    identity = json.loads(case.identity_json)
    for source in identity["sources"]:
        if digest(source["file"]) != source["sha256"]:
            raise ComparisonError("missing", "input provenance digest mismatch")
    recorded_counts = TensorBytes.parse(identity["counts_tensor"]).read()
    if recorded_counts != struct.pack("<384i", *case.counts):
        raise ComparisonError(
            "semantic_mismatch", "counts differ from the recorded capture"
        )
    missing = []
    for name, item, shape, dtypes in (
        ("activation", case.activation, (case.rows, case.width), ("U8", "F8_E4M3")),
        (
            "activation_scales",
            case.activation_scales,
            (case.rows, case.width // 32),
            ("U8", "F8_E8M0"),
        ),
    ):
        if item is None:
            missing.append(name)
        elif item.shape != shape or item.dtype not in dtypes:
            raise ComparisonError("semantic_mismatch", f"wrong {name} storage")
        else:
            item.read()
    for expert in case.weights:
        for item, shape, dtypes in (
            (expert.weight, (576, 2560), ("U8", "I8")),
            (expert.scales, (576, 160), ("U8", "F8_E8M0")),
        ):
            if item.shape != shape or item.dtype not in dtypes:
                raise ComparisonError(
                    "semantic_mismatch", "wrong original weight storage"
                )
            item.read()
    if case.expected:
        if case.expected.shape != (case.rows, 576) or case.expected.dtype != "BF16":
            raise ComparisonError("semantic_mismatch", "wrong complete expected output")
        case.expected.read()
    if missing:
        raise ComparisonError(
            "missing", "original encoded input absent: " + ", ".join(missing)
        )
    verify_generation(case, identity)


def load_operands(case):
    import torch

    def device_tensor(item):
        return (
            torch.frombuffer(bytearray(item.read()), dtype=torch.uint8)
            .reshape(item.shape)
            .cuda()
        )

    return {
        "activation": device_tensor(case.activation),
        "scales": device_tensor(case.activation_scales),
        "weights": tuple(
            (device_tensor(x.weight), device_tensor(x.scales)) for x in case.weights
        ),
        "counts": torch.tensor(case.counts, dtype=torch.int32, device="cuda"),
    }


def prepare_native(case, operands, module):
    import torch

    packed = []
    for weight, scales in operands["weights"]:
        target = torch.empty((640 * 160,), dtype=torch.uint8, device="cuda")
        module.pack_weights(weight, scales, weight, target, 4)
        packed.append((weight, target))
    pointers = tuple(
        torch.tensor(
            [pair[i].data_ptr() for pair in packed], dtype=torch.int64, device="cuda"
        )
        for i in (0, 1)
    )
    output = torch.empty((case.rows, 576), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty(
        module.grouped_workspace_size(case.rows, 384, case.rows, 576, 5120),
        dtype=torch.uint8,
        device="cuda",
    )

    def invoke():
        module.grouped_project(
            operands["activation"],
            operands["scales"],
            *pointers,
            operands["counts"],
            output,
            scratch,
            case.rows,
        )
        return output

    return invoke, (operands, packed, pointers, output, scratch)


def pack_scales(scales, module):
    import torch

    mn = scales.size(-2)
    padded_mn = module.get_tma_aligned_size(mn, 4)
    words = scales.contiguous().view(torch.int32)
    storage = torch.empty(
        (*words.shape[:-2], words.size(-1), padded_mn),
        dtype=torch.int32,
        device=scales.device,
    )
    output = storage[..., :mn].transpose(-1, -2)
    output.copy_(words)
    return output


def prepare_deepgemm(case, operands, module):
    import torch

    weights = torch.stack([pair[0] for pair in operands["weights"]]).view(torch.int8)
    weight_scales = pack_scales(
        torch.stack([pair[1] for pair in operands["weights"]]), module
    )
    alignment = module.get_theoretical_mk_alignment_for_contiguous_layout()
    module.set_mk_alignment_for_contiguous_layout(alignment)

    def invoke():
        # Counts and all activation layout work belong to each full invocation.
        counts = operands["counts"]
        padded_counts = (
            (counts.to(torch.int64) + alignment - 1) // alignment
        ) * alignment
        ends = torch.cumsum(padded_counts, 0)
        starts = ends - padded_counts
        physical_rows = int(ends[-1].item())
        groups = torch.repeat_interleave(
            torch.arange(384, device="cuda"), counts.to(torch.int64)
        )
        logical_starts = torch.cumsum(counts.to(torch.int64), 0) - counts
        row_map = (
            starts[groups]
            + torch.arange(case.rows, device="cuda")
            - logical_starts[groups]
        )
        encoded = torch.zeros((physical_rows, 5120), dtype=torch.uint8, device="cuda")
        scales = torch.full((physical_rows, 160), 127, dtype=torch.uint8, device="cuda")
        layout = torch.repeat_interleave(
            torch.arange(384, dtype=torch.int32, device="cuda"), padded_counts
        )
        encoded.index_copy_(0, row_map, operands["activation"])
        scales.index_copy_(0, row_map, operands["scales"])
        activation_scales = pack_scales(scales, module)
        output = torch.empty((physical_rows, 576), dtype=torch.bfloat16, device="cuda")
        module.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (encoded.view(torch.float8_e4m3fn), activation_scales),
            (weights, weight_scales),
            output,
            layout,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=True,
            use_psum_layout=False,
            ensure_zero_padding=True,
        )
        return output.index_select(0, row_map)

    return invoke, (operands, weights, weight_scales, alignment)


def compare_outputs(case, candidate, reference):
    paths = (Path(candidate), Path(reference))
    size = case.rows * case.outputs * 2
    if any(not path.is_file() or path.stat().st_size != size for path in paths):
        return {
            "state": "numeric_failed",
            "reason": "missing or incomplete full BF16 output",
        }
    left, right = (path.read_bytes() for path in paths)
    codes = [list(struct.unpack(f"<{size // 2}H", value)) for value in (left, right)]
    different = sum(x != y for x, y in zip(left, right, strict=True))
    nonfinite = [sum((x & 0x7F80) == 0x7F80 for x in side) for side in codes]
    errors = []
    for a, b in zip(*codes, strict=True):
        fa, fb = (struct.unpack("<f", struct.pack("<I", x << 16))[0] for x in (a, b))
        if math.isfinite(fa) and math.isfinite(fb):
            errors.append(abs(fa - fb))
    expected_difference = None
    if case.expected:
        expected = case.expected.read()
        expected_difference = [
            sum(x != y for x, y in zip(value, expected, strict=True))
            for value in (left, right)
        ]
    failed = different or any(nonfinite) or any(expected_difference or ())
    return {
        "state": "numeric_failed" if failed else "unmeasured",
        "reason": "full output differs or is nonfinite"
        if failed
        else "this input matches; performance unmeasured",
        "different_bytes": different,
        "different_elements": sum(a != b for a, b in zip(*codes, strict=True)),
        "nonfinite_elements": nonfinite,
        "max_abs_error": max(errors, default=0),
        "expected_different_bytes": expected_difference,
        "output_sha256": [digest(path) for path in paths],
        "output_bytes": size,
        "scope": (
            "One complete input only; arithmetic equivalence "
            "and model gate remain unverified."
        ),
    }
