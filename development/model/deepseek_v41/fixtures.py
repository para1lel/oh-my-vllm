import hashlib
import io
import json
from pathlib import Path

import numpy as np

SHORT_PATH = Path(__file__).with_name("reference.npz")
SHORT_SHA256 = "3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0"
CODEC_PATH = Path(__file__).with_name("codec.json")
CODEC_SHA256 = "30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75"
DSPARK_PATH = Path(__file__).with_name("dspark.npz")
DSPARK_SHA256 = "c78d89d66d2ff38999af2aa44b845c6d2fea1702280c217a92e7a0d693ee6fa8"
MEDIA_PATH = Path(__file__).with_name("media.npz")
MEDIA_SHA256 = "a847254bb8a1b858d67eeb7405edb4a5435c24f2280585ed18f6b4b0eee86223"
MEDIA_INPUTS_SHA256 = "968e863392fe7ec3531e4df6a7d43851a295f070c486fec80565abc63a7449bc"
MEDIA_SOURCES_SHA256 = (
    "c099eb57db022052006567e726dbd3cd4f56c5bebd35eaf7bce45a3ed2383dd7"
)
MEDIA_SHAPES = {"prefill": (129280,), "decode": (129280,)}
EXACT_OPERATORS_PATH = Path(__file__).with_name("exact-operators.npz")
EXACT_OPERATORS_SHA256 = (
    "a422c1f4cadeeb4ef918a18253834eae848c32b5173b0b88398e9bf0ede23e79"
)
EXACT_OPERATORS_METADATA_SHA256 = (
    "cce21cf671863f16e5a8e5c03a4becddff2bce0263af05636c76bb980b27a994"
)
ENGRAM_LOOKUP_PATH = Path(__file__).with_name("engram-lookup.npz")
ENGRAM_LOOKUP_SHA256 = (
    "930cee947ae1ee8bcdd3d15571732713305d849c18df8c4d98418dc309e40c2c"
)
ENGRAM_LOOKUP_METADATA_SHA256 = (
    "5833b226c5c3914ed11a6bbbf60a89e989a280e18f43142bfb1e77944a6dbada"
)
ENGRAM_CHECKPOINT_PATH = Path(__file__).with_name("engram-checkpoint.npz")
ENGRAM_CHECKPOINT_SHA256 = (
    "967109d7d713191212dc37fc18335c926fa3e2420a79e9fe03f5f2bbe365efce"
)
ENGRAM_CHECKPOINT_METADATA_SHA256 = (
    "90b9b52d5e371c375d13d4c4a4fae397cc393689aee0123961fde355dee4550a"
)
HYPERCONNECTION_PATH = Path(__file__).with_name("hyperconnection")
HYPERCONNECTION_SHA256 = (
    "933323bf315a75806e9dc8f66b0da47777622cd36c8b98f262f29b3978afdb47"
)
ROUTING_PATH = Path(__file__).with_name("routing")
ROUTING_SHA256 = "bbb8538a912bede95a6d4364f52a63a026b04385de2e34cad5030109d305bf28"
PROJECTION_PATH = Path(__file__).with_name("projection")
PROJECTION_SHA256 = "db9c81bb0dd4a2f10aa3fdd289349dfb9162ede223a7ceb5cbe0e80444646fc6"
CHECKPOINT_MANIFEST_SHA256 = (
    "0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67"
)
SHORT_TOKENS = (
    671,
    4787,
    13769,
    46012,
    54994,
    1060,
    270,
    41638,
    6397,
    16,
    28231,
    3939,
    270,
    12709,
    344,
    8295,
)
DSPARK_SHAPES = {
    "prefill": (129280,),
    "draft_logits": (5, 129280),
    "draft_confidence": (5,),
    "verify_logits": (6, 129280),
    "restored_logits": (129280,),
    "next_draft_logits": (5, 129280),
    "next_draft_confidence": (5,),
}
DSPARK_INPUTS = {
    "tokens": list((SHORT_TOKENS * 9)[:129]),
    "pending_token": 4787,
    "proposal_tokens": [13769, 46012, 54994, 1060, 270],
    "verification_tokens": [4787, 13769, 46012, 54994, 1060, 270],
    "accepted_input_rows": 2,
    "accepted_drafts": 1,
    "correction_token": 46012,
    "next_pending_position": 131,
    "next_proposal_tokens": [54994, 1060, 270, 41638, 6397],
}


def _load_operator_bundle(path, identity):
    from development.operators.cases import Case
    from development.operators.fixtures import load_fixture

    path = Path(path)
    raw = (path / "index.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity:
        raise ValueError("operator fixture index identity differs")
    index = json.loads(raw)
    cases = {}
    for name, record in index["cases"].items():
        case = Case(**record["case"], fixture_sha256=record["fixture_sha256"])
        cases[name] = load_fixture(case, path / case.id)
    return index, cases


def load_hyperconnection(path=HYPERCONNECTION_PATH):
    return _load_operator_bundle(path, HYPERCONNECTION_SHA256)


def load_routing(path=ROUTING_PATH):
    return _load_operator_bundle(path, ROUTING_SHA256)


def load_projection(path=PROJECTION_PATH):
    return _load_operator_bundle(path, PROJECTION_SHA256)


def load_engram_lookup(path=ENGRAM_LOOKUP_PATH):
    """Load encoded local lookup inputs and exact BF16 bits without an oracle."""
    return _load_engram(path, ENGRAM_LOOKUP_SHA256, ENGRAM_LOOKUP_METADATA_SHA256)


def load_engram_checkpoint(path=ENGRAM_CHECKPOINT_PATH):
    """Load original checkpoint samples without checkpoint or runtime access."""
    return _load_engram(
        path, ENGRAM_CHECKPOINT_SHA256, ENGRAM_CHECKPOINT_METADATA_SHA256
    )


def _load_engram(path, digest, metadata_digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("frozen Engram lookup artifact identity differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        metadata = json.loads(stored["metadata"].item())
        canonical = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != metadata_digest:
            raise ValueError("Engram lookup case or provenance differs")
        names = set(metadata["arrays"])
        if (
            set(stored.files) != {"metadata", *names}
            or len(stored.files) != len(names) + 1
        ):
            raise ValueError("Engram lookup inputs or outputs are incomplete")
        arrays = {}
        for name, description in metadata["arrays"].items():
            value = stored[name].copy()
            if (
                list(value.shape) != description["shape"]
                or value.dtype != np.dtype(description["dtype"])
                or hashlib.sha256(value.tobytes()).hexdigest() != description["sha256"]
            ):
                raise ValueError(f"Engram lookup tensor identity differs: {name}")
            value.setflags(write=False)
            arrays[name] = value
    return metadata, arrays


def load_exact_operators(path=EXACT_OPERATORS_PATH):
    """Load synthetic source-precision literals, without reference execution."""
    from development.operators.fixtures import Fixture

    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != EXACT_OPERATORS_SHA256:
        raise ValueError("frozen exact-operator artifact identity differs")
    cases = {}
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        metadata = json.loads(stored["metadata"].item())
        canonical = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != EXACT_OPERATORS_METADATA_SHA256:
            raise ValueError("exact-operator case definitions or provenance differ")
        names = {
            f"{name}.{field}"
            for name, record in metadata["cases"].items()
            for field in record["tensors"]
        }
        if (
            set(stored.files) != {"metadata", *names}
            or len(stored.files) != len(names) + 1
        ):
            raise ValueError("exact-operator inputs, state or outputs are incomplete")
        for name, record in metadata["cases"].items():
            groups = {"inputs": {}, "state": {}, "expected": {}}
            for field, description in record["tensors"].items():
                group, key = field.split(".", 1)
                value = stored[f"{name}.{field}"].copy()
                if (
                    list(value.shape) != description["shape"]
                    or value.dtype != np.dtype(description["dtype"])
                    or hashlib.sha256(value.tobytes()).hexdigest()
                    != description["sha256"]
                ):
                    raise ValueError(
                        f"exact-operator tensor identity differs: {name}.{field}"
                    )
                value.setflags(write=False)
                groups[group][key] = value
            cases[name] = Fixture(record, **groups)
    return metadata, cases


def load_media(path=MEDIA_PATH):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != MEDIA_SHA256:
        raise ValueError("frozen media-reference artifact identity differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        if set(stored.files) != {"metadata", *MEDIA_SHAPES}:
            raise ValueError("media-reference vocabulary endpoints are incomplete")
        metadata = json.loads(stored["metadata"].item())
        arrays = {name: stored[name].copy() for name in MEDIA_SHAPES}
    manifest = (
        json.dumps(metadata["checkpoint"], sort_keys=True, indent=2) + "\n"
    ).encode()
    inputs = json.dumps(
        metadata["inputs"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    sources = json.dumps(
        metadata["reference_sources"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if (
        metadata["schema_version"] != 1
        or metadata["case"] != "still-and-video-frames"
        or metadata["reference_kind"]
        != "independent-pytorch-fp64-no-activation-quantization"
        or metadata["checkpoint_manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or hashlib.sha256(manifest).hexdigest() != CHECKPOINT_MANIFEST_SHA256
        or metadata["checkpoint"]["verification"] != "full"
        or metadata["execution"]
        != {"device": "cpu", "threads": 8, "projection_row_limit": 2048}
        or hashlib.sha256(inputs).hexdigest() != MEDIA_INPUTS_SHA256
        or hashlib.sha256(sources).hexdigest() != MEDIA_SOURCES_SHA256
        or metadata["token_map_sha256"]
        != "26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496"
    ):
        raise ValueError("media-reference inputs or provenance differ")
    if set(metadata["arrays"]) != set(MEDIA_SHAPES):
        raise ValueError("media-reference array metadata is incomplete")
    for name, value in arrays.items():
        shape = MEDIA_SHAPES[name]
        if (
            value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or metadata["arrays"][name]
            != {
                "shape": list(shape),
                "dtype": "float64",
                "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            }
        ):
            raise ValueError(f"media-reference full-vector identity differs: {name}")
    if int(arrays["prefill"].argmax()) != metadata["inputs"]["decode_token"]:
        raise ValueError("media-reference continuation token differs")
    return metadata, arrays


def load_dspark(path=DSPARK_PATH):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != DSPARK_SHA256:
        raise ValueError("frozen DSpark-reference artifact identity differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        if set(stored.files) != {"metadata", *DSPARK_SHAPES}:
            raise ValueError("DSpark-reference vocabulary endpoints are incomplete")
        metadata = json.loads(stored["metadata"].item())
        arrays = {name: stored[name].copy() for name in DSPARK_SHAPES}
    manifest = (
        json.dumps(metadata["checkpoint"], sort_keys=True, indent=2) + "\n"
    ).encode()
    if (
        metadata["schema_version"] != 1
        or metadata["case"] != "dspark-restored"
        or metadata["reference_kind"]
        != "independent-pytorch-fp64-no-activation-quantization"
        or metadata["checkpoint_manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or hashlib.sha256(manifest).hexdigest() != CHECKPOINT_MANIFEST_SHA256
        or metadata["checkpoint"]["verification"] != "full"
        or metadata["execution"]["device"] != "cpu"
        or metadata["inputs"] != DSPARK_INPUTS
        or metadata["state_selection"]
        != "forced-two-input-prefix; correction is target argmax, not sampled rejection"
    ):
        raise ValueError("DSpark-reference inputs or provenance differ")
    if set(metadata["arrays"]) != set(DSPARK_SHAPES):
        raise ValueError("DSpark-reference array metadata is incomplete")
    for name, value in arrays.items():
        shape = DSPARK_SHAPES[name]
        if (
            value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or metadata["arrays"][name]
            != {
                "shape": list(shape),
                "dtype": "float64",
                "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            }
        ):
            raise ValueError(f"DSpark-reference full-vector identity differs: {name}")
    return metadata, arrays


def load_short(path=SHORT_PATH):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHORT_SHA256:
        raise ValueError("frozen short-reference artifact identity differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as stored:
        if set(stored.files) != {"metadata", "prefill", "decode"}:
            raise ValueError("short-reference vocabulary endpoints are incomplete")
        metadata = json.loads(stored["metadata"].item())
        arrays = {name: stored[name].copy() for name in ("prefill", "decode")}
    if (
        metadata["schema_version"] != 1
        or metadata["case"] != "short"
        or metadata["reference_kind"]
        != "independent-pytorch-fp64-no-activation-quantization"
        or metadata["checkpoint_manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or tuple(metadata["tokens"]) != SHORT_TOKENS
        or metadata["decode_token"] != 16
    ):
        raise ValueError("short-reference inputs or provenance differ")
    for name, value in arrays.items():
        if (
            value.shape != (129280,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or metadata["arrays"][name]["sha256"]
            != hashlib.sha256(value.astype("<f8").tobytes()).hexdigest()
        ):
            raise ValueError(f"short-reference full-vector identity differs: {name}")
    return metadata, arrays


def load_codec(path=CODEC_PATH):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != CODEC_SHA256:
        raise ValueError("frozen prompt-codec artifact identity differs")
    artifact = json.loads(data)
    if (
        artifact["schema_version"] != 1
        or artifact["reference_kind"] != "local-python-prompt-codec-cpu"
        or artifact["checkpoint_manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or artifact["tokenization"]
        != {
            "loader": "Tokenizer.from_buffer(verified_bytes)",
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        }
    ):
        raise ValueError("prompt-codec provenance differs")
    cases = artifact["cases"]
    if len(cases) != 20 or len({case["name"] for case in cases}) != 20:
        raise ValueError("prompt-codec case set differs")
    for case in cases:
        if not case["token_ids"] or any(
            type(token) is not int or not 0 <= token < 129280
            for token in case["token_ids"]
        ):
            raise ValueError("prompt-codec token IDs differ")
    return artifact
