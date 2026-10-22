import hashlib
import io
import json
from pathlib import Path

import numpy as np

SHORT_PATH = Path(__file__).with_name("reference.npz")
SHORT_SHA256 = "3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0"
CODEC_PATH = Path(__file__).with_name("codec.json")
CODEC_SHA256 = "30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75"
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
