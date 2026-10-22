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
