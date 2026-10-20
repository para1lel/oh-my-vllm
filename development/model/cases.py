import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from development.operators.cases import digest
from development.operators.fixtures import load_fixture

FIXTURE_ROOT = Path(__file__).with_name("fixtures")


@dataclass(frozen=True)
class Media:
    name: str
    grid: tuple[int, int]
    frames: int
    spans: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Case:
    id: str
    prefix_length: int
    outputs: tuple[str, ...]
    verification_rows: int = 0
    accepted_rows: int = 0
    restore_checkpoint: bool = False
    media: tuple[Media, ...] = ()
    fixture_sha256: str | None = None

    def definition(self):
        values = asdict(self)
        values.pop("fixture_sha256")
        return json.loads(json.dumps(values))

    @property
    def identity(self):
        return digest(self.definition())

    @property
    def output_shapes(self):
        return {
            name: (self.verification_rows, 248320) if name == "verified" else (248320,)
            for name in self.outputs
        }

    @property
    def input_shapes(self):
        shapes = {
            "tokens": ("int64", (self.prefix_length,)),
            "positions": ("int64", (3, self.prefix_length)),
            "continuation": ("int64", (1,)),
        }
        if self.id == "mtp-restored":
            shapes.update(verification=("int64", (5,)), draft_token=("int64", (1,)))
        for media in self.media:
            shapes[media.name + "_pixels"] = (
                "uint8",
                (media.frames, media.grid[0] * 16, media.grid[1] * 16, 3),
            )
        return shapes


def load_cases():
    definitions = (
        dict(
            id="mtp-restored",
            prefix_length=784,
            outputs=("verified", "restored", "mtp_first", "mtp_next"),
            verification_rows=5,
            accepted_rows=2,
            restore_checkpoint=True,
        ),
        dict(
            id="image-video",
            prefix_length=178,
            outputs=("prefill", "decode", "mtp_first"),
            media=(
                Media("image", (16, 16), 1, ((4, 64, 0),)),
                Media("video", (14, 14), 3, ((72, 49, 0), (125, 49, 49))),
            ),
        ),
        dict(id="context-32769", prefix_length=32768, outputs=("prefill", "decode")),
    )
    path = FIXTURE_ROOT / "index.json"
    identities = {}
    if path.exists():
        index = json.loads(path.read_text())
        if (
            set(index) != {"version", "fixtures"}
            or index["version"] != "full-model-fixtures-v1"
        ):
            raise ValueError("invalid full-model fixture index")
        identities = index["fixtures"]
        if set(identities) != {definition["id"] for definition in definitions}:
            raise ValueError("full-model fixture index is incomplete")
    return tuple(
        Case(**definition, fixture_sha256=identities.get(definition["id"]))
        for definition in definitions
    )


def load_case_fixture(case, root=None):
    fixture = load_fixture(case, FIXTURE_ROOT / case.id if root is None else root)
    if fixture.state or fixture.manifest["writes"]:
        raise ValueError(
            "full-model cases create state from their stored prefix inputs"
        )
    if set(fixture.inputs) != set(case.input_shapes):
        raise ValueError("full-model fixture input fields differ from its case")
    for name, (dtype, shape) in case.input_shapes.items():
        array = fixture.inputs[name]
        if str(array.dtype) != dtype or array.shape != shape:
            raise ValueError(
                "full-model fixture input differs from its fixed shape or dtype"
            )
    for name, array in fixture.inputs.items():
        if name not in {"positions", "image_pixels", "video_pixels"} and (
            np.any(array < 0) or np.any(array >= 248320)
        ):
            raise ValueError("full-model input contains invalid tokens")
    for name, array in fixture.expected.items():
        if (
            array.shape != case.output_shapes[name]
            or array.dtype != np.float64
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "full-model endpoint must contain the complete finite FP64 vocabulary"
            )
    return fixture
