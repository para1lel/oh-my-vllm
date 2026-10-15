"""Validated external corpus for the accepted translation workload.

The retained directory holds the original download manifest, the normalized
complete book, and the six raw volume downloads. The accepted digests are pinned
here, so a rewritten manifest cannot certify its own replacement and a
same-sized substituted file is rejected before any lane starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "manifest.json"
STAGED_DIRECTORY = "corpus"
MAX_MANIFEST_BYTES = 1 << 20
VOLUME_NAME = re.compile(r"pg[1-9][0-9]{0,5}\.txt")
DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VolumeIdentity:
    """One raw Gutenberg volume: its number, identifier, size and digest."""

    volume: int
    gutenberg_id: int
    bytes: int
    sha256: str

    @property
    def name(self) -> str:
        return f"pg{self.gutenberg_id}.txt"


@dataclass(frozen=True)
class CorpusIdentity:
    """Retained identity of one complete book and its downloaded volumes."""

    manifest_sha256: str
    book_file: str
    book_sha256: str
    book_bytes: int
    book_english_words: int
    volumes: tuple[VolumeIdentity, ...]


ACCEPTED = CorpusIdentity(
    manifest_sha256="437ab1493f865ef0bc2b986562b3bc6ae05553dac95e6488a8126b10642cafd8",
    book_file="book.txt",
    book_sha256="ac06cd5616c0d77093da396a48d21dfe289ed49fef43ceebc04d976b1639ec99",
    book_bytes=10_500_674,
    book_english_words=1_558_522,
    volumes=(
        VolumeIdentity(
            1,
            731,
            1_820_548,
            "7568212cddf0780c66d46444eefb8f46e9dfe3be9328d90452c7b006ff975ca3",
        ),
        VolumeIdentity(
            2,
            732,
            2_002_534,
            "14da613aa65ca79fd10724dcd7f17cd51c4759ef87bb747b9055f4dabeec9777",
        ),
        VolumeIdentity(
            3,
            733,
            1_694_109,
            "049787410750146ad7d63988fa69a9bafe71f779aa25bf08bd1486a6a3022ea0",
        ),
        VolumeIdentity(
            4,
            734,
            1_893_450,
            "435303e8a4c4cf2ad6fc7c3e1ab7c88935c91f088cdbf70723b9e0ba62cb177e",
        ),
        VolumeIdentity(
            5,
            735,
            1_781_293,
            "aa4d88193599ece7491d44066af1fde3299734054a778e3905f1eda1ee5a1ecc",
        ),
        VolumeIdentity(
            6,
            736,
            1_598_836,
            "f30c4488d4cd93ab246fce972981f8168927bd9a31587e3b2df852bb60143d8e",
        ),
    ),
)


def _digest(path: Path) -> str:
    collected = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            collected.update(chunk)
    return collected.hexdigest()


def _text(manifest: dict, name: str, description: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"corpus manifest {description} is invalid")
    return value


def _positive(value, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"corpus manifest {description} is invalid")
    return value


def _checked_manifest(raw: bytes, identity: CorpusIdentity) -> dict:
    """Validate the retained manifest against the pinned accepted identity."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("corpus manifest exceeds its size bound")
    if hashlib.sha256(raw).hexdigest() != identity.manifest_sha256:
        raise ValueError("corpus manifest differs from the retained download record")
    try:
        manifest = json.loads(raw)
    except ValueError as error:
        raise ValueError("corpus manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest must be a JSON object")
    for name in ("title", "author", "source", "derivation"):
        _text(manifest, name, name)
    book_file = _text(manifest, "book_file", "book file")
    if book_file != identity.book_file or Path(book_file).name != book_file:
        raise ValueError("corpus manifest book file differs from the accepted book")
    if _text(manifest, "book_sha256", "book digest") != identity.book_sha256:
        raise ValueError("corpus manifest book digest differs from the accepted book")
    if _positive(manifest.get("book_bytes"), "book size") != identity.book_bytes:
        raise ValueError("corpus manifest book size differs from the accepted book")
    if (
        _positive(manifest.get("book_english_words"), "book word count")
        != identity.book_english_words
    ):
        raise ValueError("corpus manifest book words differ from the accepted book")
    downloaded = manifest.get("downloaded_unix_s")
    if (
        type(downloaded) not in (int, float)
        or not math.isfinite(downloaded)
        or downloaded <= 0
    ):
        raise ValueError("corpus manifest download time is invalid")
    volumes = manifest.get("volumes")
    if not isinstance(volumes, list) or len(volumes) != len(identity.volumes):
        raise ValueError("corpus manifest must list the accepted six volumes")
    seen = set()
    for record in volumes:
        if not isinstance(record, dict):
            raise ValueError("corpus manifest volume must be an object")
        number = _positive(record.get("volume"), "volume number")
        identifier = _positive(record.get("gutenberg_id"), "volume identifier")
        if number in seen:
            raise ValueError("corpus manifest repeats a volume number")
        seen.add(number)
        expected = next(
            (item for item in identity.volumes if item.gutenberg_id == identifier), None
        )
        if expected is None or expected.volume != number:
            raise ValueError("corpus manifest lists an unknown volume")
        if record.get("sha256") != expected.sha256:
            raise ValueError(f"corpus manifest volume {identifier} digest differs")
        if _positive(record.get("bytes"), "volume size") != expected.bytes:
            raise ValueError(f"corpus manifest volume {identifier} size differs")
        _positive(record.get("body_english_words"), "volume word count")
        for name in ("url", "final_url"):
            if not _text(record, name, f"volume {identifier} {name}").startswith(
                "https://"
            ):
                raise ValueError(f"corpus manifest volume {identifier} {name} differs")
    if seen != {item.volume for item in identity.volumes}:
        raise ValueError("corpus manifest does not cover the accepted volumes")
    return manifest


def _verify_files(directory: Path, identity: CorpusIdentity):
    """Bind the book and every volume on disk to the accepted digests."""
    book = directory / identity.book_file
    if not book.is_file():
        raise ValueError(f"accepted book is missing: {identity.book_file}")
    if (
        book.stat().st_size != identity.book_bytes
        or _digest(book) != identity.book_sha256
    ):
        raise ValueError("accepted book bytes differ from the retained identity")
    for volume in identity.volumes:
        path = directory / volume.name
        if not path.is_file():
            raise ValueError(f"corpus volume is missing: {volume.name}")
        if path.stat().st_size != volume.bytes or _digest(path) != volume.sha256:
            raise ValueError(
                f"corpus volume differs from the retained identity: {volume.name}"
            )


class Corpus:
    """One validated external book; verify it before and after a run."""

    def __init__(self, directory: Path, identity: CorpusIdentity = ACCEPTED):
        self.directory = directory.resolve()
        self.identity = identity
        if self.directory.is_relative_to(ROOT):
            raise ValueError("corpus must live outside the source repository")
        manifest_path = self.directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError("corpus manifest is missing")
        self.manifest = _checked_manifest(manifest_path.read_bytes(), identity)
        self.verify()

    def verify(self):
        """Every accepted input must match its pinned size and full digest."""
        _verify_files(self.directory, self.identity)

    def verify_staged(self, staged: Path):
        """Recheck the staged manifest, book and volumes after collection."""
        manifest = staged / MANIFEST_NAME
        if not manifest.is_file():
            raise ValueError("staged corpus manifest is missing")
        _checked_manifest(manifest.read_bytes(), self.identity)
        _verify_files(staged, self.identity)

    def record(self) -> dict:
        """Portable identity without host paths."""
        return {
            "title": self.manifest["title"],
            "author": self.manifest["author"],
            "source": self.manifest["source"],
            "derivation": self.manifest["derivation"],
            "downloaded_unix_s": self.manifest["downloaded_unix_s"],
            "manifest_sha256": self.identity.manifest_sha256,
            "book": {
                "name": self.identity.book_file,
                "bytes": self.identity.book_bytes,
                "sha256": self.identity.book_sha256,
                "english_words": self.identity.book_english_words,
            },
            "volumes": [
                {
                    "volume": volume.volume,
                    "gutenberg_id": volume.gutenberg_id,
                    "name": volume.name,
                    "bytes": volume.bytes,
                    "sha256": volume.sha256,
                }
                for volume in self.identity.volumes
            ],
        }

    def stage(self, target: Path) -> dict:
        """Copy the verified inputs into one lane workspace and rebind them."""
        staged = target / STAGED_DIRECTORY
        staged.mkdir(parents=True)
        shutil.copyfile(self.directory / MANIFEST_NAME, staged / MANIFEST_NAME)
        shutil.copyfile(
            self.directory / self.identity.book_file, staged / self.identity.book_file
        )
        for volume in self.identity.volumes:
            shutil.copyfile(self.directory / volume.name, staged / volume.name)
        self.verify_staged(staged)
        return self.record()
