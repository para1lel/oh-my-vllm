"""Retained-shape corpus books for identity, staging and gate tests."""

import hashlib
import json
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def write_manifest(directory: Path, manifest: dict) -> str:
    """Write one manifest and return its digest."""
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (directory / MANIFEST_NAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def synthetic_identity(manifest: dict, manifest_sha256: str):
    from benchmarks.agentic_corpus import CorpusIdentity, VolumeIdentity

    return CorpusIdentity(
        manifest_sha256=manifest_sha256,
        book_file=manifest["book_file"],
        book_sha256=manifest["book_sha256"],
        book_bytes=manifest["book_bytes"],
        book_english_words=manifest["book_english_words"],
        volumes=tuple(
            VolumeIdentity(
                record["volume"],
                record["gutenberg_id"],
                record["bytes"],
                record["sha256"],
            )
            for record in manifest["volumes"]
        ),
    )


def synthetic_book(tmp_path: Path, *, identifiers=(731, 732), mutate=None):
    """Build a retained-shape book whose volume bytes differ from the book total."""
    directory = tmp_path / "book"
    directory.mkdir()
    book = b"The complete normalized book.\n" * 4
    (directory / "book.txt").write_bytes(book)
    volumes = []
    for number, identifier in enumerate(identifiers, start=1):
        payload = f"volume {identifier}\n".encode() * (3 + number)
        (directory / f"pg{identifier}.txt").write_bytes(payload)
        volumes.append(
            {
                "volume": number,
                "gutenberg_id": identifier,
                "url": f"https://www.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt",
                "final_url": f"https://www.gutenberg.org/cache/epub/{identifier}/pg{identifier}.txt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "body_english_words": 10 + number,
            }
        )
    manifest = {
        "title": "A retained book",
        "author": "An author",
        "source": "Project Gutenberg, test volumes",
        "book_file": "book.txt",
        "book_sha256": hashlib.sha256(book).hexdigest(),
        "book_bytes": len(book),
        "book_english_words": 20,
        "volumes": volumes,
        "downloaded_unix_s": 1_791_000_000.5,
        "derivation": "Normalized line endings; wrapper removed.",
    }
    if mutate is not None:
        mutate(manifest)
    digest = write_manifest(directory, manifest)
    return directory, synthetic_identity(manifest, digest), manifest
