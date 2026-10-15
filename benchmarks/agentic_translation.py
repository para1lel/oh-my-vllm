"""The accepted soak workload: one complete English book in four sealed lanes.

Four independent disposable sandboxes each translate the same validated corpus
into Simplified Chinese through native OMP goals. Translation progress and
completion are workload observations; they never decide framework acceptance.
The staged inputs stay read-only and their identity is a collection gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from benchmarks.agentic_corpus import MANIFEST_NAME, STAGED_DIRECTORY

TARGET_LANGUAGE = "Simplified Chinese"
TRANSLATION_DIRECTORY = "translation"
TASK = f"""# Translate this English book into {TARGET_LANGUAGE}

The directory corpus/ holds one complete English book, retained from its
Project Gutenberg download. corpus/{MANIFEST_NAME} lists the normalized whole
book and its six source volumes. corpus/book.txt is the complete normalized
book. corpus/pg731.txt through corpus/pg736.txt are the same book as its six
source volumes. Treat all of corpus/ as read-only input. Work only in this
directory.

Translate the complete book into {TARGET_LANGUAGE}, one bounded passage at a
time. Translate each source volume into one output file in
{TRANSLATION_DIRECTORY}/, named after its volume with the .md extension:
corpus/pg731.txt becomes {TRANSLATION_DIRECTORY}/pg731.md. Preserve paragraph
boundaries and order, and keep the output free of added commentary, summaries
or invented text. Mark a passage you cannot translate accurately instead of
replacing it with prose of your own. Do not translate the Gutenberg wrapper
lines; start after the volume's start marker.

This is a long task. Start by reading this specification and the manifest, then
work in increments that fit one model call. After each increment, record the
last finished passage, the next unfinished passage and any open terminology
question in notes/progress.md. Automatic context compaction can happen during
the work: after compaction, read notes/progress.md first and continue the same
passage without restarting the book.

Use the read, grep and write tools; the guard rejects edit because this workload
writes whole files. There is no network access and no external translation
service. The guard also rejects writes to corpus/ and this specification.
Verify each volume at the end: compare the number
of translated passages with the source, and correct missing, repeated or
reordered passages before you report the volume as translated.

Complete one goal only when its passage range is translated and recorded. Keep
notes/progress.md truthful about what remains.
"""


def output_name(volume: str) -> str:
    """Map one corpus volume to its fixed translation output file name."""
    return Path(volume).stem + ".md"


def prepare_workspace(directory: Path, corpus) -> dict:
    """Stage the validated corpus and the specification outside inference source."""
    if any(directory.iterdir()):
        raise ValueError("translation workspace must start empty")
    (directory / "TASK.md").write_text(TASK)
    (directory / "notes").mkdir()
    (directory / TRANSLATION_DIRECTORY).mkdir()
    record = corpus.stage(directory)
    return {
        "kind": "native_goal_translation",
        "files": [
            "TASK.md",
            f"{TRANSLATION_DIRECTORY}/",
            f"{STAGED_DIRECTORY}/",
        ],
        "specification_sha256": hashlib.sha256(TASK.encode()).hexdigest(),
        "target_language": TARGET_LANGUAGE,
        "corpus": record,
        "source_root": None,
    }


def objective(index: int) -> str:
    """One queued native goal; the next goal resumes from the recorded passage."""
    if index:
        return (
            "Continue the book translation. Read notes/progress.md, then the "
            f"unfinished passage of the current volume in {STAGED_DIRECTORY}/, and "
            f"keep writing its {TRANSLATION_DIRECTORY}/ file. Translate every "
            "remaining passage of every volume listed by the manifest in order. "
            "When the last passage is translated, check each output against its "
            "source for missing, repeated or reordered passages and correct the "
            "gaps. Preserve earlier output, and record honest remaining gaps and "
            "terminology decisions in notes/progress.md."
        )
    return (
        f"Read TASK.md and {STAGED_DIRECTORY}/{MANIFEST_NAME}. Translate the first "
        f"volume into {TARGET_LANGUAGE} one bounded passage at a time, writing "
        f"{TRANSLATION_DIRECTORY}/<volume stem>.md next to this task. Record the "
        "last finished passage and the next unfinished passage in "
        "notes/progress.md after each increment, and continue this same goal "
        "across automatic context compaction without restarting the book."
    )


def _decoded(path: Path) -> int | None:
    try:
        return len(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError:
        return None


def _files(directory: Path) -> dict:
    return {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _identity_unchanged(staged: Path, record: dict, workspace: Path) -> tuple[bool]:
    """Compare the staged inputs and the specification with their first identity."""
    corpus = record["corpus"]
    manifest = staged / MANIFEST_NAME
    manifest_unchanged = (
        manifest.is_file()
        and hashlib.sha256(manifest.read_bytes()).hexdigest()
        == corpus["manifest_sha256"]
    )
    book = staged / corpus["book"]["name"]
    book_unchanged = (
        book.is_file()
        and book.stat().st_size == corpus["book"]["bytes"]
        and hashlib.sha256(book.read_bytes()).hexdigest() == corpus["book"]["sha256"]
    )
    volumes_unchanged = all(
        (staged / volume["name"]).is_file()
        and (staged / volume["name"]).stat().st_size == volume["bytes"]
        and hashlib.sha256((staged / volume["name"]).read_bytes()).hexdigest()
        == volume["sha256"]
        for volume in corpus["volumes"]
    )
    specification = workspace / "TASK.md"
    specification_unchanged = (
        specification.is_file()
        and hashlib.sha256(specification.read_bytes()).hexdigest()
        == record["specification_sha256"]
    )
    return (
        manifest_unchanged and book_unchanged and volumes_unchanged,
        specification_unchanged,
    )


def observe(client) -> dict:
    """Record translation progress and input integrity without claiming success."""
    workspace = client.workspace
    staged = client.workspace_manifest
    source = workspace / STAGED_DIRECTORY
    output = workspace / TRANSLATION_DIRECTORY
    record = staged["corpus"]
    expected = {volume["name"]: volume for volume in record["volumes"]}
    output_files = _files(output) if output.is_dir() else {}
    corpus_unchanged, specification_unchanged = _identity_unchanged(
        source, staged, workspace
    )
    outputs = {output_name(name): name for name in expected}
    written = sorted(name for name in outputs if (output / name).is_file())
    source_characters = sum(
        count for name in expected if (count := _decoded(source / name)) is not None
    )
    return {
        "kind": "native_goal_translation",
        "target_language": TARGET_LANGUAGE,
        "expected_volumes": len(expected),
        "output_volumes": len(written),
        "missing_output_volumes": sorted(
            outputs[name] for name in outputs if name not in written
        ),
        "translated_characters": sum(
            count for name in written if (count := _decoded(output / name)) is not None
        ),
        "source_characters": source_characters,
        "character_decode": "utf-8",
        "book_unchanged": _book_unchanged(source, record),
        "translation_files": dict(output_files),
        "corpus_unchanged": corpus_unchanged,
        "specification_unchanged": specification_unchanged,
        "progress_notes_present": (workspace / "notes/progress.md").is_file(),
        "task_complete": False,
        "observation_only": (
            "translation progress and completion are workload observations; "
            "they do not decide framework acceptance"
        ),
    }


def _book_unchanged(source: Path, record: dict) -> bool:
    book = source / record["book"]["name"]
    return (
        book.is_file()
        and book.stat().st_size == record["book"]["bytes"]
        and hashlib.sha256(book.read_bytes()).hexdigest() == record["book"]["sha256"]
    )


def require_intact(client, corpus) -> dict:
    """Fail the collection when a lane changed its staged read-only inputs."""
    corpus.verify_staged(client.workspace / STAGED_DIRECTORY)
    specification = client.workspace / "TASK.md"
    if (
        not specification.is_file()
        or hashlib.sha256(specification.read_bytes()).hexdigest()
        != client.workspace_manifest["specification_sha256"]
    ):
        raise ValueError("lane specification changed during collection")
    return {"staged_inputs_verified": True, "specification_verified": True}
