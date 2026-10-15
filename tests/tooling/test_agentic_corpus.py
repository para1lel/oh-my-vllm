"""The retained download identity is pinned; a rewritten manifest cannot pass."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.agentic_corpus import (
    ACCEPTED,
    MANIFEST_NAME,
    ROOT,
    STAGED_DIRECTORY,
    Corpus,
)
from tests.support.corpus import synthetic_book, write_manifest


def test_accepted_identity_pins_the_retained_gutenberg_download():
    assert ACCEPTED.manifest_sha256 == (
        "437ab1493f865ef0bc2b986562b3bc6ae05553dac95e6488a8126b10642cafd8"
    )
    assert ACCEPTED.book_file == "book.txt"
    assert ACCEPTED.book_sha256 == (
        "ac06cd5616c0d77093da396a48d21dfe289ed49fef43ceebc04d976b1639ec99"
    )
    assert (ACCEPTED.book_bytes, ACCEPTED.book_english_words) == (10_500_674, 1_558_522)
    assert [volume.gutenberg_id for volume in ACCEPTED.volumes] == [
        731,
        732,
        733,
        734,
        735,
        736,
    ]
    assert [volume.volume for volume in ACCEPTED.volumes] == [1, 2, 3, 4, 5, 6]
    assert [volume.name for volume in ACCEPTED.volumes] == [
        f"pg{identifier}.txt" for identifier in range(731, 737)
    ]
    for volume in ACCEPTED.volumes:
        assert volume.bytes > 0 and len(volume.sha256) == 64
    # Raw downloads include their Gutenberg wrappers: their total is independent.
    assert sum(volume.bytes for volume in ACCEPTED.volumes) != ACCEPTED.book_bytes


def test_retained_schema_validates_and_stages_every_input(tmp_path):
    directory, identity, manifest = synthetic_book(tmp_path)
    corpus = Corpus(directory, identity)
    record = corpus.record()
    assert record["manifest_sha256"] == identity.manifest_sha256
    assert record["book"] == {
        "name": "book.txt",
        "bytes": manifest["book_bytes"],
        "sha256": manifest["book_sha256"],
        "english_words": manifest["book_english_words"],
    }
    assert [volume["name"] for volume in record["volumes"]] == [
        f"pg{item['gutenberg_id']}.txt" for item in manifest["volumes"]
    ]
    assert json.dumps(record).find(str(tmp_path)) == -1

    target = tmp_path / "lane"
    target.mkdir()
    staged = corpus.stage(target)
    assert staged == record
    staged_directory = target / STAGED_DIRECTORY
    assert sorted(path.name for path in staged_directory.iterdir()) == [
        "book.txt",
        "manifest.json",
        "pg731.txt",
        "pg732.txt",
    ]
    assert (
        hashlib.sha256((staged_directory / "book.txt").read_bytes()).hexdigest()
        == manifest["book_sha256"]
    )
    corpus.verify_staged(staged_directory)


@pytest.mark.parametrize("rewrite_manifest", [False, True])
def test_same_size_replacement_with_a_rewritten_digest_is_rejected(
    tmp_path, rewrite_manifest
):
    directory, identity, manifest = synthetic_book(tmp_path)
    volume = directory / "pg731.txt"
    replaced = bytes(byte ^ 0xFF for byte in volume.read_bytes())
    assert len(replaced) == volume.stat().st_size
    volume.write_bytes(replaced)
    if rewrite_manifest:
        # The rewritten manifest certifies the substitute; the pinned digest must not.
        manifest["volumes"][0]["sha256"] = hashlib.sha256(replaced).hexdigest()
        write_manifest(directory, manifest)
        with pytest.raises(ValueError, match="differs from the retained download"):
            Corpus(directory, identity)
    else:
        with pytest.raises(ValueError, match="differs from the retained identity"):
            Corpus(directory, identity)


def test_manifest_mutation_after_initial_validation_is_rejected(tmp_path):
    directory, identity, manifest = synthetic_book(tmp_path)
    corpus = Corpus(directory, identity)

    manifest["title"] = "A different book"
    write_manifest(directory, manifest)
    with pytest.raises(ValueError, match="differs from the retained download"):
        Corpus(directory, identity)

    volume = directory / "pg731.txt"
    volume.write_bytes(b"z" * volume.stat().st_size)
    with pytest.raises(ValueError, match="corpus volume differs"):
        corpus.verify()


@pytest.mark.parametrize(
    "damage,message",
    [
        ("manifest", "differs from the retained download"),
        ("book", "accepted book bytes differ"),
        ("volume", "corpus volume differs"),
    ],
)
def test_staged_inputs_are_rechecked_after_collection(tmp_path, damage, message):
    directory, identity, _manifest = synthetic_book(tmp_path)
    corpus = Corpus(directory, identity)
    target = tmp_path / "lane"
    target.mkdir()
    corpus.stage(target)
    staged = target / STAGED_DIRECTORY
    path = {
        "manifest": staged / MANIFEST_NAME,
        "book": staged / "book.txt",
        "volume": staged / "pg732.txt",
    }[damage]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match=message):
        corpus.verify_staged(staged)


@pytest.mark.parametrize(
    "damage,message",
    [
        ("book_missing", "accepted book is missing"),
        ("book_size", "accepted book bytes differ"),
        ("book_content", "accepted book bytes differ"),
        ("volume_missing", "corpus volume is missing"),
        ("volume_size", "corpus volume differs"),
        ("volume_content", "corpus volume differs"),
    ],
)
def test_missing_or_changed_files_fail_validation(tmp_path, damage, message):
    directory, identity, _manifest = synthetic_book(tmp_path)
    book, volume = directory / "book.txt", directory / "pg732.txt"
    if damage == "book_missing":
        book.unlink()
    elif damage == "book_size":
        book.write_bytes(book.read_bytes()[:5])
    elif damage == "book_content":
        book.write_bytes(b"x" * book.stat().st_size)
    elif damage == "volume_missing":
        volume.unlink()
    elif damage == "volume_size":
        volume.write_bytes(volume.read_bytes()[:-1])
    else:
        volume.write_bytes(b"y" * volume.stat().st_size)
    with pytest.raises(ValueError, match=message):
        Corpus(directory, identity)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda m: m.pop("title"), "title is invalid"),
        (lambda m: m.update(book_file="../book.txt"), "book file differs"),
        (lambda m: m.update(book_sha256="0" * 64), "book digest differs"),
        (lambda m: m.update(book_bytes=1), "book size differs"),
        (lambda m: m.update(book_english_words=1), "book words differ"),
        (lambda m: m.update(downloaded_unix_s=-1), "download time is invalid"),
        (lambda m: m.update(derivation=""), "derivation is invalid"),
        (lambda m: m.update(volumes=[]), "six volumes"),
        (
            lambda m: m["volumes"][0].update(gutenberg_id=999),
            "unknown volume",
        ),
        (lambda m: m["volumes"][1].update(volume=1), "repeats a volume number"),
        (lambda m: m["volumes"][0].update(sha256="0" * 64), "digest differs"),
        (lambda m: m["volumes"][0].update(bytes=1), "size differs"),
        (lambda m: m["volumes"][0].update(url="http://example.invalid"), "url differs"),
    ],
)
def test_manifest_identity_fields_are_enforced(tmp_path, mutate, message):
    directory, identity, manifest = synthetic_book(tmp_path)
    mutate(manifest)
    digest = write_manifest(directory, manifest)
    import dataclasses

    with pytest.raises(ValueError, match=message):
        Corpus(directory, dataclasses.replace(identity, manifest_sha256=digest))


@pytest.mark.parametrize("raw", [b"", b"[]", b"not json", b"x" * ((1 << 20) + 1)])
def test_broken_manifest_bytes_are_rejected(tmp_path, raw):
    directory = tmp_path / "book"
    directory.mkdir()
    (directory / MANIFEST_NAME).write_bytes(raw)
    with pytest.raises(ValueError, match="manifest"):
        Corpus(directory, ACCEPTED)


def test_book_inside_the_source_repository_is_rejected():
    with pytest.raises(ValueError, match="outside the source repository"):
        Corpus(ROOT / "benchmarks", ACCEPTED)


def test_retained_corpus_validates_when_the_external_directory_is_provided():
    """Run with OH_MY_VLLM_CORPUS to bind the real retained download."""
    external = os.environ.get("OH_MY_VLLM_CORPUS")
    if not external:
        pytest.skip("set OH_MY_VLLM_CORPUS to the retained corpus directory")
    corpus = Corpus(Path(external))
    assert corpus.identity == ACCEPTED
    record = corpus.record()
    assert record["manifest_sha256"] == ACCEPTED.manifest_sha256
    assert record["book"]["sha256"] == ACCEPTED.book_sha256
    assert len(record["volumes"]) == 6
