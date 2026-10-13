"""Check document rules against failures that would affect a portable clone."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from scripts.documentation_contracts import english_word_count, update_source_links

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "docs_check", ROOT / "scripts/check_docs.py"
)
docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs)


def test_document_limit_counts_visible_words_and_protects_code_and_destinations():
    text = (
        "# Title\n\n" + "word " * 1997 + "`cache_block`\n\n"
        "[Guide](https://example.test/a-long-address)\n\n"
        "~~~python\n" + "code " * 500 + "\n~~~\n"
    )
    assert english_word_count(text) == 2001
    assert english_word_count(text.replace("word ", "", 1)) == 2000


def test_source_links_detect_ambiguity_stale_lines_and_invalid_ranges(tmp_path):
    source = tmp_path / "worker.py"
    source.write_text("import torch\n\ndef execute_model():\n    pass\n")
    document = tmp_path / "README.md"
    text = "[Execute](worker.py#L1) <!-- source: def execute_model( -->\n"
    assert any("stale" in issue for issue in docs.link_issues(document, text, tmp_path))
    updated = update_source_links(document, text, tmp_path)
    assert "worker.py#L3" in updated
    assert docs.link_issues(document, updated, tmp_path) == []
    source.write_text(source.read_text() + "\ndef execute_model():\n    pass\n")
    assert any(
        "found 2" in issue for issue in docs.link_issues(document, updated, tmp_path)
    )
    assert update_source_links(document, updated, tmp_path) == updated
    assert docs.link_issues(document, "[Bad](worker.py#L2-L100)", tmp_path)
    assert docs.link_issues(document, "[Bad](worker.py#L4-L2)", tmp_path)
    assert docs.link_issues(document, "[File](worker.py)", tmp_path) == []
    assert any(
        "unique anchor" in issue
        for issue in docs.link_issues(document, "[Line](worker.py#L1)", tmp_path)
    )
    assert docs.link_issues(document, "```md\n" + text + "```\n", tmp_path) == []


def test_translation_checks_commands_identifiers_values_and_structure():
    english = (
        "# Run\n\nUse `model` with 784 tokens.\n\n```sh\nrun --model example\n```\n"
    )
    chinese = (
        "# 运行\n\n使用 `model`, block 为 784 个 token.\n\n"
        "```sh\nrun --model example\n```\n"
    )
    assert docs.pairing_issues(english, chinese) == []
    for before, after, expected in (
        ("run --model example", "run --model other", "fenced"),
        ("`model`", "`other`", "identifiers"),
        ("784", "785", "numeric"),
        ("# 运行", "## 运行", "heading"),
    ):
        assert any(
            expected in issue
            for issue in docs.pairing_issues(english, chinese.replace(before, after))
        )


def test_angle_and_reference_links_check_targets_and_update_source_lines(tmp_path):
    source = tmp_path / "source file.py"
    source.write_text("\ndef execute():\n    pass\n")
    document = tmp_path / "README.md"
    text = (
        "[Execute][worker]\n\n"
        "[worker]: <source file.py#L1> <!-- source: def execute( -->\n"
    )
    updated = update_source_links(document, text, tmp_path)
    assert "<source file.py#L2>" in updated
    assert docs.link_issues(document, updated, tmp_path) == []
    assert docs.link_issues(document, "[Missing](<missing source.py>)", tmp_path)
    assert docs.link_issues(document, "[Missing][undefined]", tmp_path)
    assert docs.link_issues(document, "[Missing][x]\n\n[x]: missing.py", tmp_path)


def test_owned_document_enumeration_does_not_follow_external_symlinks(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    target = tmp_path / "external.md"
    original = "[Missing](outside.py#L1) <!-- source: def execute( -->\n"
    target.write_text(original)
    (repository / "linked.md").symlink_to(target)
    assert docs.owned_files(repository) == []
    assert target.read_text() == original


def test_reference_before_inline_updates_keep_offsets_when_line_width_changes(tmp_path):
    (tmp_path / "worker.py").write_text("\n" * 30 + "def execute():\n    pass\n")
    document = tmp_path / "README.md"
    text = (
        "[worker]: worker.py#L1 <!-- source: def execute( -->\n\n"
        "[Inline](worker.py#L1) <!-- source: def execute( -->\n"
    )
    expected = text.replace("worker.py#L1", "worker.py#L31")
    assert update_source_links(document, text, tmp_path) == expected
    assert docs.link_issues(document, expected, tmp_path) == []


def test_commonmark_links_use_visible_labels_and_protect_inline_examples(tmp_path):
    document = tmp_path / "README.md"
    for text in (
        "[Source](missing.py (Source title))",
        r"[Source\] label](missing.py)",
    ):
        assert docs.link_issues(document, text, tmp_path)
    assert docs.link_issues(document, "`[Source][undefined]`", tmp_path) == []
    assert (
        english_word_count("[One][hidden-long-label]\n\n[hidden-long-label]: file.py")
        == 1
    )
    assert english_word_count("[One](file.py)") == 1


def test_sentence_limits_include_parenthetical_internal_sentences():
    descriptive = "The model " + " ".join(["has"] * 24) + "."
    procedure = "Use " + " ".join(["the"] * 20) + "."
    assert any("25 words" in issue for issue in docs.markdown_issues(descriptive))
    assert any("20 words" in issue for issue in docs.markdown_issues(procedure))
    assert any(
        "25 words" in issue
        for issue in docs.markdown_issues(f"The model works ({descriptive}).")
    )
    assert (
        docs.word_count(
            "The model uses 2.14.0 and 12 MiB (one long explanatory phrase)."
        )
        == 7
    )


def test_prose_checks_protect_commands_but_reject_unclosed_fences():
    assert (
        docs.markdown_issues("# Command\n\n```sh\nretain; " + "word " * 30 + "\n```\n")
        == []
    )
    assert any(
        "semicolon" in issue
        for issue in docs.markdown_issues("The model runs; the process stops.")
    )
    assert any(
        "unclosed" in issue for issue in docs.markdown_issues("```sh\ncommand\n")
    )
    assert any(
        "6 sentences" in issue for issue in docs.markdown_issues("It works. " * 7)
    )


@pytest.mark.parametrize(
    "text",
    [
        "模型，数据.",  # noqa: RUF001
        "模型(token).",
        "模型,数据.",
        "模型GPU.",
        '他说 "你好。".',
        "这里 1.这是下一句.",
        "模型、数据.",
    ],
)
def test_chinese_prose_spacing_and_punctuation(text):
    assert docs.markdown_issues(text, chinese=True)
    assert (
        docs.markdown_issues("模型 (token), GPU 使用 784 个 token.", chinese=True) == []
    )


def test_links_check_unicode_anchors_missing_targets_and_repository_boundary(tmp_path):
    target = tmp_path / "target.md"
    target.write_text("# 缓存\n\n# 缓存\n")
    source = tmp_path / "README.md"
    source.write_text("[缓存](target.md#缓存-1)")
    assert docs.link_issues(source, source.read_text(), tmp_path) == []
    assert docs.link_issues(source, "[bad](target.md#other)", tmp_path)
    assert docs.link_issues(source, "[bad](missing.md)", tmp_path)
    assert docs.link_issues(source, "[bad](../outside.md)", tmp_path)
    target.write_text("# Real\n\n```sh\n# fake-heading\n```\n")
    assert docs.link_issues(source, "[bad](target.md#fake-heading)", tmp_path)


def test_images_do_not_add_prose_punctuation_and_keep_path_checks(tmp_path):
    text = "# 请求图\n\n![MTP4 请求吞吐](requests.svg)\n"
    source = tmp_path / "README.zh.md"
    image = tmp_path / "requests.svg"
    image.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    assert docs.markdown_issues(text, chinese=True) == []
    assert docs.link_issues(source, text, tmp_path) == []
    image.unlink()
    assert docs.link_issues(source, text, tmp_path) == [
        "missing local link: requests.svg"
    ]
    assert docs.markdown_issues("错误!标点.", chinese=True)


def test_host_checks_distinguish_identity_from_package_version():
    path = str(Path("/") / "home" / "fixture" / "model")
    uuid = "GPU-" + "abcdef01-1234"
    address = ".".join(["10", "20", "30", "40"])
    for identity in (path, uuid, address):
        assert docs.host_issues(identity)
    assert docs.host_issues("-I" + path)
    assert docs.host_issues("GPU-" + "a4b4...")
    assert docs.host_issues("nvidia-curand==10.4.0.35 other=" + address)
    assert docs.host_issues("/cuda/home/bin/nvcc") == []
    assert docs.host_issues("nvidia-curand==10.4.0.35\nListen on 127.0.0.1\n") == []


def test_repository_check_requires_pairs_ignored_storage_and_consistent_glossary(
    tmp_path,
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/LOCAL.md\n/.local/\n")
    (tmp_path / "README.md").write_text("# Model\n\nThe model works.\n")
    (tmp_path / "README.zh.md").write_text("# 模型\n\n模型可以运行.\n")
    directory = tmp_path / "docs"
    directory.mkdir()
    terms = {
        "terms": [
            {
                "term": "token",
                "part_of_speech": "noun",
                "meaning": "One vocabulary ID.",
                "translation": "一个词表 ID.",
                "forms": ["token"],
            }
        ]
    }
    (directory / "terms.json").write_text(json.dumps(terms))
    (directory / "glossary.md").write_text(
        "# Terms\n\n| `token` | noun | One vocabulary ID. |\n"
    )
    (directory / "glossary.zh.md").write_text(
        "# 术语\n\n| `token` | 名词 | 一个词表 ID. |\n"
    )
    assert docs.check(tmp_path)[0] == []
    (tmp_path / "README.md").write_text("# Model\n\n" + "word.\n\n" * 2000)
    assert any("2000" in issue for issue in docs.check(tmp_path)[0])
    (tmp_path / "README.md").write_text("# Model\n\nThe model works.\n")
    (tmp_path / "LOCAL.md").write_text("# Local\n\n" + "word.\n\n" * 2000)
    assert any("LOCAL.md: 2001" in issue for issue in docs.check(tmp_path)[0])
    (tmp_path / "LOCAL.md").unlink()
    (tmp_path / "README.zh.md").unlink()
    assert any("missing translation" in issue for issue in docs.check(tmp_path)[0])
    (directory / "glossary.zh.md").write_text(
        "# 术语\n\n| `token` | 名词 | 修改后的含义. |\n"
    )
    assert any("glossary definition" in issue for issue in docs.check(tmp_path)[0])
    (tmp_path / "LOCAL.md").write_text("# Local\n")
    subprocess.run(["git", "add", "-f", "LOCAL.md"], cwd=tmp_path, check=True)
    assert any("must not be tracked" in issue for issue in docs.check(tmp_path)[0])
