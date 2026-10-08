"""Check portable documents, translation contracts, and mechanical STE rules.

Approved meanings, actual parts of speech, and translation fidelity require
independent review against the official standard. This tool is not certification.
"""

import argparse
import collections
import ipaddress
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```[^\n]*\n[\s\S]*?^```\s*$", re.M)
INLINE = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\[([^\]]+)\]\(([^\s)]+)\)")
SENTENCE = re.compile(r"(?<=[.!?])(?:\s+|$)")
WORDS = re.compile(r"[A-Za-z0-9]+(?:[-'/][A-Za-z0-9]+)*")
HOST_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9/])|(?<=-I)|(?<=-L))/(?:data\d+/shared|home/[A-Za-z0-9_.-]+|root)(?:/|\b)"
)
GPU_ID = re.compile(r"GPU-[0-9a-f]{4,}(?:-[0-9a-f]+)*(?:\.{3}|…)?", re.I)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
UNSUITABLE = re.compile(
    r"\b(?:preserv(?:e|es|ed)|retain(?:s|ed)?|confirm(?:s|ed)?|"
    r"inspect(?:s|ed)?|expos(?:e|es|ed)|need(?:s)?|require(?:s)?|"
    r"remain(?:s|ed|ing)?|establish(?:es|ed)?|exhaust(?:s|ed)?|"
    r"explain(?:s|ed)?|describ(?:e|es|ed)|every|both|including|"
    r"outside|within|under|actual|real)\b",
    re.I,
)
ALIASES = {"KV-cache": "KV cache", "prefilling": "prefill", "inferencing": "inference"}


def owned_files(root):
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    return sorted(
        {
            Path(name.decode())
            for name in output.split(b"\0")
            if name
            and not name.startswith(b"3rdparty/")
            and (root / name.decode()).is_file()
        }
    )


def prose(text, keep_quotes=False):
    text = FENCE.sub("", text)
    text = LINK.sub(lambda m: m[1], text)
    text = INLINE.sub(" QUOTED ", text)
    text = re.sub(r"https?://\S+", " QUOTED ", text)
    if not keep_quotes:
        text = re.sub(r'"[^"\n]+"', " QUOTED ", text)
    return text


def word_count(text):
    # Rule 8: inline code/quotes/formulas and parenthetical phrases count once.
    text = re.sub(r"\([^()]*\)", " PAREN ", text)
    text = re.sub(r"\$[^$]+\$", " FORMULA ", text)
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s+(?:ms|s|seconds?|MiB|GiB|bytes|px)\b", " UNIT ", text
    )
    text = re.sub(r"\b\d+(?:\.\d+)+(?:[a-z]+\d*)?\b", " NUMBER ", text)
    return len(WORDS.findall(text))


def sentences(text):
    # Decimal dots and dotted names do not introduce whitespace boundaries.
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]


def markdown_issues(text, chinese=False):
    errors = []
    if len(re.findall(r"^```", text, re.M)) % 2:
        errors.append("unclosed fenced code block")
    clean = prose(text, keep_quotes=chinese)
    if chinese:
        clean = re.sub(
            r"\b(?:[A-Za-z][A-Za-z0-9_-]*\.)+[A-Za-z0-9_-]+\b|\b\d+(?:\.\d+)*:\d+\b",
            " QUOTED ",
            clean,
        )
        clean = re.sub(r"\b\d+(?:\.\d+)+(?:\.post\d+)?\b", " NUMBER ", clean)
        if re.search(r"[、，。；：！？（）【】“”‘’]", clean):  # noqa: RUF001
            errors.append("Chinese prose has fullwidth punctuation")
        if re.search(r"(?<=\S)\(", clean):
            errors.append("Chinese prose needs a space before a left parenthesis")
        if re.search(r"[,;:!?)](?=\w)|\.(?=[A-Za-z\u3400-\u9fff])", clean):
            errors.append("Chinese prose needs a space after punctuation")
        if re.search(r"[\u3400-\u9fff][A-Za-z0-9]|[A-Za-z0-9][\u3400-\u9fff]", clean):
            errors.append("Chinese and English/numeric text need separating spaces")
        return errors
    for number, paragraph in enumerate(clean.split("\n\n"), 1):
        if not paragraph.strip() or paragraph.lstrip().startswith("#"):
            continue
        fragments = []
        listing = all(
            line.lstrip().startswith(("- ", "|")) or re.match(r"\d+\. ", line)
            for line in paragraph.splitlines()
        )
        if listing:
            for line in paragraph.splitlines():
                if line.startswith("|"):
                    fragments.extend(
                        cell.strip() for cell in line.strip("|").split("|")
                    )
                else:
                    fragments.append(re.sub(r"^\s*(?:- |\d+\. )", "", line))
        else:
            fragments = [paragraph]
        for fragment in fragments:
            # Rule 8 counts parentheses once outside, but their own sentences
            # must satisfy the same limits.
            internal = re.findall(r"\(([^()]*)\)", fragment)
            outer = re.sub(r"\([^()]*\)", " PAREN ", fragment)
            for sentence in sentences(outer) + [
                s for p in internal for s in sentences(p)
            ]:
                if word_count(sentence) > 25:
                    errors.append(
                        f"paragraph {number}: descriptive sentence exceeds 25 words: "
                        f"{sentence[:75]}"
                    )
                # Other procedural sentences need independent review.
                if (
                    re.match(
                        r"^(?:Use|Set|Keep|Install|Start|Stop|Read|Make|Add|Put|Select|Complete|Record|Do|Examine|Reject|Compare|Profile|Test|Run|Build|Wait|Resume|Remove|Evict|Apply|Limit|Translate|Count|Supply|Restore)\b",
                        sentence,
                    )
                    and word_count(sentence) > 20
                ):
                    errors.append(
                        f"paragraph {number}: procedure sentence exceeds 20 words: "
                        f"{sentence[:75]}"
                    )
        if not listing and len(sentences(paragraph)) > 6:
            errors.append(
                f"paragraph {number}: descriptive paragraph exceeds 6 sentences"
            )
    if ";" in clean:
        errors.append("English prose has a semicolon")
    for word in sorted(set(UNSUITABLE.findall(clean))):
        errors.append(f"use approved general wording instead of {word!r}")
    for old, canonical in ALIASES.items():
        if re.search(rf"\b{re.escape(old)}\b", clean, re.I):
            errors.append(f"use canonical term {canonical!r} instead of {old!r}")
    return errors


def pairing_issues(english, chinese):
    errors = []
    if FENCE.findall(english) != FENCE.findall(chinese):
        errors.append("translation changes fenced code/commands")
    a = collections.Counter(INLINE.findall(FENCE.sub("", english)))
    b = collections.Counter(INLINE.findall(FENCE.sub("", chinese)))
    if a != b:
        errors.append(
            f"translation changes inline identifiers: missing={dict(a - b)}, "
            f"extra={dict(b - a)}"
        )
    # Explicit numbers must occur in the translation. Spelled number meanings
    # and row-by-row semantics remain an independent translation review.
    numeric = re.compile(r"(?<!\d)\d+(?:\.\d+)*(?:%|px)?(?!\d)")
    missing = set(numeric.findall(prose(english))) - set(
        numeric.findall(prose(chinese))
    )
    if missing:
        errors.append(f"translation omits explicit numeric values: {sorted(missing)}")
    if re.findall(r"^#+ ", english, re.M) != re.findall(r"^#+ ", chinese, re.M):
        errors.append("translation changes heading structure")
    return errors


def anchors(text):
    result = set()
    counts = collections.Counter()
    for match in re.finditer(r"^#+ (.+)$", FENCE.sub("", text), re.M):
        title = re.sub(r"[`*_]", "", match[1]).lower()
        slug = "".join(
            c for c in title if c in " -_" or unicodedata.category(c)[0] in "LN"
        ).replace(" ", "-")
        result.add(slug if counts[slug] == 0 else f"{slug}-{counts[slug]}")
        counts[slug] += 1
    return result


def link_issues(path, text, root):
    errors = []
    for _, destination in LINK.findall(FENCE.sub("", text)):
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc:
            continue
        target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        if not target.exists():
            errors.append(f"missing local link: {destination}")
        elif (
            parsed.fragment
            and target.suffix == ".md"
            and unquote(parsed.fragment) not in anchors(target.read_text())
        ):
            errors.append(f"missing Markdown anchor: {destination}")
        if not target.is_relative_to(root.resolve()):
            errors.append(f"local link leaves repository: {destination}")
    return errors


def host_issues(text):
    errors = []
    if HOST_PATH.search(text):
        errors.append("host-specific absolute path")
    if GPU_ID.search(text):
        errors.append("host-specific GPU identity")
    for match in IPV4.finditer(text):
        try:
            address = ipaddress.ip_address(match[0])
        except ValueError:
            continue
        # Private network addresses are host facts; package versions are not.
        if (
            address.is_private
            and not address.is_loopback
            and not address.is_unspecified
        ):
            line = text[
                text.rfind("\n", 0, match.start()) + 1 : text.find("\n", match.end())
                if "\n" in text[match.end() :]
                else len(text)
            ]
            if not ("nvidia-curand" in line and match[0] == "10.4.0.35"):
                errors.append("host-specific private network address")
    return errors


def glossary_issues(root):
    errors = []
    data = json.loads((root / "docs/terms.json").read_text())
    seen = set()
    for entry in data["terms"]:
        key = entry["term"], entry["part_of_speech"]
        if key in seen or entry["part_of_speech"] not in {"noun", "verb"}:
            errors.append(f"invalid or duplicate glossary entry: {key}")
        seen.add(key)
        if (
            not entry["meaning"]
            or not entry["translation"]
            or entry["term"] not in entry["forms"]
        ):
            errors.append(f"incomplete glossary contract: {key}")
        if not all(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[ -][A-Za-z][A-Za-z0-9]*)*", form)
            for form in entry["forms"]
        ):
            errors.append(f"invalid registered glossary form: {key}")
        for language in ("glossary.md", "glossary.zh.md"):
            text = (root / "docs" / language).read_text()
            meaning = (
                entry["translation"]
                if language.endswith(".zh.md")
                else entry["meaning"]
            )
            pos = entry["part_of_speech"]
            if language.endswith(".zh.md"):
                pos = {"noun": "名词", "verb": "动词"}[pos]
            if f"| `{entry['term']}` | {pos} | {meaning} |" not in text:
                errors.append(f"glossary definition differs in {language}: {key}")
    return errors


def check(root):
    issues = []
    files = owned_files(root)
    for relative in files:
        if relative == Path("LOCAL.md") or relative.parts[0] == ".local":
            issues.append(f"{relative}: local maintenance content must not be tracked")
        path = root / relative
        try:
            text = path.read_text()
        except (UnicodeError, OSError):
            continue
        issues.extend(f"{relative}: {error}" for error in host_issues(text))
        if path.suffix != ".md":
            continue
        chinese = path.name.endswith(".zh.md")
        issues.extend(
            f"{relative}: {error}" for error in markdown_issues(text, chinese)
        )
        issues.extend(f"{relative}: {error}" for error in link_issues(path, text, root))
        counterpart = (
            path.with_name(path.name.removesuffix(".zh.md") + ".md")
            if chinese
            else path.with_name(path.stem + ".zh.md")
        )
        if not counterpart.is_file():
            issues.append(f"{relative}: missing translation companion")
        elif not chinese:
            issues.extend(
                f"{relative}: {error}"
                for error in pairing_issues(text, counterpart.read_text())
            )
    issues.extend(f"docs/terms.json: {error}" for error in glossary_issues(root))
    for ignored in ("LOCAL.md", ".local/evidence/example.json"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", ignored], cwd=root, check=False
        )
        if result.returncode != 0:
            issues.append(f".gitignore: {ignored} must be ignored")
    return issues, sum(p.suffix == ".md" for p in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    issues, count = check(args.root.resolve())
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print(
        f"Document checks passed: {count} Markdown files; "
        "independent STE/translation review remains mandatory."
    )


if __name__ == "__main__":
    main()
