import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

MAX_ENGLISH_WORDS = 2000
ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
LINK = re.compile(
    r"!?\[(?P<label>[^\]\n]*)\]\(\s*"
    r"(?P<destination><[^>\n]*>|[^\s)]+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
    r"(?:[ \t]*<!-- source: (?P<anchor>[^\n]+?) -->)?"
)
REFERENCE = re.compile(
    r"^ {0,3}\[(?P<label>[^\]\n]+)\]:[ \t]*"
    r"(?P<destination><[^>\n]*>|[^\s]+)"
    r"(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'))?"
    r"(?:[ \t]*<!-- source: (?P<anchor>[^\n]+?) -->)?[ \t]*$",
    re.M,
)
REFERENCE_USE = re.compile(r"!?\[([^\]\n]+)\](?:\[([^\]\n]*)\])?")
SOURCE_SUFFIXES = {
    ".rs",
    ".py",
    ".cu",
    ".cuh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".js",
    ".mjs",
    ".ts",
    ".tsx",
    ".sh",
    ".toml",
    ".json",
}
LINE_FRAGMENT = re.compile(r"L([1-9]\d*)(?:-L([1-9]\d*))?")
MARKDOWN = MarkdownIt("commonmark").enable("table")


@dataclass(frozen=True)
class DocumentLink:
    destination: str
    anchor: str | None
    destination_span: tuple[int, int]


def reference_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def document_links(text: str) -> list[DocumentLink]:
    text = mask_fenced_code(text)
    text = re.sub(
        r"(?<!`)(`+)(?!`)([\s\S]*?)\1(?!`)",
        lambda match: re.sub(r"[^\n]", " ", match[0]),
        text,
    )
    matches = [*LINK.finditer(text), *REFERENCE.finditer(text)]
    links = []
    for match in matches:
        destination = match["destination"]
        start, end = match.span("destination")
        if destination.startswith("<"):
            destination = destination[1:-1]
            start, end = start + 1, end - 1
        links.append(DocumentLink(destination, match["anchor"], (start, end)))
    return links


def reference_link_issues(text: str) -> list[str]:
    text = mask_fenced_code(text)
    text = re.sub(r"(`+)([^\n]*?)\1", "", text)
    definitions = {reference_label(m["label"]) for m in REFERENCE.finditer(text)}
    occupied = [m.span() for m in LINK.finditer(text)]
    occupied.extend(m.span() for m in REFERENCE.finditer(text))
    issues = []
    for match in REFERENCE_USE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        if match[2] is not None:
            label = reference_label(match[2] or match[1])
            if label not in definitions:
                issues.append(f"missing link reference: {label}")
    return issues


def fenced_code_ranges(text: str) -> list[tuple[int, int, bool]]:
    ranges = []
    fence = None
    offset = start = 0
    for line in text.splitlines(keepends=True):
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            closing = re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line
            )
            if closing:
                ranges.append((start, offset + len(line), True))
                fence = None
        elif opening:
            fence = opening[1]
            start = offset
        offset += len(line)
    if fence:
        ranges.append((start, len(text), False))
    return ranges


def mask_fenced_code(text: str) -> str:
    for start, end, _ in reversed(fenced_code_ranges(text)):
        text = text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]
    return text


def english_word_count(text: str) -> int:
    return sum(
        len(ENGLISH_WORD.findall(re.sub(r"https?://\S+", "", token.content)))
        for token in inline_tokens(text)
        if token.type in {"text", "code_inline"}
    )


def inline_tokens(text: str):
    def descendants(tokens):
        for token in tokens:
            yield token
            if token.children:
                yield from descendants(token.children)

    return descendants(MARKDOWN.parse(text))


def rendered_link_destinations(text: str) -> set[str]:
    return {
        token.attrGet("href" if token.type == "link_open" else "src")
        for token in inline_tokens(text)
        if token.type in {"link_open", "image"}
    }


def local_source(path: Path, destination: str, root: Path) -> Path | None:
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    target = (path.parent / unquote(parsed.path)).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return target if target.suffix in SOURCE_SUFFIXES else None


def source_link_issues(path: Path, text: str, root: Path) -> list[str]:
    issues = []
    links = document_links(text)
    known = {unquote(link.destination) for link in links}
    links.extend(
        DocumentLink(destination, None, (-1, -1))
        for destination in rendered_link_destinations(text)
        if unquote(destination) not in known
    )
    for match in links:
        destination, anchor = match.destination, match.anchor
        target = local_source(path, destination, root)
        if target is None:
            if anchor:
                issues.append(f"source anchor has no local source file: {destination}")
            continue
        source = target.read_text()
        fragment = unquote(urlsplit(destination).fragment)
        if anchor:
            occurrences = source.count(anchor)
            if occurrences != 1:
                issues.append(
                    f"source anchor must occur once, found {occurrences}: "
                    f"{destination} / {anchor}"
                )
                continue
            line = source[: source.index(anchor)].count("\n") + 1
            if fragment != f"L{line}":
                issues.append(f"stale source line: {destination}; expected L{line}")
        elif fragment:
            location = LINE_FRAGMENT.fullmatch(fragment)
            count = len(source.splitlines())
            if not location or not (
                1 <= int(location[1]) <= int(location[2] or location[1]) <= count
            ):
                issues.append(f"invalid source line range: {destination}")
            else:
                issues.append(
                    f"source line reference needs a unique anchor: {destination}"
                )
    return issues


def update_source_links(path: Path, text: str, root: Path) -> str:
    replacements = []
    for match in document_links(text):
        anchor = match.anchor
        target = local_source(path, match.destination, root)
        if not anchor or target is None:
            continue
        source = target.read_text()
        if source.count(anchor) != 1:
            continue
        line = source[: source.index(anchor)].count("\n") + 1
        destination = match.destination.split("#", 1)[0] + f"#L{line}"
        replacements.append((*match.destination_span, destination))
    for start, end, destination in sorted(replacements, reverse=True):
        text = text[:start] + destination + text[end:]
    return text
