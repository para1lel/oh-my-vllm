import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    ("model", "Term reference A", "术语参考 A", 0, 60),
    ("runtime", "Term reference B", "术语参考 B", 60, 120),
    ("gpu-service", "Term reference C", "术语参考 C", 120, None),
)


def glossary_documents(root: Path) -> dict[Path, str]:
    terms = json.loads((root / "docs/terms.json").read_text())["terms"]
    documents = {}
    for name, title, title_zh, start, end in GROUPS:
        for chinese in (False, True):
            suffix = ".zh.md" if chinese else ".md"
            rows = [
                "# " + (title_zh if chinese else title),
                "",
                "| 术语 | 词性 | 项目含义 |"
                if chinese
                else "| Term | Part of speech | Project meaning |",
                "|---|---|---|",
            ]
            for term in terms[start:end]:
                part = term["part_of_speech"]
                if chinese:
                    part = {"noun": "名词", "verb": "动词"}[part]
                meaning = term["translation"] if chinese else term["meaning"]
                rows.append(
                    "| `" + term["term"] + "` | " + part + " | " + meaning + " |"
                )
            documents[root / "docs/glossary" / (name + suffix)] = "\n".join(rows) + "\n"
    return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    if not (ROOT / "docs/terms.json").is_file():
        print("No glossary source")
        return 0
    failed = False
    for path, text in glossary_documents(ROOT).items():
        if options.check:
            if not path.is_file() or path.read_text() != text:
                print(str(path.relative_to(ROOT)) + ": regenerate from terms.json")
                failed = True
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
