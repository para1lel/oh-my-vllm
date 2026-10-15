import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def glossary_documents(root: Path) -> dict[Path, str]:
    terms = json.loads((root / "docs/terms.json").read_text())["terms"]
    documents = {}
    for chinese in (False, True):
        suffix = ".zh.md" if chinese else ".md"
        rows = [
            "# " + ("项目术语" if chinese else "Project terms"),
            "",
            "| 术语 | 词性 | 项目含义 |"
            if chinese
            else "| Term | Part of speech | Project meaning |",
            "|---|---|---|",
        ]
        for term in terms:
            part = term["part_of_speech"]
            if chinese:
                part = {"noun": "名词", "verb": "动词"}[part]
            meaning = term["translation"] if chinese else term["meaning"]
            rows.append("| `" + term["term"] + "` | " + part + " | " + meaning + " |")
        documents[root / "docs" / ("glossary" + suffix)] = "\n".join(rows) + "\n"
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
