"""Use Python AST, Syn, Acorn, and pinned Tree-sitter grammars for spans."""

import ast
import importlib
import io
import json
import subprocess
import sys
import tokenize
from importlib.metadata import version
from pathlib import Path

PINS = {
    "tree-sitter": "0.25.2",
    "tree-sitter-cpp": "0.23.4",
    "tree-sitter-cuda": "0.21.2",
    "tree-sitter-bash": "0.25.0",
    "identify": "2.6.20",
}
EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".rs": "rust",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".sh": "bash",
    ".bash": "bash",
}
DATA_SUFFIXES = {
    ".md",
    ".json",
    ".lock",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".csv",
    ".png",
    ".npz",
    ".svg",
    ".pdf",
    ".jpg",
    ".jpeg",
}
DATA_NAMES = {".gitignore", ".gitattributes", ".clang-format", ".gitmodules", "LICENSE"}
FUNCTIONS = {
    "cpp": {"function_definition", "lambda_expression"},
    "cuda": {"function_definition", "lambda_expression"},
    "bash": {"function_definition"},
}


def validate_versions() -> None:
    if sys.version_info[:3] != (3, 15, 2):
        raise RuntimeError("Python 3.15.2 required for the pinned Python AST parser")
    for package, expected in PINS.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"{package}=={expected} required; found {actual}")


def source_language(path: Path) -> str | None:
    from identify.identify import (
        parse_shebang,
        tags_from_interpreter,
        tags_from_path,
    )

    if path.suffix in EXTENSIONS:
        return EXTENSIONS[path.suffix]
    tags = tags_from_path(str(path))
    with path.open("rb") as stream:
        if stream.read(2) == b"#!":
            stream.seek(0)
            for interpreter in parse_shebang(stream):
                tags.update(tags_from_interpreter(interpreter))
    if "shell" in tags:
        if tags.intersection({"csh", "tcsh", "zsh", "fish"}):
            raise ValueError(f"unsupported shell language: {sorted(tags)}")
        return "bash"
    if "python" in tags:
        return "python"
    if path.suffix in DATA_SUFFIXES or path.name in DATA_NAMES or "binary" in tags:
        return None
    raise ValueError(f"unclassified owned file: {path.name}; tags={sorted(tags)}")


def parser_for(language: str):
    from tree_sitter import Language, Parser

    module = importlib.import_module(f"tree_sitter_{language}")
    return Parser(Language(module.language()))


def walk(node):
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(reversed(current.children))


def function_start(node):
    start = node
    while start.parent and start.parent.type == "template_declaration":
        start = start.parent
    return start.start_point.row + 1


def function_name(node) -> str:
    name = node.child_by_field_name("name")
    declarator = node.child_by_field_name("declarator")
    while name is None and declarator is not None:
        if declarator.type in {
            "identifier",
            "field_identifier",
            "qualified_identifier",
            "operator_name",
            "destructor_name",
        }:
            name = declarator
            break
        declarator = declarator.child_by_field_name("declarator")
    return (
        name.text.decode("utf-8")
        if name
        else f"<{node.type}@{node.start_point.row + 1}>"
    )


def inspect_python(source: bytes) -> tuple[list, list]:
    try:
        syntax = ast.parse(source)
    except SyntaxError as error:
        return [], [{"line": error.lineno, "message": str(error)}]
    functions = []
    tokens = list(tokenize.tokenize(io.BytesIO(source).readline))
    for node in ast.walk(syntax):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            start, end = python_span(node, tokens)
            functions.append(
                {
                    "name": getattr(node, "name", "<lambda>"),
                    "kind": type(node).__name__,
                    "start": start,
                    "end": end,
                    "lines": end - start + 1,
                }
            )
    return functions, []


def python_span(node, tokens: list) -> tuple[int, int]:
    start, end = node.lineno, node.end_lineno
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        expression_start = decorators[0].lineno
        start = max(
            token.start[0]
            for token in tokens
            if token.type == tokenize.OP
            and token.string == "@"
            and token.start[1] == node.col_offset
            and token.start[0] <= expression_start
        )
    if not isinstance(node, ast.Lambda):
        for token in tokens:
            if token.start[0] <= node.end_lineno or token.type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                continue
            if token.type != tokenize.COMMENT or token.start[1] <= node.col_offset:
                break
            end = token.end[0]
    return start, end


def inspect_source(source: bytes, language: str, parser=None) -> tuple[list, list]:
    if language == "python":
        return inspect_python(source)
    parser = parser or parser_for(language)
    if language in {"cpp", "cuda"}:
        from scripts.cuda_syntax import parser_source

        source = parser_source(source, parser)
    syntax = parser.parse(source)
    functions, errors = tree_spans(syntax, language)
    if language in {"cpp", "cuda"}:
        macro_functions, macro_errors = macro_spans(syntax, parser, language)
        functions.extend(macro_functions)
        errors.extend(macro_errors)
    return functions, errors


def tree_spans(syntax, language: str, offset: int = 0) -> tuple[list, list]:
    functions, errors = [], []
    for node in walk(syntax.root_node):
        if node.is_error or node.is_missing:
            errors.append(
                {
                    "line": node.start_point.row + 1 + offset,
                    "column": node.start_point.column + 1,
                    "message": "syntax "
                    + ("missing " if node.is_missing else "")
                    + node.type,
                }
            )
        if node.type in FUNCTIONS[language]:
            start = function_start(node) + offset
            end = node.end_point.row + int(node.end_point.column > 0) + offset
            functions.append(
                {
                    "name": function_name(node),
                    "kind": node.type,
                    "start": start,
                    "end": end,
                    "lines": end - start + 1,
                }
            )
    return functions, errors


def macro_spans(syntax, parser, language: str) -> tuple[list, list]:
    from scripts.cuda_syntax import parser_source, tokens

    functions, errors = [], []
    for node in walk(syntax.root_node):
        if node.type not in {"preproc_def", "preproc_function_def"}:
            continue
        value = node.child_by_field_name("value")
        if value is None:
            continue
        body = value.text.replace(b"\\\r\n", b" \r\n").replace(b"\\\n", b" \n")
        parsed = parser.parse(body)
        if not any(token.text == b"{" for token in tokens(parsed.root_node)):
            continue
        body = parser_source(body, parser)
        parsed = parser.parse(body)
        offset = value.start_point.row
        if parsed.root_node.has_error:
            parsed = parser.parse(b"void __code_size_macro__(){\n" + body + b";\n}")
            offset -= 1
        found, failed = tree_spans(parsed, language, offset)
        functions.extend(
            entry for entry in found if entry["name"] != "__code_size_macro__"
        )
        errors.extend(failed)
    return functions, errors


def external_spans(root: Path, language: str, paths: list[Path]) -> list:
    tool_root = Path(__file__).resolve().parents[1]
    if language == "rust":
        arguments = [
            "cargo",
            "run",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(tool_root / "development/style-check/Cargo.toml"),
            "--",
            "--function-spans",
        ]
    else:
        arguments = [
            "node",
            str(tool_root / "development/javascript/check-spacing.mjs"),
            "--function-spans",
        ]
    result = subprocess.run(
        [*arguments, *(str(root / path) for path in paths)],
        cwd=tool_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{language} parser failed")
    return json.loads(result.stdout)
