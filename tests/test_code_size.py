import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.check_code_size import audit, main, physical_lines, size_violations
from scripts.code_size_syntax import external_spans, inspect_source, parser_for
from scripts.cuda_syntax import parser_source
from scripts.format_cuda import SOURCE_SUFFIXES
from scripts.repo_files import ROOT, owned_files


class CodeSize(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_file_limit_counts_blank_and_comment_lines_without_final_newline(self):
        for count in (500, 501):
            with self.subTest(count=count):
                text = "# comment\n\n" * (count // 2) + ("# tail" if count % 2 else "")
                self.write("boundary.py", text)
                report = audit(self.root)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["files"][0]["lines"], count)
                self.assertEqual(bool(report["violations"]), count == 501)
        self.assertEqual(physical_lines(b""), 0)
        self.assertEqual(physical_lines(b"one\r\ntwo"), 2)
        for count in (500, 501):
            self.write("boundary.py", "# comment\r\n" * count)
            report = audit(self.root)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["files"][0]["lines"], count)
            self.assertEqual(bool(report["violations"]), count == 501)
        self.write("boundary.py", "# comment\r" * 501)
        self.assertIn("bare CR", audit(self.root)["errors"][0]["message"])
        config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
        native_hook = next(
            hook for hook in config["repos"][0]["hooks"] if hook["id"] == "cuda-format"
        )
        for suffix in SOURCE_SUFFIXES:
            self.assertRegex("source" + suffix, native_hook["files"])

    def test_python_decorators_comments_nested_functions_and_lambda_spans(self):
        source = (
            b"@(\n    decorate\n)\nasync def outer():\n"
            b"    def inner():\n        return 1\n        # trailing inner comment\n"
            b"\n    return lambda: 2\n    # trailing outer comment\n"
        )
        functions, errors = inspect_source(source, "python")
        self.assertEqual(errors, [])
        spans = {
            function["name"]: (function["start"], function["end"])
            for function in functions
        }
        self.assertEqual(spans, {"outer": (1, 10), "inner": (5, 7), "<lambda>": (9, 9)})
        for count in (80, 81):
            source = (
                b"@decorate\nasync def boundary():\n"
                + b"    # comment\n" * (count - 3)
                + b"    pass\n"
            )
            functions, errors = inspect_source(source, "python")
            self.assertEqual(errors, [])
            record = {"path": "boundary.py", "lines": count, "functions": functions}
            self.assertEqual(bool(size_violations(record)), count == 81)

    def test_cuda_templates_attributes_lambda_and_opaque_text(self):
        source = b"""// __launch_bounds__(9) void fake() {}
const char *text = R"tag(CUTE_DEVICE __align__(99) #pragma unroll)tag";
template<int N>
CUTE_HOST_DEVICE
int outer() {
  auto closure = []() {
    return 1;
  };
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) {}
  return closure();
}
__launch_bounds__(128, 2, 1)
__global__ void kernel() {
  __shared__ __align__(16) char memory[16];
}
"""
        parser = parser_for("cuda")
        normalized = parser_source(source, parser)
        self.assertEqual(len(normalized), len(source))
        self.assertEqual(
            [i for i, byte in enumerate(source) if byte == 10],
            [i for i, byte in enumerate(normalized) if byte == 10],
        )
        self.assertIn(
            b'R"tag(CUTE_DEVICE __align__(99) #pragma unroll)tag"', normalized
        )
        functions, errors = inspect_source(source, "cuda", parser)
        self.assertEqual(errors, [])
        self.assertEqual(
            [
                (function["name"], function["start"], function["end"])
                for function in functions
            ],
            [("outer", 3, 14), ("<lambda_expression@6>", 6, 8), ("kernel", 15, 18)],
        )
        cpp = (
            b"struct Example {\n  [[nodiscard]]\n  int method(){ return 1; }\n};\n"
            b"CUTE_HOST_DEVICE\nint plain(){ return 1; }\n"
        )
        functions, errors = inspect_source(cpp, "cpp")
        self.assertEqual(errors, [])
        self.assertEqual(
            [(item["start"], item["end"]) for item in functions], [(2, 3), (5, 6)]
        )

    def test_cuda_81_line_annotation_function_and_macro_closure_fail(self):
        source = b"CUTE_HOST_DEVICE\nvoid boundary() {\n" + b"\n" * 78 + b"}\n"
        functions, errors = inspect_source(source, "cuda")
        self.assertEqual(errors, [])
        self.assertEqual(functions[0]["lines"], 81)
        source = (
            b"#define MAKE_FN \\\nint generated() { \\\n"
            b"  return []() { return 1; }(); \\\n}\n"
        )
        functions, errors = inspect_source(source, "cuda")
        self.assertEqual(errors, [])
        self.assertEqual(
            [(function["kind"], function["lines"]) for function in functions],
            [("function_definition", 3), ("lambda_expression", 1)],
        )

    def test_cuda_typename_adapter_is_narrow_and_syntax_failures_survive(self):
        valid = b"template<typename T> void f(){ consume(typename T::Value{}); }"
        functions, errors = inspect_source(valid, "cuda")
        self.assertEqual(errors, [])
        self.assertEqual([function["name"] for function in functions], ["f"])
        for source in (
            b"template<typename> void f( { }",
            b"void f(){ consume(typename T::Value{); }",
            b"UNKNOWN_ANNOTATION void f( { }",
            b"__global__ void f(){ int x = ; }",
            b"__launch_bounds__(,) __global__ void f() {}",
            b'TVM_FFI_DECLARE_OBJECT_INFO_FINAL("Name", Bad??, Object);',
            b'void f(){ const char *s = "typename T::Value{}"; broken( ; }',
        ):
            with self.subTest(source=source):
                self.assertTrue(inspect_source(source, "cuda")[1])

    def rust_report(self, source):
        self.write("sample.rs", source)
        return external_spans(self.root, "rust", [Path("sample.rs")])[0]

    def test_rust_attributes_raw_identifier_and_macro_closure(self):
        result = self.rust_report(
            "#[inline]\nfn outer(raw: &str) {\n    let _ = &raw;\n"
            "    custom!({ let inner = || {\n        1\n    }; });\n}\n"
        )
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            [(f["name"], f["start"], f["end"]) for f in result["functions"]],
            [("outer", 1, 7), ("<closure>", 4, 6)],
        )
        for declaration in ("fn $x() {", "fn fixed() { $x;", "$(fn $x() {"):
            repeated = declaration.startswith("$(")
            result = self.rust_report(
                "macro_rules! item { ($x:ident) => {\n"
                + declaration
                + "\n"
                + "\n" * 80
                + ("} )*\n" if repeated else "}\n")
                + "}; }\n"
            )
            self.assertEqual(result["errors"], [])
            self.assertEqual([f["lines"] for f in result["functions"]], [82])
        for header, footer, expected in (
            ("wrap!(tag => fn large(){\n", "});\n", 82),
            (
                "macro_rules! outer { () => { wrap!(tag => #[inline]\n"
                "async unsafe fn large(){\n",
                "}); }; }\nouter!();\n",
                83,
            ),
        ):
            result = self.rust_report(
                "macro_rules! wrap { ($tag:ident => $item:item) => { $item }; }\n"
                + header
                + "\n" * 80
                + footer
            )
            self.assertEqual(result["errors"], [])
            self.assertEqual([f["lines"] for f in result["functions"]], [expected])
        for header, footer in (
            (
                "macro_rules! item { ($($x:expr),*) => { $(let f = || {\n",
                "$x\n};)* }; }\n",
            ),
            (
                "macro_rules! closure { ($tag:ident => $c:expr) => { $c }; }\n"
                "fn host(){ let _ = closure!(tag => || {\n",
                "1\n});}\n",
            ),
        ):
            result = self.rust_report(header + "\n" * 80 + footer)
            self.assertEqual(result["errors"], [])
            self.assertTrue(
                any(
                    f["kind"] == "closure" and f["lines"] > 80
                    for f in result["functions"]
                )
            )
        for source in (
            "fn broken( {\n",
            "macro_rules! item { ($x:ident) => { fn $x(){let = ;} }; }",
            "choose!(tag => fn broken(){let = ;});",
        ):
            self.assertTrue(self.rust_report(source)["errors"])
        result = self.rust_report(
            'macro_rules! item { (fn $x:ident) => { "fn fake() {}" }; }'
        )
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["functions"], [])
        self.assert_rust_pattern_controls()

    def assert_rust_pattern_controls(self):
        cases = (
            ("fn f(){ assert_eq!(3, 1 | 2); }", ["f"]),
            ("const FLAG: bool = matches!(0,\n0 | 1\n" + "\n" * 80 + "| 2\n);\n", []),
        )
        for source, names in cases:
            result = self.rust_report(source)
            self.assertEqual(result["errors"], [])
            self.assertEqual(
                [function["name"] for function in result["functions"]], names
            )

    def test_javascript_methods_arrows_and_typescript_keep_physical_spans(self):
        self.write(
            "sample.mjs",
            "class Example {\n  static\n  method() {\n    return () => {\n"
            "      return 1;\n    };\n  }\n}\n",
        )
        result = external_spans(self.root, "javascript", [Path("sample.mjs")])[0]
        self.assertEqual(result["errors"], [])
        self.assertEqual(
            [(function["start"], function["end"]) for function in result["functions"]],
            [(2, 7), (4, 6)],
        )
        self.write(
            "sample.ts",
            "function typed(\n  value: number\n): number {\n  return value;\n}\n",
        )
        result = external_spans(self.root, "typescript", [Path("sample.ts")])[0]
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["functions"][0]["lines"], 5)
        self.write("sample.mjs", "function broken( {\n")
        self.assertTrue(
            external_spans(self.root, "javascript", [Path("sample.mjs")])[0]["errors"]
        )

    def test_shell_heredoc_and_shebang_are_code_and_errors_fail(self):
        source = b"#!/bin/bash\nwork() {\n  cat <<'DATA'\nfunction fake() {\nDATA\n}\n"
        self.write("runner", source.decode())
        report = audit(self.root)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["files"][0]["language"], "bash")
        self.assertEqual(
            [
                (function["name"], function["lines"])
                for function in report["files"][0]["functions"]
            ],
            [("work", 5)],
        )
        self.write("runner", "#!/bin/bash\nwork() {\n")
        self.assertTrue(audit(self.root)["errors"])

    def test_owned_files_exclude_ignored_generated_third_party_and_symlinks(self):
        self.write(".gitignore", "ignored.py\n")
        self.write(".gitattributes", "generated.py linguist-generated\n")
        for name in (
            "owned.py",
            ".claude/helper.py",
            "ignored.py",
            "generated.py",
            "3rdparty/dependency.py",
        ):
            self.write(name, "pass\n")
        subprocess.run(
            ["git", "-C", str(self.root), "add", "-f", "ignored.py"], check=True
        )
        (self.root / "alias.py").symlink_to(self.root / "owned.py")
        (self.root / "alias_directory").symlink_to(
            self.root / ".claude", target_is_directory=True
        )
        names = {path.as_posix() for path in owned_files(self.root)}
        self.assertEqual(
            names, {".gitignore", ".gitattributes", "owned.py", ".claude/helper.py"}
        )

    def test_cli_inventory_and_missing_parser_fail_closed(self):
        self.write("oversized.py", "# comment\n" * 501)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                main(["--root", str(self.root), "--inventory", "--json"]), 0
            )
        self.assertEqual(json.loads(output.getvalue())["violations"][0]["lines"], 501)
        self.write("oversized.py", "def invalid(\n")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                main(["--root", str(self.root), "--inventory", "--json"]), 1
            )
        self.assertTrue(json.loads(output.getvalue())["errors"])
        with (
            patch(
                "scripts.code_size_syntax.version",
                side_effect=PackageNotFoundError("tree-sitter"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(main(["--root", str(self.root), "--json"]), 1)
        self.assertIn(
            "tree-sitter", json.loads(output.getvalue())["errors"][0]["message"]
        )
        self.write("unsupported.rb", "def function\nend\n")
        self.assertTrue(
            any(
                "unclassified owned file" in error["message"]
                for error in audit(self.root)["errors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
