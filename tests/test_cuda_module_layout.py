"""The native backend's size check covers source and template headers."""

from scripts.format_cuda import oversized_files


def test_cuda_module_limit_counts_headers_and_python(tmp_path):
    for name in ("kernel.cu", "nested/template.cuh", "binding.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("line\n" * 801)
    (tmp_path / "small.cuh").write_text("line\n" * 800)
    (tmp_path / "not-code.txt").write_text("line\n" * 900)
    assert [
        (path.relative_to(tmp_path).as_posix(), lines)
        for path, lines in oversized_files(tmp_path)
    ] == [
        ("binding.py", 801),
        ("kernel.cu", 801),
        ("nested/template.cuh", 801),
    ]
