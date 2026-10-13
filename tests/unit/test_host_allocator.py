"""Host diagnostics report glibc storage without a leak or full-RSS claim."""

import ctypes
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from oh_my_vllm.worker import host_allocator as host


def test_allocator_view_keeps_free_storage_separate_from_used_and_mmap(monkeypatch):
    values = host.Mallinfo2(arena=1000, uordblks=400, fordblks=600, hblkhd=500)
    monkeypatch.setattr(
        host,
        "allocator",
        lambda: (SimpleNamespace(mallinfo2=lambda: values), "fixture"),
    )
    result = host.host_allocator_snapshot()
    assert result["scope"] == "glibc_malloc_only"
    assert result["used_or_retained_bytes"] == 900
    assert result["fields"]["fordblks"] == 600
    assert result["fields"]["arena"] == 1000
    assert "live_objects" not in result and "rss_bytes" not in result
    assert result["elapsed_seconds"] >= 0


def test_unsupported_allocator_reports_absence_without_zero_measurements(monkeypatch):
    monkeypatch.setattr(host, "allocator", lambda: (None, None))
    result = host.host_allocator_snapshot()
    assert result["available"] is False
    assert "fields" not in result and "used_or_retained_bytes" not in result


def test_current_libc_snapshot_and_header_abi_agree(tmp_path):
    result = host.host_allocator_snapshot()
    if not result["available"]:
        pytest.skip("Current host does not supply glibc mallinfo2")
    compiler = shutil.which("cc")
    assert compiler is not None
    statements = ['printf("[%zu", sizeof(struct mallinfo2));']
    statements.extend(
        f'printf(",%zu", offsetof(struct mallinfo2, {name}));' for name in host.FIELDS
    )
    source = tmp_path / "abi.c"
    source.write_text(
        "#include <stddef.h>\n#include <malloc.h>\n#include <stdio.h>\n"
        + "int main(void) {"
        + "".join(statements)
        + 'puts("]"); return 0;}\n'
    )
    binary = tmp_path / "abi"
    subprocess.run(
        [compiler, str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    measured = json.loads(subprocess.check_output([str(binary)], text=True, timeout=5))
    expected = [ctypes.sizeof(host.Mallinfo2)] + [
        getattr(host.Mallinfo2, name).offset for name in host.FIELDS
    ]
    assert measured == expected
    assert set(result["fields"]) == set(host.FIELDS)
    assert all(type(value) is int and value >= 0 for value in result["fields"].values())
