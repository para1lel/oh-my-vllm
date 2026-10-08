"""Keep the key model operator inventory tied to registered IR semantics."""

from pathlib import Path

from oh_my_vllm.ir import attention as ir_attention  # noqa: F401
from oh_my_vllm.ir import dspark as ir_dspark  # noqa: F401
from oh_my_vllm.ir import operations
from oh_my_vllm.ir.coverage import SITES, check_coverage
from oh_my_vllm.models import qwen  # noqa: F401 - registers model operators


def test_model_operator_sites_are_covered():
    root = Path(__file__).resolve().parents[1] / "python/oh_my_vllm"
    check_coverage(root, set(operations()))
    assert len(SITES) >= 18


def test_coverage_rejects_direct_backend_call_through_allowed_type_import(tmp_path):
    root = Path(__file__).resolve().parents[1] / "python/oh_my_vllm"
    for file in {site.file for site in SITES} | {
        "worker/model_runner.py",
        "worker/mtp.py",
        "worker/dspark.py",
        "worker/dspark_graph.py",
    }:
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((root / file).read_text())
    qwen_path = tmp_path / "models/qwen.py"
    qwen_path.write_text(
        qwen_path.read_text() + "\ndef rogue(x):\n    return attention.append(x)\n"
    )
    import pytest

    with pytest.raises(AssertionError, match="calls a backend through attention"):
        check_coverage(tmp_path, set(operations()))
