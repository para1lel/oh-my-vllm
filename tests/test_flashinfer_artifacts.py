"""The pinned companion supplies checked B200 artifacts without a download."""

import hashlib
import importlib.metadata
from pathlib import Path

import pytest
import torch


@pytest.fixture
def offline_loader(monkeypatch):
    # Import in the test body: GPU collection must not initialize a context.
    initialized = torch.cuda.is_initialized()
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    monkeypatch.delenv("FLASHINFER_CUBIN_CHECKSUM_DISABLED", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")
    import flashinfer_cubin
    from flashinfer.jit import cubin_loader, env

    assert (
        importlib.metadata.version("flashinfer-python") == flashinfer_cubin.__version__
    )
    assert env._get_cubin_dir() == Path(flashinfer_cubin.get_cubin_dir())
    monkeypatch.setattr(cubin_loader, "FLASHINFER_CUBIN_DIR", env._get_cubin_dir())

    def forbidden_download(*args, **kwargs):
        pytest.fail("bundled artifacts must not use the downloader")

    monkeypatch.setattr(cubin_loader, "download_file", forbidden_download)
    yield cubin_loader
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("family", ["TRTLLM_GEN_GEMM", "TRTLLM_GEN_FMHA"])
def test_bundled_b200_manifest_headers_and_kernel_load_offline(offline_loader, family):
    from flashinfer.artifacts import ArtifactPath, CheckSumHash

    prefix = getattr(ArtifactPath, family).rstrip("/")
    digest = getattr(CheckSumHash, family)
    manifest = offline_loader.get_artifact(f"{prefix}/checksums.txt", digest)
    assert hashlib.sha256(manifest).hexdigest() == digest
    entries = [
        line.split(maxsplit=1) for line in manifest.decode().splitlines() if line
    ]
    headers = [(sha, name) for sha, name in entries if name.endswith(".h")]
    kernels = [
        (sha, name)
        for sha, name in entries
        if name.endswith(".cubin") and "sm100" in name.lower()
    ]
    assert headers and kernels
    for sha, name in [*headers, kernels[0]]:
        data = offline_loader.get_artifact(f"{prefix}/{name}", sha)
        assert data and hashlib.sha256(data).hexdigest() == sha


def test_corrupt_artifact_is_rejected_without_download(
    offline_loader, monkeypatch, tmp_path
):
    monkeypatch.setattr(offline_loader, "FLASHINFER_CUBIN_DIR", tmp_path)
    (tmp_path / "kernel.cubin").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Artifact not found locally"):
        offline_loader.get_artifact(
            "kernel.cubin", hashlib.sha256(b"valid").hexdigest()
        )


def test_companion_version_mismatch_is_rejected(offline_loader, monkeypatch):
    import flashinfer_cubin
    from flashinfer.jit import env

    monkeypatch.setattr(flashinfer_cubin, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="does not match"):
        env._get_cubin_dir()
