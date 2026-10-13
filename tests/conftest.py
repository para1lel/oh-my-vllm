"""The real OMP compatibility fixtures share a disposable rootless image."""

import os
import shutil

import pytest

from benchmarks.agentic_sandbox import SandboxImage


@pytest.fixture(scope="session")
def agentic_sandbox_image(tmp_path_factory):
    if not shutil.which("docker") or not shutil.which("omp"):
        pytest.skip("rootless Docker and pinned OMP are required")
    gateway = os.environ.get("OH_MY_VLLM_SANDBOX_HOST")
    if not gateway:
        pytest.skip("set OH_MY_VLLM_SANDBOX_HOST for the rootless host gateway")
    image = SandboxImage(
        "omp",
        tmp_path_factory.mktemp("sandbox-image") / "runtime",
        gateway,
    )
    try:
        yield image
    finally:
        image.close()
