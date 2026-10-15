"""The real OMP compatibility fixtures share a disposable rootless image."""

import shutil

import pytest

from benchmarks.agentic_sandbox import SandboxImage


@pytest.fixture(scope="session")
def agentic_sandbox_image(tmp_path_factory):
    if not shutil.which("docker") or not shutil.which("omp"):
        pytest.skip("rootless Docker and pinned OMP are required")
    image = SandboxImage(
        "omp",
        tmp_path_factory.mktemp("sandbox-image") / "runtime",
    )
    try:
        yield image
    finally:
        image.close()
