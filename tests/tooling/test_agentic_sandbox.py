"""Disposable image and filesystem cleanup under failure and changed permissions."""

import json
from types import SimpleNamespace

import pytest

from benchmarks import agentic_sandbox as sandbox


def test_readonly_agent_directories_are_removed_without_following_links(
    tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    directory = tmp_path / "client"
    directory.mkdir()
    agent = sandbox.AgentSandbox(SimpleNamespace(owner="owned"), directory)
    nested = agent.root / "workspace/private"
    nested.mkdir()
    (nested / "change.txt").write_text("temporary")
    (nested / "link").symlink_to(outside, target_is_directory=True)
    nested.chmod(0)
    monkeypatch.setattr(sandbox, "docker", lambda *args, **kwargs: "")
    agent.close()
    assert not agent.root.exists()
    assert (outside / "keep.txt").read_text() == "keep"


def test_archive_failure_still_deletes_sandbox(tmp_path, monkeypatch):
    directory = tmp_path / "client"
    directory.mkdir()
    agent = sandbox.AgentSandbox(SimpleNamespace(owner="owned"), directory)
    monkeypatch.setattr(sandbox, "docker", lambda *args, **kwargs: "")

    def fail():
        raise OSError("injected archive failure")

    monkeypatch.setattr(agent, "preserve_logs", fail)
    with pytest.raises(ValueError, match="archive failure"):
        agent.close()
    assert not agent.root.exists()
    assert (
        json.loads((directory / "sandbox-cleanup.json").read_text())["closed"] is False
    )


def test_image_removal_accepts_automatic_parent_layer_deletion(tmp_path, monkeypatch):
    image = sandbox.SandboxImage.__new__(sandbox.SandboxImage)
    image.owner, image.directory = "owned", tmp_path
    images = ["child", "parent"]
    calls = []

    def docker(*arguments, **kwargs):
        calls.append(arguments)
        if arguments[:2] == ("container", "ls"):
            return ""
        if arguments[:2] == ("image", "ls"):
            return "\n".join(images)
        if arguments[:2] == ("image", "inspect"):
            assert arguments[2] in images
            return json.dumps([{"Config": {"Labels": {sandbox.LABEL: "owned"}}}])
        if arguments[:2] == ("image", "rm"):
            images.clear()  # Docker also removes the now-unreferenced parent.
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(sandbox, "docker", docker)
    image.close()
    assert not images
    assert not any(arguments == ("image", "inspect", "parent") for arguments in calls)
    assert json.loads((tmp_path / "cleanup.json").read_text())["closed"]
