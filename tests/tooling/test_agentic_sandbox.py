"""Disposable image and filesystem cleanup under failure and changed permissions."""

import json
from types import SimpleNamespace

import pytest

from benchmarks import agentic_sandbox as sandbox
from benchmarks.agentic_lane_transport import LaneBridge


def sealed_lane(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "docker", lambda *args, **kwargs: "")
    directory = tmp_path / "lane"
    directory.mkdir()
    agent = sandbox.AgentSandbox(
        SimpleNamespace(owner="owned", identity={"Id": "image-id"}), directory
    )
    bridge = LaneBridge(
        directory / "transport",
        "http://127.0.0.1:8123",
        telemetry_upstream="http://127.0.0.1:8124",
        read_timeout=30,
    )
    return agent, bridge


def environment_of(command):
    values = {}
    for index, item in enumerate(command):
        if item == "--env":
            key, _, value = command[index + 1].partition("=")
            values[key] = value
    return values


def test_sealed_lane_command_mounts_sockets_and_rewrites_endpoints(
    tmp_path, monkeypatch
):
    agent, bridge = sealed_lane(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="attached bridged transport"):
            agent.command(["omp"], {})
        agent.attach_transport(bridge)
        with pytest.raises(ValueError, match="already has a transport"):
            agent.attach_transport(bridge)
        command = agent.command(
            ["omp", "--mode", "rpc"],
            {
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                    "http://127.0.0.1:8123/abc/v1/traces"
                ),
                "OTEL_METRICS_EXPORTER": "none",
            },
        )
        assert command[command.index("--network") + 1] == "none"
        mounts = [item for item in command if item.startswith("type=bind,source=")]
        assert mounts == [
            f"type=bind,source={agent.root},target={agent.root}",
            *bridge.mounts(),
        ]
        values = environment_of(command)
        assert values["CUDA_VISIBLE_DEVICES"] == ""
        assert values["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
            "http://127.0.0.1:18081/abc/v1/traces"
        )
        assert values["OTEL_METRICS_EXPORTER"] == "none"
        assert command[-4:] == ["--", "/usr/local/bin/omp", "--mode", "rpc"]
        assert "/usr/local/bin/lane-forward.py" in command
        assert str(bridge.model_socket) in command
    finally:
        bridge.close()
        agent.close()


def test_lane_transport_must_listen_outside_the_sandbox(tmp_path, monkeypatch):
    agent, bridge = sealed_lane(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="must already listen"):
            agent.attach_transport(SimpleNamespace(directory=tmp_path / "missing"))
        inside = LaneBridge(
            agent.root / "bridge", "http://127.0.0.1:8123", read_timeout=30
        )
        try:
            assert inside.model_socket.is_socket()
            with pytest.raises(ValueError, match="outside the workspace"):
                agent.attach_transport(inside)
        finally:
            inside.close()
    finally:
        bridge.close()
        agent.close()


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
