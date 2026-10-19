import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = Path(__file__).with_name("headers.json")
MARKER = "oh-my-vllm-public-baselines.json"


def outside_repository(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError(
            "the baseline environment and cache must be outside the repository"
        )
    return path


def install_headers(cache):
    destination = cache / "headers"
    destination.mkdir(parents=True, exist_ok=True)
    for name, pin in json.loads(PINS.read_text()).items():
        target = destination / name
        marker = target / MARKER
        if marker.exists() and json.loads(marker.read_text()) == pin:
            continue
        if target.exists():
            raise ValueError(f"unrecognized header directory: {target}")
        url = (
            f"https://codeload.github.com/{pin['repository']}/tar.gz/{pin['revision']}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != pin["sha256"]:
            raise ValueError(f"source archive digest differs: {name}")
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            archive = Path(temporary) / "source.tar.gz"
            archive.write_bytes(data)
            unpacked = Path(temporary) / "unpacked"
            with tarfile.open(archive) as source:
                source.extractall(unpacked, filter="data")
            entries = list(unpacked.iterdir())
            if len(entries) != 1 or not entries[0].is_dir():
                raise ValueError("source archive has an unexpected root")
            shutil.move(str(entries[0]), target)
        marker.write_text(json.dumps(pin, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=outside_repository, required=True)
    parser.add_argument("--cache", type=outside_repository, required=True)
    parser.add_argument("--python", required=True)
    arguments = parser.parse_args()
    prefix = arguments.environment
    marker = prefix / MARKER
    if prefix.exists() and not marker.exists():
        raise ValueError("existing environment has no baseline ownership marker")
    if not prefix.exists():
        subprocess.run(
            ["uv", "venv", "--python", arguments.python, str(prefix)],
            check=True,
        )
        marker.write_text(json.dumps({"purpose": "public attention baselines"}) + "\n")
    environment = {
        **os.environ,
        "FLASHINFER_BUILD_NO_PIP": "1",
        "BUILD_NVEP": "0",
    }
    subprocess.run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(prefix / "bin/python"),
            "--require-hashes",
            str(ROOT / "requirements/baselines.txt"),
        ],
        env=environment,
        check=True,
    )
    install_headers(arguments.cache)


if __name__ == "__main__":
    main()
