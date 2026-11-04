"""Content identity for the service, model runtime, and measurement tools."""

import hashlib
import subprocess
from pathlib import Path


def source_identity():
    root = Path(__file__).resolve().parents[2]
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = root / name
        if path.is_file() and (
            name.startswith(("python/", "crates/", "development/serving/"))
            or name
            in {
                "Cargo.toml",
                "Cargo.lock",
                "requirements/development.txt",
                "requirements/runtime.txt",
                "development/native_build.py",
            }
        ):
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
