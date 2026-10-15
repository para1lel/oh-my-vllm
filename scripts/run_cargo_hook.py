import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "fmt": ["fmt", "--all", "--check"],
    "clippy": ["clippy", "--workspace", "--all-targets", "--all-features"],
    "test": ["test", "--workspace"],
}


def main() -> int:
    command = sys.argv[1]
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    manifests = sorted(
        {
            ROOT / name
            for name in names
            if name
            and Path(name).name == "Cargo.toml"
            and not name.startswith("3rdparty/")
            and (ROOT / name).is_file()
            and not (ROOT / name).is_symlink()
            and (ROOT / name).resolve().is_relative_to(ROOT)
        },
        key=lambda path: len(path.parts),
    )
    covered = set()
    failed = False
    for manifest in manifests:
        if manifest.parent in covered:
            continue
        metadata = subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(manifest),
            ],
            cwd=ROOT,
        )
        workspace = json.loads(metadata)
        members = set(workspace["workspace_members"])
        covered.update(
            Path(package["manifest_path"]).parent
            for package in workspace["packages"]
            if package["id"] in members
        )
        arguments = ["cargo", *COMMANDS[command], "--manifest-path", str(manifest)]
        if command != "fmt":
            arguments.append("--locked")
        if command == "clippy":
            arguments.extend(["--", "-D", "warnings"])
        failed |= subprocess.run(arguments, cwd=ROOT, check=False).returncode != 0
    if not manifests:
        print("No owned Cargo manifests")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
