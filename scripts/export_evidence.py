"""Export historical measurements without deployment-specific identities.

The output is for reference and offline analysis. Formal comparison requires
unaltered raw artifacts, including their complete hardware and run identities.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

KIND = "portable-derived-evidence"
# Raw logs and executable arguments can contain arbitrary host data.
OMIT = {
    "stdout",
    "stderr",
    "command",
    "commands",
    "cpu_affinity",
    "gpu",
    "gpu_uuid",
    "uuid",
    "log",
    "logs",
    "log_path",
    "raw_path",
    "raw_log",
    "raw_json",
    "python",
    "checkout",
    "hostname",
    "host",
    "pid",
    "pids",
    "nvcc_path",
    "so_path",
    "flashinfer_workspace",
    "triton_cache",
    "tilelang_cache",
    "native_cuda_cache",
    "cache_roots",
    "cache_paths",
    "last_lines",
    "base_url",
}
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9:/])/(?:[^\s,;\"'<>|)\]}]+)")
GPU_UUID = re.compile(r"GPU-[0-9a-f]{4,}(?:-[0-9a-f]+)*(?:\.{3}|…)?", re.I)
IP = re.compile(
    r"(?<![\d.])(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
    r"(?:\.\d{1,3}){2,3}(?![\d.])"
)


def portable(value, removed, location="data"):
    """Preserve measurements and relative source names; remove host identities."""
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            field = f"{location}.{key}"
            if key in OMIT or key.endswith(
                ("_command", "_commands", "_stdout", "_stderr")
            ):
                removed.append(field)
                if key == "cpu_affinity" and isinstance(item, list):
                    if "cpu_affinity_count" in value:
                        raise ValueError(
                            "derived cpu_affinity_count collides with input"
                        )
                    output["cpu_affinity_count"] = len(item)
                continue
            clean_key = portable(key, removed, field)
            if clean_key != key:
                clean_key += "#" + hashlib.sha256(key.encode()).hexdigest()[:12]
            if clean_key in output:
                raise ValueError(f"redacted key collision at {location}")
            output[clean_key] = portable(item, removed, field)
        return output
    if isinstance(value, list):
        return [
            portable(item, removed, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        clean = ABSOLUTE_PATH.sub("<local-path>", value)
        clean = GPU_UUID.sub("<gpu-identity>", clean)
        # Package versions are data, including nvidia-curand 10.4.0.35.
        if ".packages." not in location:
            clean = IP.sub("<host-address>", clean)
        if clean != value:
            removed.append(location)
        return clean
    return value


def derive(raw: bytes, name: str):
    removed = []
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "artifact_kind" in parsed:
        raise ValueError("input is already derived evidence")
    data = portable(parsed, removed)
    return {
        "artifact_kind": KIND,
        "schema": 1,
        "use": "historical reference and offline analysis; not formal comparison",
        "original": {
            "name": Path(name).name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "removed_fields": sorted(
            {portable(field, [], "removed_fields") for field in removed}
        ),
        "data": data,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or (
        args.output.exists() and args.input.samefile(args.output)
    ):
        parser.error("keep the original input unchanged")
    summary = derive(args.input.read_bytes(), args.input.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
