"""Development-only TileFoundry analysis and Nsight CSV integration."""

import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def analyze(symbol):
    """Return explicitly estimated HIR metrics, not native-kernel measurements."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="oh-my-vllm-foundry-") as directory:
        output = Path(directory) / "analysis.txt"
        command = [
            str(Path(sys.executable).parent / "tilefoundry"),
            "analyze",
            f"development/kernels/semantics.py:Operators.{symbol}",
            str(output),
            "--compute-cost",
            "--memory",
            "--roofline",
        ]
        environment = {**os.environ}
        environment["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        subprocess.run(
            command, cwd=ROOT, env=environment, check=True, capture_output=True
        )
        return dict(
            kind="estimated",
            source="TileFoundry HIR",
            symbol=symbol,
            report=output.read_text(),
            limitation="Representative HIR shape; not native layout or FP64 phase cost",
        )


def read_nsight(path):
    """Import separate profiler observations without using its latency as a gate."""
    lines = Path(path).read_text().splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header is None:
        raise ValueError(
            "Nsight CSV metrics missing; check profiler errors/permissions"
        )
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    required = {"Kernel Name", "Metric Name", "Metric Unit", "Metric Value"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("incomplete Nsight metric report")
    return dict(
        kind="measured", source="Nsight Compute (separate replay)", metrics=rows
    )
