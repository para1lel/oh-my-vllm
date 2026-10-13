"""Bind observed HTTP and Python workers to committed inference sources."""

import hashlib
import json
from pathlib import Path

from oh_my_vllm.performance.coverage import runtime_contract

from benchmarks.evidence import observed_configuration


def server_identity(diagnostics, log, source):
    """Bind the live process and its startup source record, not only a local build."""
    configured = observed_configuration(log)
    if configured.get("rust_sources_sha256") != source["rust_sources_sha256"]:
        raise ValueError("running Rust build stamp differs")
    pid = diagnostics["server_pid"]
    stat_path = Path("/proc", str(pid), "stat")
    before = int(stat_path.read_text().rsplit(") ", 1)[1].split()[19])
    binary = hashlib.sha256(Path("/proc", str(pid), "exe").read_bytes()).hexdigest()
    after = int(stat_path.read_text().rsplit(") ", 1)[1].split()[19])
    if before != after or binary != source["binary_sha256"]:
        raise ValueError("live server process or executable differs")
    runtimes = []
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("message") == "Independent runtime":
            runtimes.append(row)
    if (
        len(runtimes) != 1
        or runtimes[0].get("sources_sha256") != runtime_contract()["sources"]
    ):
        raise ValueError("worker startup source identities differ")
    worker_pid = runtimes[0]["pid"]
    if diagnostics.get("worker_pid") != worker_pid:
        raise ValueError("worker diagnostics and startup identities differ")
    worker_stat = (
        Path("/proc", str(worker_pid), "stat").read_text().rsplit(") ", 1)[1].split()
    )
    if int(worker_stat[1]) != pid:
        raise ValueError("worker does not belong to the live server")
    return {
        "pid": pid,
        "start_ticks": before,
        "binary_sha256": binary,
        "worker_pid": worker_pid,
        "worker_start_ticks": int(worker_stat[19]),
        "worker_sources": runtimes[0]["sources_sha256"],
        "configuration": configured,
    }
