"""Diagnostic native-goal fixture that builds a database outside inference source.

The accepted soak workload is the corpus translation in
[agentic_translation](agentic_translation.py). This fixture keeps the original
goal evidence reproducible and runs only under `--diagnostic`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

OMP_VERSION = "18.10.3"
FOLLOWUPS = (
    "Audit SQL results against sqlite3 with seeded independent generated queries. "
    "Cover NULLs, joins, duplicates, aggregates and parameter binding. "
    "Fix discrepancies.",
    "Exercise every WAL/commit/checkpoint write boundary with subprocess crashes. "
    "Test truncated and corrupt records, then fix any durability or recovery defect.",
    "Build a deterministic concurrent transaction harness. Check conflicting writers, "
    "repeatable reads, rollback, deadlocks and stale indexes against a serial oracle.",
    "Audit parser precedence and syntax errors with a generated adversarial corpus. "
    "Test nested subqueries, escaped literals, type coercion and atomic failed "
    "statements.",
    "Compare indexed and unindexed plans with an independent operation-sequence "
    "oracle. Exercise index updates, NULLs, duplicate keys and table rebuilds. "
    "Fix discrepancies.",
    "Implement and test schema/storage migration and import/export across restarts. "
    "Preserve transactions and reject unsupported versions without changing "
    "stored data.",
    "Measure scaling for scans, joins, indexes, commits and checkpoints. Investigate "
    "bottlenecks, improve them, and publish reproducible measurements with "
    "limitations.",
    "Review public APIs and CLI end to end. Add failure-injection and resource "
    "lifetime tests for file handles, locks, repeated close, malformed input and "
    "partial startup.",
    "Audit constraints, defaults, foreign keys and savepoint interactions with "
    "independent reference tests. Fix statement atomicity and rollback defects.",
    "Review every remaining TASK.md requirement against code and test evidence. Add "
    "missing behavior and adversarial regression tests, then rerun the complete suite.",
)
TASK = """# Build a durable relational database from an empty project

Implement a usable embedded database in the goaldb/ Python package using only
the standard library. Do not use sqlite3 or another database as the implementation.
sqlite3 may supply an independent oracle in tests. Work only in this directory.
This is an implementation task: write code, run tests, correct failures, and
produce a README with commands, supported behavior, and honest limitations.

Public API: Database(path).execute(sql, parameters=()) returns a list of row
dictionaries for queries. Provide begin(), commit(), rollback(), and close().
The CLI python -m goaldb DATABASE reads SQL from stdin and prints JSON results.

Implement all of these features with one coherent design:
1. A lexer and parser with position-aware errors, quoted strings, identifiers,
   positional parameters, precedence, NULL and three-valued boolean logic.
2. CREATE/DROP TABLE, typed columns, defaults, NOT NULL, primary/unique keys,
   foreign keys, INSERT, UPDATE, DELETE, and atomic statement failure.
3. SELECT expressions, aliases, WHERE, INNER/LEFT joins, GROUP BY, HAVING,
   COUNT/SUM/MIN/MAX/AVG, DISTINCT, ORDER BY, LIMIT/OFFSET, and subqueries.
4. A query plan and EXPLAIN, equality/range indexes, index maintenance, and
   equivalent indexed and unindexed results. Preserve duplicate and NULL semantics.
5. Explicit transactions, savepoints, repeatable snapshots, writer conflict
   detection, isolation between database handles, and rollback after constraint errors.
6. Versioned persistent storage, checksummed write-ahead records, checkpointing,
   restart recovery, truncated-tail recovery, and rejection of corrupt committed data.
7. A bounded lock/deadlock policy, deterministic ordering, concurrent readers,
   multiple writers, and subprocess crash tests at commit/checkpoint boundaries.
8. At least 200 meaningful test scenarios, including independent SQL differential
   tests, seeded generated operation sequences, persistence, concurrency, and faults.
   Parameterized scenarios count, but duplicated trivial assertions do not.
9. A reproducible benchmark with actual measured numbers, documented complexity,
   a migration/version policy, and an end-to-end example that survives restart.

Start by reading this specification and writing a plan in notes/progress.md.
Implement working increments and run their tests. Keep progress notes current
so automatic context compaction can resume the same goal. Use small successful
source reads to inspect your implementation and test failures. Do not substitute
design prose, mock test output, sleeps, busy waits, or repeated plans for real work.
Run python -m unittest discover -s tests -q for validation. Additional permitted
commands are python -m compileall -q goaldb tests and python -m goaldb.benchmark.
Complete the native goal only when the implementation and tests support every
requirement. If something is incomplete, record the gap and continue implementing.
"""


def prepare_workspace(directory: Path, lane: int) -> dict:
    """Seed only the new task specification, with no inference-project snapshot."""
    if any(directory.iterdir()):
        raise ValueError("goal workspace must start empty")
    specification = TASK + f"\nUse deterministic test seed {784 + lane}.\n"
    (directory / "TASK.md").write_text(specification)
    (directory / "notes").mkdir()
    return {
        "kind": "native_goal_database",
        "files": ["TASK.md"],
        "specification_sha256": hashlib.sha256(specification.encode()).hexdigest(),
        "test_seed": 784 + lane,
        "source_root": None,
    }


def objective(lane: int, index: int = 0) -> str:
    if index:
        focus = FOLLOWUPS[(index - 1) % len(FOLLOWUPS)]
        return (
            f"Follow-up goal {index}. Read TASK.md and notes/progress.md, inspect the "
            "existing implementation, and finish missing foundations first. "
            f"{focus} Use deterministic seed {784 + lane + index * 1009}. "
            "Implement and run meaningful tests; preserve the same project "
            "and history. "
            "Record honest results and gaps in notes/progress.md. Do not repeat plans, "
            "pad output, sleep, or invent successful tests. Complete this goal only "
            "when its changes and independent verification are finished."
        )
    return (
        "Read TASK.md and implement its complete durable relational database "
        f"in this new workspace, using test seed {784 + lane}. "
        "Use tools to implement and verify the project, and continue this same "
        "goal across automatic compaction. Save truthful progress in notes/progress.md."
    )


def artifacts(directory: Path) -> dict:
    """Bind generated implementation bytes without treating a paused goal as done."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file() and path.suffix in (".py", ".md", ".json"):
            files[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    products = {
        "implementation": any(
            name.startswith("goaldb/") and name.endswith(".py") for name in files
        ),
        "tests": any(
            name.startswith("tests/") and name.endswith(".py") for name in files
        ),
        "progress_notes": "notes/progress.md" in files,
    }
    return {
        "files_sha256": files,
        "task_complete": False,
        "products_present": products,
        "missing_products": [name for name, present in products.items() if not present],
    }


def observe_artifacts(client) -> dict:
    """Record task files/tests without making their outcome a framework gate."""
    errors, checks = [], []
    before = {}
    try:
        before = artifacts(client.workspace)
    except Exception as error:
        errors.append(f"artifact hashes before checks: {type(error).__name__}: {error}")
    for command in (
        ["python", "-m", "unittest", "discover", "-s", "tests", "-q"],
        ["python", "-m", "compileall", "-q", "goaldb", "tests"],
    ):
        try:
            checks.append(client.sandbox.execute(command))
        except Exception as error:
            checks.append({"command": command, "returncode": None, "error": str(error)})
    after = {}
    try:
        after = artifacts(client.workspace)
    except Exception as error:
        errors.append(f"artifact hashes after checks: {type(error).__name__}: {error}")
    specification = client.workspace / "TASK.md"
    unchanged = specification.is_file() and hashlib.sha256(
        specification.read_bytes()
    ).hexdigest() == getattr(client, "workspace_manifest", {}).get(
        "specification_sha256"
    )
    return dict(
        after,
        before_checks=before,
        independent_checks=checks,
        errors=errors,
        specification_unchanged=unchanged,
    )
