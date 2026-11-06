"""CPU failure checks for the public grouped comparison boundary."""

import contextlib
import ctypes
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from development.operators.compare_deepseek import (
    ROOT,
    ComparisonError,
    numeric_comparison,
    plan_comparison,
    run_worker,
)
from development.operators.grouped_deepgemm import (
    TensorBytes,
    canonical_hash,
    compare_outputs,
    digest,
)


def write_document(path, data):
    path.write_text(json.dumps(data))
    return {"file": str(path), "sha256": digest(path)}


def input_spec(root):
    contract = root / "contract.txt"
    contract.write_text("raw K32 zero, MUL RN, FMA RN, final BF16")
    required = write_document(
        root / "required.json",
        {
            "required_case_ids": ["real-up", "other-required-shape"],
            "sources": [],
            "coverage_complete": False,
            "scope": "independent test inventory",
        },
    )
    counts = [1] + [0] * 383
    weights = []
    for expert in range(384):
        prefix = f"layers.0.ffn.experts.{expert}.w3."
        tensors = [
            {
                "file": str(root / "absent-checkpoint.safetensors"),
                "key": prefix + name,
                "row_range": [0, 576],
                "shape": [576, columns],
                "dtype": "U8",
                "sha256": "0" * 64,
            }
            for name, columns in (("weight", 2560), ("scale", 160))
        ]
        weights.append({"expert": expert, "weight": tensors[0], "scales": tensors[1]})
    case = {
        "case_id": "real-up",
        "groups": 384,
        "outputs": 576,
        "width": 5120,
        "boundary": "strict_encoded_expert_major_to_bf16",
        "output_dtype": "BF16",
        "counts": counts,
        "counts_sha256": hashlib.sha256(struct.pack("<384i", *counts)).hexdigest(),
        "identity": {
            "strategy": "tp4",
            "world_size": 4,
            "projection": "up",
            "rank": 0,
            "layer": 0,
            "origin": "captured",
            "source_commit": "old-capture-only",
        },
        "contract": {
            "file": str(contract),
            "sha256": digest(contract),
            "group_k": 32,
            "rule": "full_output_bytes",
            "accumulation": "zero_raw_k32_then_mul_rn_fma_rn",
        },
        "weights": weights,
        "activation": None,
        "activation_scales": None,
    }
    case["case_sha256"] = canonical_hash(case)
    return {"required_inventory": required, "cases": [case]}


def tensor_fixture(root, key, shape, dtype, payload):
    path = root / (key + ".safetensors")
    header = json.dumps(
        {key: {"dtype": dtype, "shape": shape, "data_offsets": [0, len(payload)]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return {
        "file": str(path),
        "key": key,
        "row_range": [0, shape[0]],
        "shape": shape,
        "dtype": dtype,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def python_command(*args):
    return [
        str(ROOT / "scripts/with-env.sh"),
        sys.executable,
        "-I",
        "-B",
        *map(str, args),
    ]


def process_probe(mode, root):
    import development.operators.compare_deepseek as comparison

    root = Path(root)
    original = subprocess.Popen
    child = "import signal;signal.pause()"
    worker = (
        "import json,os,signal,subprocess,sys;from pathlib import Path;"
        "p=subprocess.Popen([sys.argv[1],sys.executable,'-I','-B','-c',sys.argv[3]]);"
        "Path(sys.argv[2]).write_text(json.dumps([os.getpid(),p.pid]));"
        + ("sys.exit(7)" if mode == "exit" else "signal.pause()")
    )

    def substitute(_command, **kwargs):
        command = python_command(
            "-c", worker, ROOT / "scripts/with-env.sh", root / "owned.json", child
        )
        return original(command, **kwargs)

    with patch.object(comparison.subprocess, "Popen", substitute):
        run_worker(
            "candidate",
            root / "unused",
            root / "unused",
            root / "unused",
            SimpleNamespace(),
            {},
            root,
            0.5 if mode == "timeout" else 20,
        )


def ready_spec(test):
    root, value = test.root, test.spec["cases"][0]
    identity = value["identity"]
    counts = tensor_fixture(
        root, "counts", [384], "I32", struct.pack("<384i", *value["counts"])
    )
    source = tensor_fixture(root, "bf16", [1, 5120], "BF16", bytes(10240))
    identity.update(
        origin="derived_from_capture",
        input_state="derived_input_complete",
        sources=[],
        counts_tensor=counts,
        bf16_source=source,
    )
    value["activation"] = tensor_fixture(
        root, "activation", [1, 5120], "U8", bytes(5120)
    )
    value["activation_scales"] = tensor_fixture(
        root, "scales", [1, 160], "U8", bytes([127]) * 160
    )
    payload = bytes(576 * 2560)
    header = {}
    for expert in value["weights"]:
        for name, size in (("weight", len(payload)), ("scales", 576 * 160)):
            item = expert[name]
            item.update(
                file=str(root / "weights.safetensors"),
                sha256=hashlib.sha256(payload[:size]).hexdigest(),
            )
            header[item["key"]] = {
                "dtype": "U8",
                "shape": item["shape"],
                "data_offsets": [0, size],
            }
    encoded = json.dumps(header).encode()
    (root / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    native = json.loads(test.candidate.read_text())
    native["functions"] = ["quantize"]
    quantizer = write_document(root / "quantizer.json", native)
    record = {
        "status": "completed",
        "operation": "quantize",
        "parameters": [8, 32, 1, True, False],
        "bf16_source": source,
        "counts_tensor": counts,
        "counts_sha256": value["counts_sha256"],
        "outputs": {name: value[name] for name in ("activation", "activation_scales")},
        "quantizer": {
            "manifest": quantizer["file"],
            "manifest_sha256": quantizer["sha256"],
            "library": native["library"],
            "source_id": native["source_id"],
            "binary_sha256": native["binary_id"],
        },
    }
    value["generation"] = write_document(root / "generation.json", record)
    ready_reference(test)
    value["case_sha256"] = canonical_hash(
        {k: v for k, v in value.items() if k != "case_sha256"}
    )
    return value, record


def ready_reference(test):
    root = test.root
    package = root / "package"
    (package / "include").mkdir(parents=True)
    (package / "__init__.py").write_text("CPU identity fixture")
    (package / "include/kernel.cuh").write_text("original JIT header")
    library = package / "_C.so"
    library.write_bytes(b"never loaded")
    revisions = {"DeepGEMM": "fixed-test-revision"}
    sources = write_document(
        root / "package-sources.json",
        {
            "source_revisions": revisions,
            "files": {
                str(p.relative_to(package)): digest(p)
                for p in package.rglob("*")
                if p.is_file() and p != library
            },
        },
    )
    test.reference = root / "reference.json"
    write_document(
        test.reference,
        {
            "status": "HOST_BUILD_AND_CPU_IMPORT_PASSED_GPU_UNVERIFIED",
            "native": {"path": str(library), "sha256": digest(library)},
            "source_revisions": revisions,
            "package_sources": sources,
            "grouped_fp4_source_findings": {"recipe": [1, 1, 32]},
        },
    )


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = input_spec(self.root)
        self.candidate = self.root / "candidate.json"
        library = self.root / "candidate.so"
        library.write_bytes(b"CPU identity fixture, never loaded")
        write_document(
            self.candidate,
            {
                "library": str(library),
                "binary_id": digest(library),
                "source_id": "test-source",
            },
        )
        self.reference = self.root / "missing-reference.json"

    def test_plan_keeps_independent_required_cases_and_missing_reference(self):
        case, summary = plan_comparison(self.spec, self.candidate, self.reference)
        self.assertIsNotNone(case)
        self.assertEqual(
            summary["required_case_ids"], ["real-up", "other-required-shape"]
        )
        self.assertEqual([x["state"] for x in summary["cases"]], ["missing", "missing"])
        self.assertEqual(summary["measured_case_ids"], [])
        self.assertFalse(summary["model_level_gate_open"])
        self.spec["cases"][0]["case_id"] = "not-in-required-inventory"
        with self.assertRaises(ComparisonError):
            plan_comparison(self.spec, self.candidate, self.reference)

    def test_cli_plan_does_not_import_gpu_or_runtime(self):
        path = self.root / "input.json"
        write_document(path, self.spec)
        script = (
            "import builtins,sys;sys.path.insert(0,sys.argv[1]);"
            "original=builtins.__import__;"
            'exec(\'def guarded(name,*a,**k):\\n if name.split(".")[0] in '
            '("torch","deep_gemm","oh_my_vllm","tvm_ffi"):\\n  raise RuntimeError(name)'
            "\\n return original(name,*a,**k)'.replace('\\\\n','\\n'));"
            "builtins.__import__=guarded;"
            "from development.operators.compare_deepseek import main;"
            "raise SystemExit(main(sys.argv[2:]))"
        )
        command = python_command(
            "-c",
            script,
            ROOT,
            "--cases",
            path,
            "--candidate",
            self.candidate,
            "--reference",
            self.reference,
            "--output",
            self.root / "plan",
            "--plan",
        )
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "missing")
        self.assertTrue((self.root / "plan" / "summary.json").is_file())

    def test_digest_failure_cannot_become_planned(self):
        self.spec["cases"][0]["case_sha256"] = "0" * 64
        _, summary = plan_comparison(self.spec, self.candidate, self.reference)
        self.assertEqual(summary["cases"][0]["state"], "missing")
        self.assertIn("identity", summary["cases"][0]["reason"])
        self.spec["required_inventory"]["sha256"] = "0" * 64
        with self.assertRaises(ComparisonError):
            plan_comparison(self.spec, self.candidate, self.reference)

    def test_tensor_payload_hash_is_checked_without_torch(self):
        tensor = TensorBytes.parse(
            tensor_fixture(self.root, "encoded", [1, 2], "U8", b"\x01\x02")
        )
        path = tensor.file
        self.assertEqual(tensor.read(), b"\x01\x02")
        path.write_bytes(path.read_bytes()[:-1] + b"\x03")
        with self.assertRaises(ComparisonError):
            tensor.read()

    def test_complete_bytes_and_unmeasured_state(self):
        case = SimpleNamespace(rows=1, outputs=2, expected=None)
        left, right = self.root / "left.bf16", self.root / "right.bf16"
        left.write_bytes(b"\x80\x3f\x00\xc0")
        for payload, state in (
            (b"\x80\x3f", "numeric_failed"),
            (b"\x81\x3f\x00\xc0", "numeric_failed"),
            (left.read_bytes(), "unmeasured"),
            (b"\x80\x7f\x00\xc0", "numeric_failed"),
        ):
            right.write_bytes(payload)
            record = compare_outputs(case, left, right)
            self.assertEqual(record["state"], state)
            self.assertNotIn("passed", record)
        left.write_bytes(right.read_bytes())
        self.assertEqual(compare_outputs(case, left, right)["state"], "numeric_failed")

    def test_derived_record_rejects_pending_bad_source_and_incomplete_generation(self):
        value, record = ready_spec(self)
        _, summary = plan_comparison(self.spec, self.candidate, self.reference)
        self.assertEqual(summary["cases"][0]["state"], "planned")
        self.assertFalse(summary["model_level_gate_open"])
        complete = json.loads(json.dumps(value))
        problems = ("pending", "source", "args", "scale", "quantizer", "constructed")
        for problem in problems:
            with self.subTest(problem=problem):
                value = json.loads(json.dumps(complete))
                receipt = json.loads(json.dumps(record))
                if problem == "pending":
                    value["identity"]["input_state"] = "derived_input_pending"
                elif problem == "source":
                    value["identity"]["bf16_source"]["sha256"] = "0" * 64
                elif problem == "args":
                    receipt["parameters"][-1] = True
                elif problem == "scale":
                    receipt["outputs"]["activation_scales"]["sha256"] = "0" * 64
                elif problem == "quantizer":
                    receipt["quantizer"]["source_id"] = "wrong-source"
                else:
                    value["identity"]["origin"] = "constructed"
                value["generation"] = write_document(
                    self.root / "generation.json", receipt
                )
                value["case_sha256"] = canonical_hash(
                    {k: v for k, v in value.items() if k != "case_sha256"}
                )
                self.spec["cases"][0] = value
                _, summary = plan_comparison(self.spec, self.candidate, self.reference)
                self.assertIn(
                    summary["cases"][0]["state"], ("missing", "semantic_mismatch")
                )
                self.assertFalse(summary["model_level_gate_open"])

    def test_header_changes_fail_with_unchanged_host_binary(self):
        ready_spec(self)
        _, before = plan_comparison(self.spec, self.candidate, self.reference)
        self.assertEqual(before["cases"][0]["state"], "planned")
        library = self.root / "package/_C.so"
        original_hash = digest(library)
        (self.root / "package/include/kernel.cuh").write_text("changed JIT header")
        _, after = plan_comparison(self.spec, self.candidate, self.reference)
        self.assertEqual(digest(library), original_hash)
        self.assertEqual(after["cases"][0]["state"], "missing")
        self.assertIn("package source", after["cases"][0]["reason"])

    def test_worker_groups_are_cleaned_on_exit_timeout_and_controller_sigterm(self):
        libc = ctypes.CDLL(None)
        old = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(old), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        try:
            for mode in ("exit", "timeout", "sigterm"):
                root = self.root / mode
                root.mkdir()
                command = python_command(
                    "-c",
                    "import sys;sys.path.insert(0,sys.argv[1]);"
                    "from development.operators.test_deepseek_comparison "
                    "import process_probe;process_probe(*sys.argv[2:])",
                    ROOT,
                    mode,
                    root,
                )
                process = subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                owned = []
                try:
                    deadline = time.monotonic() + 10
                    while (
                        not (root / "owned.json").exists()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.01)
                    owned = json.loads((root / "owned.json").read_text())
                    if mode == "sigterm":
                        process.send_signal(signal.SIGTERM)
                    process.wait(timeout=10)
                    deadline = time.monotonic() + 2
                    for pid in owned:
                        stat = Path(f"/proc/{pid}/stat")
                        while (
                            stat.exists()
                            and stat.read_text().split()[2] not in ("Z", "X")
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue(
                            not stat.exists()
                            or stat.read_text().split()[2] in ("Z", "X"),
                            (mode, pid),
                        )
                    if mode == "sigterm":
                        self.assertEqual(process.returncode, 143)
                finally:
                    process.kill() if process.poll() is None else None
                    process.wait()
                    for pid in owned:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                        with contextlib.suppress(ChildProcessError):
                            os.waitpid(pid, 0)
        finally:
            libc.prctl(36, old.value, 0, 0, 0)

    def test_worker_failure_never_produces_numeric_success(self):
        input_path = self.root / "input.json"
        write_document(input_path, self.spec)
        case, summary = plan_comparison(self.spec, self.candidate, self.reference)
        paths = (input_path, self.candidate, self.reference)
        with self.assertRaises(ComparisonError) as caught:
            run_worker("candidate", *paths, case, {}, self.root, 10)
        self.assertEqual(caught.exception.state, "unmeasured")
        self.assertTrue((self.root / "candidate" / "worker.log").is_file())
        summary["cases"][0]["state"] = "planned"
        with (
            patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "test-only", "OH_MY_VLLM_RUN_ID": "test"},
            ),
            patch(
                "development.operators.compare_deepseek.run_worker",
                side_effect=caught.exception,
            ),
        ):
            result = numeric_comparison(case, summary, *paths, self.root, 10)
        self.assertEqual(result["cases"][0]["state"], "unmeasured")
        self.assertEqual(result["measured_case_ids"], [])
        self.assertFalse(result["model_level_gate_open"])


if __name__ == "__main__":
    unittest.main()
