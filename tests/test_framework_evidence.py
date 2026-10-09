"""Incomplete identities and reused measurements cannot produce acceptance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from oh_my_vllm.performance.analysis import validate_progress

from benchmarks.common import _parse_bench_config
from benchmarks.framework import validate_identities, validate_repetitions

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    geometry = dict(
        hidden_size=5120,
        num_hidden_layers=64,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        rms_norm_eps=1e-6,
        layer_types=[
            "full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)
        ],
    )
    files = {
        "config.json": {"bytes": 1, "sha256": "a" * 64},
        "model.safetensors": {"bytes": 1, "sha256": "b" * 64},
    }
    tree = hashlib.sha256()
    for name, row in sorted(files.items()):
        encoded = name.encode()
        tree.update(len(encoded).to_bytes(8, "little"))
        tree.update(encoded)
        tree.update(bytes.fromhex(row["sha256"]))
    checkpoint = dict(
        config={"text_config": geometry}, files=files, tree_sha256=tree.hexdigest()
    )
    cuda = dict(
        nvcc_path="/cuda/bin/nvcc",
        nvcc_version="CUDA 13.0",
        compiler_identity_source="live",
        so_path="/tmp/kernel.so",
        so_sha256="c" * 64,
        pdl=True,
    )
    uuid = "GPU-" + "00000000-0000-0000-0000-000000000001"
    policy = dict(multi_stream=True, pdl=True)
    stdout = (
        "INFO BENCH_CONFIG block_size=784 max_model_len=262144 "
        "max_num_batched_tokens=32768 num_speculative_tokens=0\n"
        "CUDA_BUILD_PROVENANCE " + json.dumps(cuda) + "\n"
    )
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "python/oh_my_vllm/performance").glob("*.py")
    }
    return dict(
        source=dict(
            git_commit="d" * 40,
            binary_sha256="e" * 64,
            git_diff_sha256="f" * 64,
            rust_sources_sha256="a" * 64,
            python_sources_sha256=sources,
        ),
        workload=dict(mode="ordinary"),
        trace=dict(
            runtime_contract=dict(sources={}, versions={"torch": "2.14.0"}),
            device=dict(
                uuid=uuid,
                name="NVIDIA B200",
                compute_capability=[10, 0],
                sm_count=148,
                max_clock_hz=1965e6,
                register_bytes_per_sm=262144,
                l2_bytes=132644864,
            ),
            execution_policy=policy,
            allocator_backend="native",
            cuda_provenance=cuda,
        ),
        hardware=dict(
            gpu=uuid,
            max_sm_clock_hz=1965e6,
            gpu_info=uuid + ", NVIDIA B200, 180000 MiB, 580.1",
            cpu_affinity=[0],
        ),
        runtime=dict(
            runner="oh_my_vllm.worker.model_runner.OhMyVllmWorker",
            kernel_backend="cuda",
            vllm_importable=False,
            execution_policy=policy,
            python="python",
            python_version="3.12",
            allocator_config="test",
            packages={"torch": "2.14.0"},
        ),
        checkpoints={"target": checkpoint},
        checkpoints_after={"target": checkpoint},
        stdout=stdout,
        cuda_provenance=cuda,
        worker_capacities=dict(fa=1400, mamba=128),
        scheduler_capacities=dict(fa=1400, mamba=128),
        measurement_audit=dict(
            log_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
            cache_tree_sha256={"cache": "b" * 64},
        ),
    )


def test_complete_identity_and_real_quoted_rust_build_stamp():
    validate_identities(evidence())
    stamp = "a" * 64
    assert (
        _parse_bench_config(
            'INFO BENCH_CONFIG rust_sources_sha256="' + stamp + '"',
            "rust_sources_sha256",
            str,
        )
        == stamp
    )


@pytest.mark.parametrize(
    "field",
    [
        "hardware",
        "runtime",
        "checkpoints",
        "checkpoints_after",
        "cuda_provenance",
        "worker_capacities",
        "scheduler_capacities",
        "measurement_audit",
        "stdout",
    ],
)
def test_missing_identity_is_rejected(field):
    artifact = evidence()
    artifact.pop(field)
    with pytest.raises((ValueError, KeyError)):
        validate_identities(artifact)


def test_empty_checkpoint_pair_is_rejected():
    artifact = evidence()
    artifact["checkpoints"] = artifact["checkpoints_after"] = {}
    with pytest.raises(ValueError, match="checkpoint identities"):
        validate_identities(artifact)


@pytest.mark.parametrize(
    "corruption",
    ["binary", "python", "gpu", "policy", "manifest", "geometry", "cuda", "cache_hash"],
)
def test_conflicting_identity_is_rejected(corruption):
    artifact = evidence()
    if corruption == "binary":
        artifact["source"]["binary_sha256"] = "bad"
    elif corruption == "python":
        artifact["source"]["python_sources_sha256"] = {}
    elif corruption == "gpu":
        artifact["trace"]["device"]["uuid"] = "GPU-other"
    elif corruption == "policy":
        artifact["trace"]["execution_policy"] = {"pdl": False}
    elif corruption == "manifest":
        artifact["checkpoints"]["target"]["tree_sha256"] = "c" * 64
    elif corruption == "geometry":
        artifact["checkpoints"]["target"]["config"]["text_config"]["hidden_size"] = 1
    elif corruption == "cuda":
        artifact["trace"]["cuda_provenance"] = {}
    else:
        artifact["measurement_audit"]["log_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        validate_identities(artifact)


def repeated_batches():
    workload = dict(mode="ordinary", batch_size=1, input_len=4, output_len=2)
    runs, steps = [], []
    for rid in range(1, 8):
        for query, context in ((4, 0), (1, 4)):
            request = dict(
                request_id=rid,
                query=query,
                context=context,
                drafts=0,
                sample_rows=1,
                prefill=query > 1,
                state_source=0 if context == 0 else 1,
                state_destinations=[1],
                state_row_writes=[-1] * (query - 1) + [1],
                fa_block_table=[1],
                mamba_block_table=[1],
                unique_token_ids=[7],
            )
            output = dict(
                request_id=rid, kept=1, computed=query, accepted=0, state_source=1
            )
            steps.append(
                dict(
                    operations=[
                        dict(kind="target", requests=[request]),
                        dict(kind="commit", copies=[], outputs=[output]),
                    ]
                )
            )
        runs.append(
            dict(
                batch_size=1,
                input_len=4,
                output_len=2,
                speculative_mode="none",
                output_tokens=2,
                preemptions=0,
                initial_prefix_hit_tokens=0,
                elapsed_s=4,
                steps=2,
                request_phases=[
                    dict(
                        request_id=rid,
                        submitted_s=0,
                        first_token_s=1,
                        last_token_s=3,
                        first_step=1,
                        last_step=2,
                    )
                ],
            )
        )
    artifact = dict(
        workload=workload, trace=dict(steps=steps), warmups=runs[:2], runs=runs[2:]
    )
    artifact["stdout"] = repetition_output(artifact)
    return artifact


def repetition_output(artifact):
    return "\n".join(
        "BENCH_PHASE iteration="
        + str(i)
        + " measured="
        + ("true" if i >= 2 else "false")
        + "\n"
        + ("BENCH_RESULT " if i >= 2 else "WARMUP_RESULT ")
        + json.dumps(run)
        for i, run in enumerate(artifact["warmups"] + artifact["runs"])
    )


def test_two_warmups_and_five_distinct_trace_intervals():
    validate_repetitions(repeated_batches())


@pytest.mark.parametrize(
    "corruption", ["duplicate", "swap", "warmup_reuse", "log_order"]
)
def test_reused_or_reordered_measurements_are_rejected(corruption):
    artifact = repeated_batches()
    if corruption == "duplicate":
        artifact["runs"] = [artifact["runs"][0]] * 5
    elif corruption == "swap":
        artifact["runs"][0], artifact["runs"][1] = (
            artifact["runs"][1],
            artifact["runs"][0],
        )
    elif corruption == "warmup_reuse":
        artifact["runs"][0] = artifact["warmups"][0]
    artifact["stdout"] = repetition_output(artifact)
    if corruption == "log_order":
        artifact["stdout"] = artifact["stdout"].replace(
            "iteration=2 measured=true", "iteration=2 measured=false"
        )
    with pytest.raises(ValueError):
        validate_repetitions(artifact)


@pytest.mark.parametrize(
    "field,value",
    [
        ("query", 99),
        ("context", 1),
        ("sample_rows", 0),
        ("drafts", 1),
        ("prefill", False),
    ],
)
def test_inflated_or_inconsistent_effective_target_rows_are_rejected(field, value):
    artifact = repeated_batches()
    trace = copy.deepcopy(artifact["trace"])
    trace["steps"][0]["operations"][0]["requests"][0][field] = value
    with pytest.raises(ValueError):
        validate_progress(trace, artifact["warmups"][0])


@pytest.mark.parametrize(
    "field,value", [("computed", 99), ("accepted", 1), ("kept", 2)]
)
def test_inconsistent_commit_cannot_raise_the_bound(field, value):
    artifact = repeated_batches()
    artifact["trace"]["steps"][0]["operations"][1]["outputs"][0][field] = value
    with pytest.raises(ValueError):
        validate_progress(artifact["trace"], artifact["warmups"][0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_source", 2),
        ("state_row_writes", [1, 1, 1, 1]),
        ("state_destinations", [2]),
        ("incoming_context", 999),
    ],
)
def test_fabricated_recurrent_or_incoming_ranges_are_rejected(field, value):
    artifact = repeated_batches()
    artifact["trace"]["steps"][0]["operations"][0]["requests"][0][field] = value
    with pytest.raises(ValueError):
        validate_progress(artifact["trace"], artifact["warmups"][0])


def test_fabricated_checkpoint_copy_is_rejected():
    artifact = repeated_batches()
    artifact["trace"]["steps"][0]["operations"][1]["copies"] = [[1, 2]]
    with pytest.raises(ValueError, match="checkpoint copies"):
        validate_progress(artifact["trace"], artifact["warmups"][0])


def test_empty_auxiliary_checkpoint_file_is_permitted():
    artifact = evidence()
    checkpoint = artifact["checkpoints"]["target"]
    checkpoint["files"]["optional.txt"] = dict(
        bytes=0, sha256=hashlib.sha256(b"").hexdigest()
    )
    tree = hashlib.sha256()
    for name, row in sorted(checkpoint["files"].items()):
        encoded = name.encode()
        tree.update(len(encoded).to_bytes(8, "little"))
        tree.update(encoded)
        tree.update(bytes.fromhex(row["sha256"]))
    checkpoint["tree_sha256"] = tree.hexdigest()
    validate_identities(artifact)


@pytest.mark.parametrize("clock", [1, 1965, 1965e6 + 1])
def test_changed_hardware_clock_cannot_increase_the_bound(clock):
    artifact = evidence()
    artifact["trace"]["device"]["max_clock_hz"] = clock
    with pytest.raises(ValueError):
        validate_identities(artifact)


def test_collector_and_worker_clock_ceilings_must_match():
    artifact = evidence()
    artifact["hardware"]["max_sm_clock_hz"] = 1
    with pytest.raises(ValueError, match="collector hardware"):
        validate_identities(artifact)


def test_lower_bound_cannot_exceed_observed_time():
    from benchmarks.framework import validate_bound_consistency

    times = [dict(prefill=1, decode=2)]
    bounds = [dict(prefill=0.9, decode=1.9, prefill_model={}, decode_model={})]
    validate_bound_consistency(times, bounds)
    bounds[0]["decode"] = 2.01
    with pytest.raises(ValueError, match="exceeds observed"):
        validate_bound_consistency(times, bounds)
