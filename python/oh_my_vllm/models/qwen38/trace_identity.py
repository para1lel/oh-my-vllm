import hashlib
import json
import os
import subprocess
from pathlib import Path

from ...operators.runtime.native import file_sha256, load, loaded_identities
from ...source import source_identity
from ...worker.device_identity import device_identity
from ...worker.execution import execution_policy


def native_identity():
    from ...operators.fa_fp8.api import FUNCTIONS as fa
    from ...operators.ffn.api import FUNCTIONS as ffn
    from ...operators.gdn_radix.api import FUNCTIONS as gdn
    from ...operators.mtp_fp8.api import FUNCTIONS as mtp
    from ...operators.output.api import FUNCTIONS as sample
    from ...operators.runtime.device import FUNCTIONS as device
    from ...operators.vision.api import FUNCTIONS as vision

    modules = dict(
        device=device,
        ffn=ffn,
        fa_fp8=fa,
        gdn_radix=gdn,
        mtp_fp8=mtp,
        sample=sample,
        vision=vision,
    )
    for name, functions in modules.items():
        load(name, functions)
    loaded = loaded_identities()
    return {name: loaded[name] for name in modules}


def loaded_identity(checkpoint, tokenizer):
    path = os.environ.get("OH_MY_VLLM_RUN_MANIFEST")
    if not path:
        return {}
    try:
        return observed_identity(path, checkpoint, tokenizer)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        return {"identity_error": str(error)}


def observed_identity(path, checkpoint, tokenizer):
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be an object")
    actual = dict(
        source=source_identity(),
        binary=file_sha256(f"/proc/{os.getppid()}/exe"),
        native=native_identity(),
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        device=device_identity(),
        execution=execution_policy(),
    )
    for name, value in actual.items():
        if manifest.get(name) != value:
            raise ValueError(f"loaded {name} differs from the run manifest")
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return actual | {"artifact_id": digest, "run_manifest": manifest}


def dispatch_facts(rows, verify, graph):
    counts = [len(row["token_ids"]) for row in rows]
    total, maximum = sum(counts), max(counts)
    return dict(
        execution="cuda_graph" if graph else "eager",
        rows=total,
        max_queries=maximum,
        verify=verify,
        gdn_ba_precision="bf16" if total > 32 or (verify and maximum == 1) else "fp32",
        requests=[
            dict(request_id=row["request_id"], lease_id=row["lease_id"]) for row in rows
        ],
    )


SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)


def record_prepared(trace, epoch, message, request, tokens):
    if trace.output is None:
        return
    try:
        bits = request.parameters().astype("<f4", copy=False).view("<u4")
        values = dict(
            worker_epoch=epoch,
            rpc_id=message["rpc_id"],
            request_id=message["request_id"],
            prompt_token_sha256=hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            sampling_f32_bits={
                name: f"{int(value):08x}"
                for name, value in zip(SAMPLING_FIELDS, bits, strict=True)
            },
            seed=request.seed,
            repetition_guard=request.options.get("sampling", {}).get(
                "repetition_guard", True
            ),
            min_tokens=request.minimum,
            max_tokens=request.limit,
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        trace.failure = str(error)
        trace.close()
        return
    trace.record("prepared", **values)
