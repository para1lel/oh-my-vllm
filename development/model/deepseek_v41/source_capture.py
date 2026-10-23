"""Short released-source MP4 evidence, separate from the mathematical FP64 oracle."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import tempfile
import types
import uuid
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import fixtures
from .identity import SOURCE_FILES, parse_json
from .layout import tensor_layout
from .reference import control_bytes, file_snapshot, output_path

LOCAL_FILES = (
    "source_capture.py",
    "source_weights.py",
    "memory.py",
    "layout.py",
    "weights.py",
    "identity.py",
    "fixtures.py",
    "reference.py",
)
REQUIREMENTS = (
    "model-reference",
    "deepseek-reference",
    "deepseek-baseline",
)
MODULES = ("engram", "image_processor", "vision", "kernel", "model")
TEXT_FIELDS = {
    "vocab_size": "vocab_size",
    "dim": "hidden_size",
    "moe_inter_dim": "moe_intermediate_size",
    "n_layers": "num_hidden_layers",
    "n_mtp_layers": "num_nextn_predict_layers",
    "n_heads": "num_attention_heads",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func",
    "norm_topk_prob": "norm_topk_prob",
    "route_scale": "routed_scaling_factor",
    "swiglu_limit": "swiglu_limit",
    "q_lora_rank": "q_lora_rank",
    "head_dim": "head_dim",
    "rope_head_dim": "qk_rope_head_dim",
    "norm_eps": "rms_norm_eps",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    "window_size": "sliding_window",
    "compress_ratios": "compress_ratios",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "compress_rope_theta": "compress_rope_theta",
    "rope_theta": "rope_theta",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "candidate_source_layer": "candidate_source_layer_id",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_pad_id": "engram_pad_token_id",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
    "dspark_block_size": "dspark_block_size",
    "dspark_noise_token_id": "dspark_noise_token_id",
    "dspark_target_layer_ids": "dspark_target_layer_ids",
    "dspark_markov_rank": "dspark_markov_rank",
    "dspark_n_routed_experts": "dspark_n_routed_experts",
    "dspark_n_activated_experts": "dspark_num_experts_per_tok",
}
VISION_FIELDS = {
    "vision_n_layers": "num_hidden_layers",
    "vision_dim": "hidden_size",
    "vision_n_heads": "num_attention_heads",
    "vision_inter_dim": "intermediate_size",
    "vision_patch_size": "patch_size",
    "vision_rope_theta": "rope_theta",
    "vision_downsample_ratio": "downsample_ratio",
    "vision_max_n_token": "max_image_tokens",
    "vision_min_pixels": "min_pixels",
    "vision_max_wh_ratio": "max_wh_ratio",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def model_arguments(root, manifest):
    if manifest.get("verification") != "full":
        raise ValueError("source capture requires a full checkpoint manifest")
    config = parse_json(control_bytes(root, "config.json", manifest))
    tensor_layout(config)
    text, vision = config["text_config"], config["vision_config"]
    values = {key: text[name] for key, name in TEXT_FIELDS.items()}
    values.update({key: vision[name] for key, name in VISION_FIELDS.items()})
    rope = text["rope_scaling"]
    values.update(
        original_seq_len=rope["original_max_position_embeddings"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        image_token_id=config["image_token_id"],
        dtype=config["quantization_config"]["quant_method"],
        expert_dtype=config["quantization_config"]["expert_dtype"],
        gate_temp=1.0,
        max_batch_size=1,
        max_seq_len=144,
        temperature=0,
    )
    # All reachable index positions fit; this path does not select a tied subset.
    if values["max_seq_len"] > values["index_topk"]:
        raise ValueError("source capture supports only the short all-visible case")
    return config, values


def local_identity():
    repository = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).parent / name for name in LOCAL_FILES]
    paths.extend(
        repository / "requirements" / f"{name}.{suffix}"
        for name in REQUIREMENTS
        for suffix in ("txt", "lock")
    )
    paths.extend(
        repository / name
        for name in ("development/serving/gpu.py", "scripts/with-gpu.sh")
    )
    return {
        str(path.relative_to(repository)): digest(path.read_bytes()) for path in paths
    }


def environment_identity():
    packages = {
        item.metadata["Name"].lower().replace("_", "-"): item.version
        for item in importlib.metadata.distributions()
    }
    lock = Path(__file__).resolve().parents[3] / "requirements/deepseek-baseline.lock"
    for name, version in re.findall(r"^([\w.-]+)==([^\s]+)", lock.read_text(), re.M):
        if packages.get(name.lower().replace("_", "-")) != version:
            raise ValueError(f"source baseline dependency differs from lock: {name}")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if (
                key.startswith(("CUDA", "NCCL", "TORCH", "TILELANG"))
                and not key.startswith("TORCHELASTIC")
            )
            or key in ("LD_LIBRARY_PATH", "CUBLAS_WORKSPACE_CONFIG")
        },
    }


def import_source(root, manifest):
    """Execute verified bytes, without a source search path or cached bytecode."""
    if not sys.dont_write_bytecode:
        raise ValueError("released-source imports require python -B")
    sources = {name: control_bytes(root, name, manifest) for name in SOURCE_FILES}
    for name in MODULES:
        if name in sys.modules:
            raise ValueError(f"released-source module name already imported: {name}")
    for name in MODULES:
        path = root / "inference" / f"{name}.py"
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(
            compile(sources[f"inference/{name}.py"], str(path), "exec"), module.__dict__
        )
    model = sys.modules["model"]
    return model, {name: digest(data) for name, data in sources.items()}


def install_index_owner_hook(source):
    original = source.Indexer.forward

    def forward(self, *args, **kwargs):
        if self.owns_k:
            source.shared_attn.index_k = self.k_cache
        return original(self, *args, **kwargs)

    source.Indexer.forward = forward
    return {
        "name": "index-owner-publication-before-forward-v1",
        "sha256": digest(inspect.getsource(install_index_owner_hook).encode()),
    }


def vector_identity(vectors, dtype):
    if set(vectors) != {"prefill", "decode"}:
        raise ValueError("short capture requires both complete endpoint vectors")
    result = {}
    for name, value in vectors.items():
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (129280,)
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"short capture requires finite full {dtype} vectors")
        result[name] = {
            "shape": [129280],
            "dtype": str(dtype),
            "sha256": digest(value.astype(dtype.newbyteorder("<")).tobytes()),
        }
    return result


def numerical_metrics(vectors, reference):
    vector_identity(reference, np.dtype("float64"))
    result = {}
    for name, value in vectors.items():
        actual, expected = value.astype(np.float64), reference[name]
        error = actual - expected
        norm = float(np.linalg.norm(expected))
        product = norm * float(np.linalg.norm(actual))
        result[name] = {
            "nrmse": float(np.linalg.norm(error) / norm) if norm else None,
            "maxabs": float(np.max(np.abs(error))),
            "cosine": float(np.dot(actual, expected) / product) if product else None,
            "source_top1": int(actual.argmax()),
            "reference_top1": int(expected.argmax()),
            "top1_equal": bool(actual.argmax() == expected.argmax()),
        }
    return result


def publish_short(output, root, metadata, vectors, reference, *, verify_unchanged):
    output = output_path(output, root)
    sidecar = output.with_suffix(".json")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("source capture JSON output already exists")
    metadata = dict(
        metadata,
        arrays=vector_identity(vectors, np.dtype("float32")),
        metrics=numerical_metrics(vectors, reference),
        metrics_policy="diagnostics-only; no compound acceptance limits",
    )
    with (
        tempfile.NamedTemporaryFile(dir=output.parent) as archive,
        tempfile.NamedTemporaryFile(dir=output.parent) as document,
    ):
        np.savez(archive, metadata=np.asarray(encoded(metadata).decode()), **vectors)
        archive.flush()
        os.fsync(archive.fileno())
        archive.seek(0)
        metadata["npz_sha256"] = hashlib.file_digest(archive, "sha256").hexdigest()
        document.write(encoded(metadata) + b"\n")
        document.flush()
        os.fsync(document.fileno())
        verify_unchanged()
        os.link(document.name, sidecar)
        try:
            # The NPZ is the commit marker, published only after its complete JSON.
            os.link(archive.name, output)
        except BaseException:
            if sidecar.is_file() and os.path.samefile(sidecar, document.name):
                sidecar.unlink()
            raise
    return metadata


def launch_identity():
    from development.serving.gpu import parse_uuids

    uuids = parse_uuids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    rank = int(os.environ.get("RANK", "-1"))
    if (
        os.environ.get("OH_MY_VLLM_GPU_COUNT") != "4"
        or not os.environ.get("OH_MY_VLLM_RUN_ID")
        or os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("LOCAL_WORLD_SIZE") != "4"
        or not 0 <= rank < 4
        or int(os.environ.get("LOCAL_RANK", "-1")) != rank
        or len(uuids) != 4
    ):
        raise ValueError("use the GPU wrapper with four local torchrun ranks")
    return rank, uuids


def cuda_uuid(value):
    return "GPU-" + str(uuid.UUID(bytes=bytes(value.bytes)))


def agree(dist, value):
    gathered = [None] * 4
    dist.all_gather_object(gathered, value)
    if any(item != value for item in gathered):
        raise ValueError("source capture identities or complete vectors differ by rank")


def capture_short(model_root, manifest_path, output, read_bytes=64 << 20):
    """GPU entry point. Call only from the explicit CLI under the GPU wrapper."""
    rank, uuids = launch_identity()
    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    if output.with_suffix(".json").exists() or output.with_suffix(".json").is_symlink():
        raise FileExistsError("source capture JSON output already exists")
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    if digest(manifest_bytes) != fixtures.CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("checkpoint manifest differs from the fixed FP64 fixture")
    manifest = parse_json(manifest_bytes)
    snapshot = file_snapshot(root, manifest["files"])
    sources = local_identity()
    environment = environment_identity()
    config, values = model_arguments(root, manifest)
    fixture_metadata, reference = fixtures.load_short()

    import torch
    import torch.distributed as dist
    from tokenizers import Tokenizer

    from .source_weights import SourceWeights

    source, released = import_source(root, manifest)
    if {field.name for field in fields(source.ModelArgs)} != values.keys():
        raise ValueError("explicit root-config mapping omits or adds source arguments")
    args = source.ModelArgs(**values)
    patch = install_index_owner_hook(source)
    reader = SourceWeights(root, manifest, config, rank, read_bytes=read_bytes)
    common = {
        "schema_version": 1,
        "case": "short",
        "reference_kind": "released-source-mp4-source-precision",
        "checkpoint_manifest_sha256": digest(manifest_bytes),
        "checkpoint": manifest,
        "root_config_sha256": manifest["files"]["config.json"]["sha256"],
        "model_args": values,
        "local_sources": sources,
        "released_sources": released,
        "source_patches": [patch],
        "environment": environment,
        "fixture_sha256": fixtures.SHORT_SHA256,
        "inputs": {
            "tokens": fixture_metadata["tokens"],
            "decode_token": fixture_metadata["decode_token"],
        },
        "execution": {
            "world_size": 4,
            "parallelism": "released-mp4; owned experts; replicated shared experts",
            "physical_uuids": list(uuids),
            "read_bytes": read_bytes,
            "run_id": os.environ["OH_MY_VLLM_RUN_ID"],
            "default_dtype": "bfloat16",
            "matmul_tf32": False,
            "cpu_threads_per_rank": 1,
            "seed": 33377335,
        },
    }
    previous_dtype = torch.get_default_dtype()
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        device = torch.device("cuda", rank)
        properties = torch.cuda.get_device_properties(device)
        physical_uuid = cuda_uuid(properties.uuid)
        if physical_uuid != uuids[rank]:
            raise ValueError("physical CUDA UUID differs from the leased rank mapping")
        agree(dist, digest(encoded(common)))
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(1)
        torch.manual_seed(33377335)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = Tokenizer.from_buffer(
            control_bytes(root, "tokenizer.json", manifest)
        )

        class SourceTokenizer:
            backend_tokenizer = tokenizer

            def __len__(self):
                return tokenizer.get_vocab_size(with_added_tokens=True)

        with torch.device(device):
            model = source.Transformer(args, SourceTokenizer()).eval()
        loader_metadata = reader.load_into(model)
        with torch.inference_mode(), torch.device(device):
            prompt = torch.tensor([fixture_metadata["tokens"]], dtype=torch.long)
            continuation = torch.tensor(
                [[fixture_metadata["decode_token"]]], dtype=torch.long
            )
            prefill = model(prompt, 0)[1]
            decode = model(continuation, prompt.size(1))[1]
        vectors = {}
        for name, logits in (("prefill", prefill), ("decode", decode)):
            if logits.shape != (1, 129280) or logits.dtype != torch.float32:
                raise ValueError("source must return full FP32 endpoint logits")
            vectors[name] = logits[0].cpu().numpy().copy()
        arrays = vector_identity(vectors, np.dtype("float32"))
        agree(dist, arrays)
        ranks = [None] * 4
        dist.all_gather_object(
            ranks,
            {
                "rank": rank,
                "physical_uuid": physical_uuid,
                "device": str(properties),
                "torch_cuda": torch.version.cuda,
                "loader": loader_metadata,
                "launcher": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith("TORCHELASTIC")
                    or key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
                },
            },
        )

        def verify_unchanged():
            reader.reader._verify_states()
            if (
                file_snapshot(root, manifest["files"]) != snapshot
                or manifest_path.read_bytes() != manifest_bytes
                or local_identity() != sources
                or environment_identity() != environment
            ):
                raise ValueError("checkpoint, source, manifest or environment changed")

        verify_unchanged()
        agree(dist, "inputs unchanged")
        status = [None]
        if rank == 0:
            try:
                publish_short(
                    output,
                    root,
                    dict(common, ranks=ranks),
                    vectors,
                    reference,
                    verify_unchanged=verify_unchanged,
                )
            except Exception as error:
                status[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0])
    finally:
        torch.set_default_dtype(previous_dtype)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Capture short released-source MP4 logits as separate evidence"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-bytes", type=int, default=64 << 20)
    parser.add_argument("--case", choices=("short",), default="short")
    parser.add_argument("--run-gpu", action="store_true", required=True)
    options = parser.parse_args()
    if not sys.dont_write_bytecode:
        parser.error("use python -B or PYTHONDONTWRITEBYTECODE=1")
    capture_short(
        options.model, options.checkpoint_manifest, options.output, options.read_bytes
    )


if __name__ == "__main__":
    main()
