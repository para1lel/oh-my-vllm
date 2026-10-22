import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import types
from pathlib import Path
from tempfile import NamedTemporaryFile

from .identity import VERSION, file_state, local_file, open_file, parse_json, unchanged

CHECKPOINT_FILES = ("encoding/encoding.py", "tokenizer.json", "tokenizer_config.json")
TOKENIZERS_VERSION = "0.23.3"
MAX_FILE_BYTES = 64 << 20
MAX_CASES = 24
MAX_CASE_INPUT_BYTES = 32768
MAX_CASE_TEXT_BYTES = 65536
MAX_CASE_TOKENS = 16384
MAX_TOTAL_TOKENS = 65536


def fixture_cases():
    cases = []

    def add(name, messages, **options):
        inputs = {
            "messages": messages,
            "thinking_mode": "chat",
            "context": None,
            "drop_thinking": True,
            "add_default_bos_token": True,
            "reasoning_effort": None,
            "return_multi_modal_data": True,
        }
        inputs.update(options)
        cases.append({"name": name, "inputs": copy.deepcopy(inputs)})

    question = [{"role": "user", "content": "What is 2 + 2?"}]
    add(
        "plain-chat",
        [{"role": "system", "content": "Reply briefly."}, *question],
    )
    add("chat-effort", question, reasoning_effort=100)
    for effort in (None, "low", 50, "high", 75, "max", 100, 1):
        name = "default" if effort is None else str(effort)
        add(
            f"thinking-{name}",
            question,
            thinking_mode="thinking",
            reasoning_effort=effort,
        )

    history = [
        {"role": "user", "content": "What is 1 + 1?"},
        {
            "role": "assistant",
            "reasoning_content": "earlier-reasoning",
            "content": "2",
        },
        *question,
    ]
    add("reasoning-drop", history, thinking_mode="thinking")
    add("reasoning-keep", history, thinking_mode="thinking", drop_thinking=False)

    tools = [
        {
            "type": "function",
            "namespace": {"name": "fixture", "description": "Fixture tools."},
            "function": {
                "name": name,
                "description": "Return a fixed result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                },
            },
        }
        for name in ("first", "second")
    ]
    calls = [
        {
            "id": "first",
            "type": "function",
            "namespace": "fixture",
            "function": {
                "name": "first",
                "arguments": '{"query":"alpha","count":1}',
            },
        },
        {
            "id": "second",
            "type": "function",
            "function": {
                "name": "fixture::second",
                "arguments": '{"query":"beta","count":2}',
            },
        },
    ]
    tool_history = [
        {"role": "system", "content": "Use the fixture tools.", "tools": tools},
        *history,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "call-reasoning",
            "tool_calls": calls,
        },
    ]
    results = [
        {"role": "tool", "tool_call_id": name, "content": f"{name}-result"}
        for name in ("second", "first")
    ]
    add("tools-ordered", tool_history + results, thinking_mode="thinking")
    add(
        "mid-system",
        [*history[:-1], {"role": "system", "content": "Now answer in one word."}],
        thinking_mode="thinking",
    )
    add(
        "assistant-continuation",
        [
            *question,
            {
                "role": "assistant",
                "reasoning_content": "Add two and two.",
                "content": "The answer is",
                "wo_eos": True,
            },
        ],
        thinking_mode="thinking",
    )
    add(
        "context-continuation",
        question,
        context=history[:-1],
        thinking_mode="thinking",
        reasoning_effort="max",
    )

    def image(name):
        return {"type": "image_url", "image_url": {"url": f"fixture://{name}"}}

    add(
        "ordered-images",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these in order."},
                    image("first"),
                    {"type": "text", "text": "Then this."},
                    image("second"),
                ],
            }
        ],
    )
    add(
        "context-images",
        [{"role": "user", "content": [image("current")]}],
        context=[
            {"role": "user", "content": [image("past")]},
            {"role": "assistant", "content": "I saw the past image."},
        ],
    )
    tool_images = [
        {
            "role": "tool",
            "tool_call_id": name,
            "content": [
                {"type": "text", "text": f"{name}-result"},
                image(name),
            ],
        }
        for name in ("second", "first")
    ]
    add("tool-result-images", tool_history + tool_images, thinking_mode="thinking")
    add(
        "tool-result-images-canonical",
        tool_history + tool_images[::-1],
        thinking_mode="thinking",
    )
    return cases


def read_tracked(path, snapshots):
    with open_file(path) as source:
        initial = file_state(os.fstat(source.fileno()))
        if initial[2] > MAX_FILE_BYTES:
            raise ValueError(f"codec input file is too large: {path.name}")
        data = source.read(MAX_FILE_BYTES + 1)
        unchanged(path, source, initial)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"codec input file is too large: {path.name}")
    snapshots[path] = initial
    return data, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def check_unchanged(snapshots):
    for path, initial in snapshots.items():
        try:
            current = file_state(path.stat(follow_symlinks=False))
        except OSError as error:
            raise ValueError(
                f"codec input changed during generation: {path.name}"
            ) from error
        if current != initial:
            raise ValueError(f"codec input changed during generation: {path.name}")


def reference_sources(snapshots):
    repository = Path(__file__).resolve().parents[3]
    names = (
        "development/model/deepseek_v41/codec_reference.py",
        "development/model/deepseek_v41/identity.py",
        "requirements/deepseek-reference.txt",
        "requirements/deepseek-reference.lock",
    )
    return {
        name: read_tracked(local_file(repository, name), snapshots)[1] for name in names
    }


def output_path(output, root):
    path = Path(output)
    if path.is_symlink():
        raise FileExistsError("codec output already exists as a link")
    path = path.resolve()
    repository = Path(__file__).resolve().parents[3]
    if path.is_relative_to(repository) or path.is_relative_to(root):
        raise ValueError("codec output must be external to repository and checkpoint")
    if path.suffix != ".json":
        raise ValueError("codec output must use the .json suffix")
    if path.exists():
        raise FileExistsError("codec output already exists")
    return path


def load_encoder(source, path):
    module = types.ModuleType("_deepseek_v41_verified_prompt_codec")
    module.__file__ = str(path)
    # Execute the verified bytes, without an import loader that can write bytecode.
    exec(compile(source, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "encode_messages", None)):
        raise ValueError("verified source has no encode_messages function")
    return module


def render_cases(encoder, tokenizer):
    fixtures = fixture_cases()
    if len(fixtures) > MAX_CASES or len({case["name"] for case in fixtures}) != len(
        fixtures
    ):
        raise ValueError("codec case manifest exceeds its bound or repeats a name")
    result, total = [], 0
    for fixture in fixtures:
        inputs = fixture["inputs"]
        if len(json.dumps(inputs, allow_nan=False).encode()) > MAX_CASE_INPUT_BYTES:
            raise ValueError("codec case inputs exceed their byte bound")
        text, media = encoder.encode_messages(**copy.deepcopy(inputs))
        if not isinstance(text, str) or len(text.encode()) > MAX_CASE_TEXT_BYTES:
            raise ValueError("codec rendered text exceeds its byte bound")
        if (
            not isinstance(media, dict)
            or set(media) != {"images"}
            or not isinstance(media["images"], list)
            or len(media["images"]) > 8
        ):
            raise ValueError("codec must return a bounded ordered image list")
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        total += len(tokens)
        if len(tokens) > MAX_CASE_TOKENS or total > MAX_TOTAL_TOKENS:
            raise ValueError("codec token IDs exceed their case or total bound")
        result.append(
            {
                "name": fixture["name"],
                "inputs": inputs,
                "rendered_text": text,
                "token_ids": tokens,
                "media": copy.deepcopy(media),
            }
        )
    return result


def generate(model_root, manifest_path, output):
    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    snapshots = {}
    manifest_bytes, manifest_identity = read_tracked(Path(manifest_path), snapshots)
    manifest = parse_json(manifest_bytes)
    if (
        manifest.get("version") != VERSION
        or manifest.get("verification") != "full"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError(
            "codec generation requires a full checkpoint identity manifest"
        )
    sources = reference_sources(snapshots)
    verified, contents = {}, {}
    for name in CHECKPOINT_FILES:
        data, actual = read_tracked(local_file(root, name), snapshots)
        if actual != manifest["files"].get(name):
            raise ValueError(f"codec checkpoint bytes differ from the manifest: {name}")
        contents[name], verified[name] = data, actual
    config = parse_json(contents["tokenizer_config.json"])
    if (
        config.get("add_bos_token") is not False
        or config.get("add_eos_token") is not False
    ):
        raise ValueError("codec tokenizer configuration must disable automatic BOS/EOS")
    if importlib.metadata.version("tokenizers") != TOKENIZERS_VERSION:
        raise ValueError("codec tokenizers dependency differs from its pinned version")

    import tokenizers

    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise ValueError(
            "loaded codec tokenizers dependency differs from its pinned version"
        )
    check_unchanged(snapshots)
    encoder = load_encoder(
        contents["encoding/encoding.py"], root / "encoding/encoding.py"
    )
    check_unchanged(snapshots)
    tokenizer = tokenizers.Tokenizer.from_buffer(contents["tokenizer.json"])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    check_unchanged(snapshots)
    cases = render_cases(encoder, tokenizer)
    result = {
        "schema_version": 1,
        "reference_kind": "local-python-prompt-codec-cpu",
        "checkpoint_manifest": manifest,
        "checkpoint_manifest_sha256": manifest_identity["sha256"],
        "verification_scope": "prompt-codec-files-only; no weight payload rescan",
        "verified_checkpoint_files": verified,
        "reference_sources": sources,
        "dependencies": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "tokenizers": tokenizers.__version__,
        },
        "tokenization": {
            "loader": "Tokenizer.from_buffer(verified_bytes)",
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        },
        "case_manifest_sha256": hashlib.sha256(
            json.dumps(fixture_cases(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "cases": cases,
    }
    encoded = json.dumps(
        result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    )
    # The exclusive link publishes a complete file without replacing a competing output.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent) as target:
        target.write(encoded + "\n")
        target.flush()
        check_unchanged(snapshots)
        os.link(target.name, output)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounded CPU prompt-codec fixtures"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", "--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.model, args.manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "case_count": len(result["cases"]),
                "token_count": sum(len(case["token_ids"]) for case in result["cases"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
