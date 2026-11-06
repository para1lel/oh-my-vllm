import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import benchmark


def _run_http_fixture(self, service, events, response, batch=1):
    service.trace.write_text(json.dumps(events[0]) + "\n")

    def respond(port, body):
        self.assertEqual(body["tool_choice"], "auto")
        self.assertTrue(body["tools"])
        self.assertIn(body["seed"], range(batch))
        for name in (
            "temperature",
            "top_k",
            "top_p",
            "presence_penalty",
            "repetition_guard",
        ):
            self.assertNotIn(name, body)
        with service.trace.open("a") as trace:
            trace.write("\n".join(map(json.dumps, events[1:])) + "\n")
        return response

    with (
        patch.object(benchmark, "prompt", return_value=[]),
        patch.object(benchmark, "post", side_effect=respond),
        patch.object(benchmark.time, "perf_counter", side_effect=(100.0, 102.0)),
    ):
        return service.run(
            SimpleNamespace(generation=dict(temperature=1.0, top_k=20, top_p=0.95)),
            batch,
            16,
            1,
        )


def _invoke_benchmark(root, binary, service, artifacts, *extra):
    arguments = [
        "benchmark",
        "--model",
        str(root),
        "--artifacts",
        str(artifacts),
        "--binary",
        str(binary),
        *extra,
    ]
    with (
        patch.object(benchmark, "selected_uuid", return_value="GPU-test"),
        patch.object(
            benchmark,
            "device_identity",
            return_value={"name": "NVIDIA B200"},
        ),
        patch.object(benchmark, "Service", side_effect=service),
        patch.object(benchmark, "Tokenization"),
        patch.object(benchmark, "source_identity", return_value="source"),
        patch.object(benchmark, "cache_directory", return_value=root),
        patch.object(benchmark.subprocess, "check_output", return_value="CPU mock"),
        patch("builtins.print"),
        patch("sys.argv", arguments),
    ):
        return benchmark.main()


def tokenizer_fixture(root):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    codec = Tokenizer(models.BPE({"x": 0, "[EOS]": 1, "</think>": 2}, []))
    codec.add_special_tokens(["[EOS]", "</think>"])
    codec.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    codec.decoder = decoders.ByteLevel()
    (root / "tokenizer.json").write_text(codec.to_str())
    (root / "generation_config.json").write_text('{"eos_token_id":1}')
    (root / "chat_template.jinja").write_text('{{ messages[0]["content"] }}')


def _write_benchmark_inputs(root):
    binary = root / "service"
    binary.write_bytes(b"CPU mock")
    vendor_files = {}
    for name in ("vision",):
        image, header = root / f"{name}-vendor.so", root / f"{name}-vendor.h"
        image.write_bytes(b"CPU vendor image mock")
        header.write_bytes(b"CPU vendor header mock")
        vendor_files[name] = (image, header)
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"CPU weights mock")
    (root / "config.json").write_text("{}")
    for name in (
        "tokenizer.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        (root / name).write_text("CPU tokenizer asset mock")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}})
    )
    for name in benchmark.NATIVE_MODULES:
        library = root / f"{name}.so"
        library.write_bytes(f"CPU {name} mock".encode())
        metadata = dict(library=str(library), binary_id=benchmark.file_sha256(library))
        if name in vendor_files:
            image, header = vendor_files[name]
            metadata["vision_vendor"] = dict(
                images=[dict(path=str(image), sha256=benchmark.file_sha256(image))],
                headers={str(header): benchmark.file_sha256(header)},
            )
        (root / f"{name}.json").write_text(json.dumps(metadata))
    return binary, vendor_files, shard


def _verify_checkpoint_identity(self, events):
    checkpoint = {
        "config": "config",
        "index": "index",
        "weights.safetensors": "weights",
    }
    identity = events[0]
    identity["checkpoint"] = dict(checkpoint)
    identity["config"]["model_path"] = "/model"
    expected = dict(num_speculative_tokens=0, model_path="/model")
    benchmark.verify_trace_identity(events, checkpoint, expected)
    for name in checkpoint:
        identity["checkpoint"][name] = "changed"
        with (
            self.subTest(checkpoint=name),
            self.assertRaisesRegex(ValueError, "checkpoint"),
        ):
            benchmark.verify_trace_identity(events, checkpoint, expected)
        identity["checkpoint"][name] = checkpoint[name]
    identity["checkpoint"] = {"config": "config", "index": "index"}
    with self.assertRaisesRegex(ValueError, "checkpoint"):
        benchmark.verify_trace_identity(events, checkpoint, expected)
    identity["checkpoint"] = dict(checkpoint)
    identity["config"]["model_path"] = "/other"
    with self.assertRaisesRegex(ValueError, "configuration"):
        benchmark.verify_trace_identity(events, checkpoint, expected)
    identity["config"]["model_path"] = "/model"
    with self.assertRaisesRegex(ValueError, "exactly one identity"):
        benchmark.verify_trace_identity([identity, *events], checkpoint, expected)
    return checkpoint, identity, expected


def _verify_tokenizer_identity(self, directory, events, identity, checkpoint, expected):
    root = Path(directory)
    tokenizer_fixture(root)
    tokenizer = benchmark.tokenizer_identity(root)
    loaded = benchmark.Tokenization(root)
    loaded.verify_identity(tokenizer)
    for actual in (None, dict(tokenizer, **{"chat_template.jinja": "changed"})):
        if actual is None:
            identity.pop("tokenizer", None)
        else:
            identity["tokenizer"] = actual
        with (
            self.subTest(tokenizer=actual),
            self.assertRaisesRegex(ValueError, "loaded tokenizer assets"),
        ):
            benchmark.verify_trace_identity(events, checkpoint, expected, tokenizer)
    identity["tokenizer"] = loaded.identity
    benchmark.verify_trace_identity(events, checkpoint, expected, tokenizer)
    template = root / "chat_template.jinja"
    original = template.read_bytes()
    template.write_bytes(original + b"x")
    changed = benchmark.Tokenization(root)
    request = {"messages": [{"role": "user", "content": "x"}], "effort": "off"}
    self.assertNotEqual(loaded.prepare(request), changed.prepare(request))
    template.write_bytes(original)
    self.assertEqual(benchmark.tokenizer_identity(root), tokenizer)
    identity["tokenizer"] = changed.identity
    with self.assertRaisesRegex(ValueError, "loaded tokenizer assets"):
        benchmark.verify_trace_identity(events, checkpoint, expected, tokenizer)


def request_manifest(ids=(1,), length=16, output=1, mode="ordinary"):
    return dict(
        mode=mode,
        requests=[
            dict(
                ordinal=index,
                id=key,
                response_id=f"chatcmpl_744872_6ad586dc_{key}",
                input_tokens=length,
                output_tokens=output,
            )
            for index, key in enumerate(ids)
        ],
    )


def prefill_trace(ids=(1,), length=16, chunk=16):
    events = [dict(operation="identity", config=dict(num_speculative_tokens=0))]
    for key in ids:
        for start in range(0, length, chunk):
            count = min(chunk, length - start)
            events.extend(
                [
                    dict(
                        operation="target",
                        verify=False,
                        requests=[
                            dict(id=key, start=start, count=count, prompt=length)
                        ],
                    ),
                    dict(
                        operation="commit",
                        requests=[
                            dict(
                                id=key,
                                tokens=int(start + count == length),
                                accepted=0,
                                proposed=0,
                                position=start + count,
                            )
                        ],
                    ),
                ]
            )
    return events


def _verification_trace():
    mtp = [
        dict(operation="identity", config=dict(num_speculative_tokens=4)),
        dict(
            operation="target",
            verify=False,
            requests=[dict(id=1, start=0, count=4, prompt=4)],
        ),
        dict(operation="context", requests=[dict(id=1, start=1, count=4)]),
        dict(operation="draft", stage=0, requests=[dict(id=1, position=4)]),
        dict(operation="draft", stage=1, requests=[dict(id=1, position=5)]),
        dict(
            operation="commit",
            requests=[dict(id=1, tokens=1, accepted=0, proposed=2, position=4)],
        ),
        dict(
            operation="target",
            verify=True,
            requests=[dict(id=1, start=4, count=3, prompt=4)],
        ),
        dict(operation="context", requests=[dict(id=1, start=5, count=2)]),
        dict(
            operation="commit",
            requests=[dict(id=1, tokens=3, accepted=2, proposed=0, position=7)],
        ),
    ]
    return mtp
