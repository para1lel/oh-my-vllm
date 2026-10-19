import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import benchmark
from .accounting import draft_work, reference_seconds, reference_work, sample_work
from .benchmark import (
    CONFIGURATIONS,
    comparison_passed,
    gate_result,
    select_configurations,
)
from .roofline import lower_bound
from .trace import canonical_passes


def tokenizer_fixture(root):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    codec = Tokenizer(models.BPE({"x": 0, "[EOS]": 1, "</think>": 2}, []))
    codec.add_special_tokens(["[EOS]", "</think>"])
    codec.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    codec.decoder = decoders.ByteLevel()
    (root / "tokenizer.json").write_text(codec.to_str())
    (root / "generation_config.json").write_text('{"eos_token_id":1}')
    (root / "chat_template.jinja").write_text('{{ messages[0]["content"] }}')


class BenchmarkGate(unittest.TestCase):
    def test_output_lengths_and_complete_comparison(self):
        self.assertEqual(len(CONFIGURATIONS), 24)
        self.assertEqual(select_configurations(None, (64, 8192)), CONFIGURATIONS)
        self.assertEqual(select_configurations(None, (8192,)), CONFIGURATIONS[1::2])
        self.assertEqual(
            select_configurations(["mtp4:4:131072"], (64, 8192)),
            [("mtp4", 4, 131072, 64), ("mtp4", 4, 131072, 8192)],
        )
        self.assertEqual(
            select_configurations(["ordinary:1:32768:64"], (64, 8192)),
            [("ordinary", 1, 32768, 64)],
        )
        results = [
            dict(
                mode=mode,
                batch=batch,
                input_tokens=length,
                output_tokens=output,
                **gate_result((mode, batch, length, output), 2.0, 1.0),
            )
            for mode, batch, length, output in CONFIGURATIONS
        ]
        self.assertTrue(comparison_passed(results))
        self.assertFalse(comparison_passed(results[:-1]))
        self.assertFalse(comparison_passed([*results[:-1], results[0]]))
        self.assertFalse(comparison_passed([*results, results[0]]))
        self.assertFalse(
            comparison_passed([*results[:-1], dict(results[-1], passed=False)])
        )
        self.assertFalse(gate_result(CONFIGURATIONS[0], 2.01, 1.0)["passed"])
        self.assertFalse(gate_result(("ordinary", 1, 32768, 32), 2.0, 1.0)["passed"])
        self.assertTrue(gate_result(CONFIGURATIONS[0], 1, 1.06)["passed"])
        for times in ((0, 1), (-1, 1), (1, float("nan")), (float("inf"), 1)):
            with self.assertRaises(ValueError):
                gate_result(CONFIGURATIONS[0], *times)
        for selected, outputs in (
            (None, (64, 64)),
            (None, ()),
            (None, (0,)),
            (["ordinary:1:32768"] * 2, (64, 8192)),
            (["ordinary:1:32768:8192", "mtp4:4:131072"], (64,)),
            (["ordinary:3:32768"], (64, 8192)),
        ):
            with self.assertRaises(ValueError):
                select_configurations(selected, outputs)

    def test_full_command_keeps_all_outputs_and_requires_every_configuration(self):
        calls, services = [], []
        mutation = None

        def service(arguments, mode):
            def run(tokenizer, batch, length, output):
                calls.append((mode, batch, length, output))
                return dict(
                    mode=mode,
                    batch=batch,
                    input_tokens=length,
                    output_tokens=output,
                    seconds=2.0,
                    **gate_result((mode, batch, length, output), 2.0, 1.0),
                )

            def close():
                if mutation is not None:
                    mutation.write_bytes(b"changed CPU mock")

            instance = Mock(run=Mock(side_effect=run), close=Mock(side_effect=close))
            services.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "service"
            binary.write_bytes(b"CPU mock")
            vendor_files = {}
            for name in ("vision", "ffn", "gdn", "fa"):
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
                metadata = dict(
                    library=str(library), binary_id=benchmark.file_sha256(library)
                )
                if name in vendor_files:
                    image, header = vendor_files[name]
                    key = {
                        "vision": "vision_vendor",
                        "ffn": "ffn_mx_vendor",
                        "gdn": "gdn_mx_vendor",
                        "fa": "fa_mx_vendor",
                    }[name]
                    metadata[key] = dict(
                        images=[
                            dict(path=str(image), sha256=benchmark.file_sha256(image))
                        ],
                        headers={str(header): benchmark.file_sha256(header)},
                    )
                (root / f"{name}.json").write_text(json.dumps(metadata))

            def invoke(artifacts, *extra):
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
                    patch.object(
                        benchmark.subprocess, "check_output", return_value="CPU mock"
                    ),
                    patch("builtins.print"),
                    patch("sys.argv", arguments),
                ):
                    return benchmark.main()

            self.assertEqual(invoke(root / "complete"), 0)
            self.assertEqual(invoke(root / "partial", "--case", "ordinary:1:32768"), 1)
            measured = [case for case in calls if case[2] != 64]
            self.assertEqual(measured[:24], CONFIGURATIONS)
            self.assertEqual(measured[24:], CONFIGURATIONS[:2])
            for mode, batch, length, output in CONFIGURATIONS:
                result = json.loads(
                    (
                        root / "complete" / f"{mode}-{batch}-{length}-{output}.json"
                    ).read_text()
                )
                self.assertEqual(result["output_tokens"], output)
            self.assertEqual(
                json.loads((root / "complete/summary.json").read_text()),
                dict(passed=True, configurations=24),
            )
            self.assertEqual(
                json.loads((root / "partial/summary.json").read_text()),
                dict(passed=False, configurations=2),
            )
            for instance in services:
                instance.ready.assert_called_once()
                instance.close.assert_called_once()
                self.assertEqual(instance.verify_identity.call_count, 2)
            for label, target in (
                ("binary", binary),
                ("native", root / "ffn.so"),
                ("vision-vendor", vendor_files["vision"][0]),
                ("vision-header", vendor_files["vision"][1]),
                ("ffn-vendor", vendor_files["ffn"][0]),
                ("ffn-header", vendor_files["ffn"][1]),
                ("gdn-vendor", vendor_files["gdn"][0]),
                ("gdn-header", vendor_files["gdn"][1]),
                ("fa-vendor", vendor_files["fa"][0]),
                ("fa-header", vendor_files["fa"][1]),
                ("checkpoint", shard),
                ("tokenizer", root / "tokenizer.json"),
                ("generation", root / "generation_config.json"),
                ("template", root / "chat_template.jinja"),
            ):
                original = target.read_bytes()
                mutation = target
                with self.subTest(artifact=label), self.assertRaises(RuntimeError):
                    invoke(root / f"changed-{label}")
                target.write_bytes(original)
                mutation = None
                self.assertFalse((root / f"changed-{label}/summary.json").exists())
            with (
                patch.object(
                    benchmark, "cache_directory", return_value=root / "missing"
                ),
                self.assertRaises(FileNotFoundError),
            ):
                benchmark.native_identity()

    def test_http_measurement_checks_trace_identity_lengths_and_request_ids(self):
        events = [
            dict(operation="identity", config=dict(num_speculative_tokens=0)),
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=1, start=0, count=16, prompt=16)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=1, accepted=0, proposed=0, position=16)],
            ),
        ]
        response = dict(
            id="chatcmpl_744872_6ad586dc_1",
            usage=dict(
                prompt_tokens=16,
                completion_tokens=1,
                prompt_tokens_details=dict(cached_tokens=0),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            service = object.__new__(benchmark.Service)
            service.arguments = SimpleNamespace(artifact_id="test-artifact")
            service.mode, service.port = "ordinary", 0
            service.trace = Path(directory) / "trace.jsonl"

            def run(batch=1):
                service.trace.write_text(json.dumps(events[0]) + "\n")

                def respond(port, body):
                    with service.trace.open("a") as trace:
                        trace.write("\n".join(map(json.dumps, events[1:])) + "\n")
                    return response

                with (
                    patch.object(benchmark, "prompt", return_value=[]),
                    patch.object(benchmark, "post", side_effect=respond),
                    patch.object(
                        benchmark.time, "perf_counter", side_effect=(100.0, 102.0)
                    ),
                ):
                    return service.run(None, batch, 16, 1)

            result = run()
            self.assertEqual(result["seconds"], 2.0)
            self.assertEqual(
                result["lower_bound"], lower_bound(events, result["manifest"])
            )
            self.assertEqual(
                result["manifest_id"], benchmark.json_digest(result["manifest"])
            )
            self.assertEqual(result["manifest"]["artifact_id"], "test-artifact")
            self.assertFalse(result["passed"])
            with self.assertRaisesRegex(ValueError, "distinct request IDs"):
                run(batch=2)
            events[0]["config"]["num_speculative_tokens"] = 4
            with self.assertRaisesRegex(ValueError, "service mode"):
                run()
            events[0]["config"]["num_speculative_tokens"] = 0
            events[1]["requests"][0]["prompt"] = 17
            with self.assertRaisesRegex(ValueError, "prompt length"):
                run()
            events[1]["requests"][0]["prompt"] = 16
            response["usage"]["completion_tokens"] = 2
            with self.assertRaisesRegex(ValueError, "token count"):
                run()
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
                benchmark.verify_trace_identity(
                    [identity, *events], checkpoint, expected
                )

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
                    benchmark.verify_trace_identity(
                        events, checkpoint, expected, tokenizer
                    )
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


class Roofline(unittest.TestCase):
    def test_accounting_rejects_inflation_and_unbound_or_fabricated_work(self):
        manifest = request_manifest((9, 2), length=32768)
        complete = prefill_trace((9, 2), length=32768, chunk=32768)
        expected = lower_bound(complete, manifest)
        for order, chunk in (((2, 9), 4096), ((9, 2), 784), ((2, 9), 32768)):
            events = prefill_trace(order, length=32768, chunk=chunk)
            for event in events:
                event.update(tiles=100000, padded_rows=65536, copied_state_bytes=10**12)
            with self.subTest(order=order, chunk=chunk):
                self.assertEqual(lower_bound(events, manifest), expected)
        passes = canonical_passes(complete, manifest)
        self.assertEqual(
            [[row["count"] for row in item["requests"]] for item in passes],
            [[16464, 15680], [16304, 16464], [624]],
        )
        self.assertEqual([row["id"] for row in passes[0]["requests"]], [9, 2])
        for mutate in (
            lambda value: value["requests"].append(value["requests"][0]),
            lambda value: value["requests"].pop(),
            lambda value: value["requests"][0].update(
                id=100, response_id="chatcmpl_100"
            ),
            lambda value: value["requests"][0].update(response_id="chatcmpl_100"),
            lambda value: value["requests"][0].update(output_tokens=2),
            lambda value: value["requests"][0].update(input_tokens=32767),
            lambda value: value.update(mode="mtp4"),
        ):
            changed = deepcopy(manifest)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                lower_bound(complete, changed)
        for events in (
            complete[1:],
            [complete[0], *complete],
            complete[:-1],
            [*complete, dict(operation="unknown")],
        ):
            with self.subTest(events=events[-1]), self.assertRaises(ValueError):
                lower_bound(events, manifest)
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
        manifest = request_manifest(length=4, output=4, mode="mtp4")
        result = lower_bound(mtp, manifest)
        self.assertEqual(
            result["acceptance"][1], dict(tokens=4, accepted=2, proposed=2)
        )
        self.assertEqual(
            [
                row["context_count"]
                for item in canonical_passes(mtp, manifest)
                for row in item["requests"]
            ],
            [4, 2],
        )
        for index, field, value in (
            (5, "proposed", 4),
            (6, "count", 5),
            (8, "position", 6),
            (8, "accepted", 3),
            (3, "position", 3),
        ):
            changed = deepcopy(mtp)
            changed[index]["requests"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                lower_bound(changed, manifest)
        with self.assertRaises(ValueError):
            lower_bound([*mtp[:2], *mtp[3:5], mtp[2], *mtp[5:]], manifest)

    def test_reference_resources_share_tensor_capacity_and_sum_before_maximum(self):
        self.assertEqual(sample_work(1)["fp32_flops"], 25602)
        self.assertEqual(draft_work([dict(end=4)], 0)["fp32_flops"], 147286)
        work = dict(
            hbm_bytes=7.7e12,
            fp8_flops=4.5e15,
            bf16_flops=2.25e15,
            tf32_flops=1.1e15,
            fp32_flops=75e12,
            fp64_flops=37e12,
        )
        self.assertEqual(reference_seconds(work), 3)
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference_seconds(dict(work, fp8_flops=value))
        events = prefill_trace(length=32768, chunk=4096)
        manifest = request_manifest(length=32768)
        passes = canonical_passes(events, manifest)
        bound = lower_bound(events, manifest)
        self.assertEqual(bound["seconds"], reference_seconds(reference_work(passes)))
        # A complete prompt fits the budget without a checkpoint-aligned split.
        self.assertEqual([item["requests"][0]["count"] for item in passes], [32768])
        small = canonical_passes(prefill_trace(length=16), request_manifest())
        self.assertEqual(
            reference_work(small)["fp8_flops"],
            2
            * 16
            * (
                48 * (16384 * 5120 + 5120 * 6144)
                + 16 * (14336 * 5120 + 5120 * 6144)
                + 64 * (34816 * 5120 + 5120 * 17408)
            ),
        )
