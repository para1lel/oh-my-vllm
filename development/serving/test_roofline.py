import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from . import benchmark
from .benchmark import (
    CONFIGURATIONS,
    comparison_passed,
    gate_result,
    select_configurations,
)
from .roofline import lower_bound


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
        for times in ((0, 1), (-1, 1), (1, float("nan")), (float("inf"), 1), (1, 1.06)):
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
            vendor = root / "vendor.so"
            vendor.write_bytes(b"CPU vendor mock")
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
                if name == "vision":
                    metadata["vision_vendor"] = dict(
                        images=[
                            dict(path=str(vendor), sha256=benchmark.file_sha256(vendor))
                        ]
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
                ("vendor", vendor),
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
            id="chatcmpl_1",
            usage=dict(
                prompt_tokens=16,
                completion_tokens=1,
                prompt_tokens_details=dict(cached_tokens=0),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            service = object.__new__(benchmark.Service)
            service.mode, service.port = "ordinary", 0
            service.trace = Path(directory) / "trace.jsonl"

            def run(batch=1):
                service.trace.write_text("\n".join(map(json.dumps, events)) + "\n")
                with (
                    patch.object(benchmark, "prompt", return_value=[]),
                    patch.object(benchmark, "post", return_value=response),
                    patch.object(
                        benchmark.time, "perf_counter", side_effect=(100.0, 102.0)
                    ),
                ):
                    return service.run(None, batch, 16, 1)

            result = run()
            self.assertEqual(result["seconds"], 2.0)
            self.assertEqual(result["lower_bound"], lower_bound(events, {1}, 1))
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


class Roofline(unittest.TestCase):
    def test_trace_rejects_mode_mismatches_duplicate_rows_and_bad_identity(self):
        identity = dict(operation="identity", config=dict(num_speculative_tokens=0))
        target = dict(
            operation="target",
            verify=False,
            requests=[dict(id=1, start=0, count=16, prompt=16)],
        )
        commit = dict(
            operation="commit",
            requests=[dict(id=1, tokens=1, accepted=0, proposed=0, position=16)],
        )
        self.assertGreater(
            lower_bound([identity, target, commit], {1}, 1)["seconds"], 0
        )
        for operation in (
            dict(operation="context", requests=[dict(id=1, start=1, count=16)]),
            dict(operation="draft", stage=0, requests=[dict(id=1, position=16)]),
        ):
            with (
                self.subTest(operation=operation["operation"]),
                self.assertRaisesRegex(ValueError, "ordinary trace"),
            ):
                lower_bound([identity, target, operation, commit], {1}, 1)
        with self.assertRaisesRegex(ValueError, "unsupported trace operation"):
            lower_bound(
                [identity, target, dict(operation="unknown", requests=[]), commit],
                {1},
                1,
            )
        for identities in (
            [],
            [identity, identity],
            [dict(operation="identity", config=dict(num_speculative_tokens=2))],
        ):
            with (
                self.subTest(identities=identities),
                self.assertRaisesRegex(ValueError, "identity"),
            ):
                lower_bound([*identities, target, commit], {1}, 1)
        mtp = dict(operation="identity", config=dict(num_speculative_tokens=4))
        context = dict(operation="context", requests=[dict(id=1, start=1, count=15)])
        self.assertGreater(
            lower_bound([mtp, target, context, commit], {1}, 1)["seconds"], 0
        )
        repeated = dict(
            operation="context",
            requests=[dict(id=1, start=1, count=7), dict(id=1, start=8, count=8)],
        )
        with self.assertRaisesRegex(ValueError, "repeats a request ID"):
            lower_bound([mtp, target, repeated, commit], {1}, 1)

    def test_prefill_chunks_keep_the_actual_operator_calls(self):
        identity = {"operation": "identity", "config": {"num_speculative_tokens": 0}}

        def target(first, count):
            return {
                "operation": "target",
                "verify": False,
                "requests": [{"id": 1, "start": first, "count": count, "prompt": 16}],
            }

        def commit(position):
            return {
                "operation": "commit",
                "requests": [
                    {
                        "id": 1,
                        "tokens": int(position == 16),
                        "accepted": 0,
                        "proposed": 0,
                        "position": position,
                    }
                ],
            }

        complete = [identity, target(0, 16), commit(16)]
        split = [identity]
        for first in range(0, 16, 4):
            split.extend([target(first, 4), commit(first + 4)])
        split.append(
            {
                "operation": "target",
                "verify": True,
                "requests": [{"id": 2, "start": 100000, "count": 20, "prompt": 16}],
            }
        )
        expected = lower_bound(complete, {1})
        actual = lower_bound(split, {1})
        self.assertGreater(actual["work"]["hbm_bytes"], expected["work"]["hbm_bytes"])
        self.assertEqual(actual["work"]["fp8_flops"], expected["work"]["fp8_flops"])
        self.assertGreater(actual["seconds"], expected["seconds"])
        self.assertEqual(
            [
                item["rows"]
                for item in actual["components"]
                if item["operation"] == "sample"
            ],
            [1],
        )
        self.assertGreater(actual["seconds"], 0)
        work = actual["work"]
        self.assertEqual(
            actual["seconds"],
            sum(item["seconds"] for item in actual["components"]),
        )
        self.assertGreaterEqual(actual["seconds"], work["hbm_bytes"] / 7.7e12)
        large = [
            identity,
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=1, start=0, count=32768, prompt=32768)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=1, tokens=1, accepted=0, proposed=0, position=32768)],
            ),
        ]
        large_bound = lower_bound(large, {1}, 1)
        self.assertGreater(
            large_bound["seconds"], large_bound["global_resource_seconds"]
        )
        with self.assertRaises(ValueError):
            lower_bound([*complete, target(0, 16)], {1})
        with self.assertRaises(ValueError):
            lower_bound(complete, {1}, expected_tokens=8192)
        mtp = [{"operation": "identity", "config": {"num_speculative_tokens": 4}}]
        mtp.append(
            {
                "operation": "target",
                "verify": False,
                "requests": [{"id": 1, "start": 0, "count": 4, "prompt": 4}],
            }
        )
        mtp.append(
            {"operation": "context", "requests": [{"id": 1, "start": 1, "count": 4}]}
        )
        mtp.extend(
            {
                "operation": "draft",
                "stage": stage,
                "requests": [{"id": 1, "position": 4 + stage}],
            }
            for stage in range(4)
        )
        mtp.append(
            {
                "operation": "commit",
                "requests": [
                    {"id": 1, "tokens": 1, "accepted": 0, "proposed": 4, "position": 4}
                ],
            }
        )
        with self.assertRaises(ValueError):
            lower_bound(mtp, {1}, expected_tokens=2)
        unused = {
            "operation": "target",
            "verify": True,
            "requests": [{"id": 1, "start": 4, "count": 1, "prompt": 4}],
        }
        with self.assertRaises(ValueError):
            lower_bound([*mtp, unused], {1}, expected_tokens=6)

    def test_actual_batches_are_not_combined_by_request_round(self):
        identity = {"operation": "identity", "config": {"num_speculative_tokens": 0}}
        targets = [dict(id=request, start=0, count=16, prompt=16) for request in (1, 2)]
        commits = [
            dict(id=request, tokens=1, accepted=0, proposed=0, position=16)
            for request in (1, 2)
        ]
        together = [
            identity,
            dict(operation="target", verify=False, requests=targets),
            dict(operation="commit", requests=commits),
        ]
        separate = [identity]
        for target, commit in zip(targets, commits, strict=True):
            separate.extend(
                [
                    dict(operation="target", verify=False, requests=[target]),
                    dict(operation="commit", requests=[commit]),
                ]
            )
        combined = lower_bound(together, {1, 2}, 1)
        split = lower_bound(separate, {1, 2}, 1)
        self.assertGreater(split["seconds"], combined["seconds"])
        self.assertEqual(
            [
                item["rows"]
                for item in split["components"]
                if item["operation"] == "sample"
            ],
            [1, 1],
        )
        ragged = [
            identity,
            dict(
                operation="target",
                verify=False,
                requests=[targets[0], dict(id=2, start=0, count=4, prompt=8)],
            ),
            dict(
                operation="commit",
                requests=[
                    commits[0],
                    dict(id=2, tokens=0, accepted=0, proposed=0, position=4),
                ],
            ),
            dict(
                operation="target",
                verify=False,
                requests=[dict(id=2, start=4, count=4, prompt=8)],
            ),
            dict(
                operation="commit",
                requests=[dict(id=2, tokens=1, accepted=0, proposed=0, position=8)],
            ),
        ]
        result = lower_bound(ragged, {1, 2}, 1)
        samples = [
            item for item in result["components"] if item["operation"] == "sample"
        ]
        self.assertEqual([item["request_ids"] for item in samples], [[1], [2]])
        attention = next(
            item for item in result["components"] if item["operation"] == "fa_attention"
        )
        self.assertEqual(attention["work_per_call"]["bf16_flops"], 4 * 146 * 24 * 256)

    def test_mtp_keeps_verification_rows_and_actual_proposals(self):
        events = [
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
        result = lower_bound(events, {1}, 4)
        self.assertEqual(
            result["acceptance"][1], dict(tokens=4, accepted=2, proposed=2)
        )
        drafts = [
            item for item in result["components"] if item["operation"] == "mtp_draft"
        ]
        self.assertEqual([item["queries"] for item in drafts], [[(4, 1)], [(5, 1)]])
        self.assertEqual(
            drafts[0]["work_per_call"]["bf16_flops"],
            2 * 248320 * 5120 + 4 * 4 * 24 * 256,
        )
        self.assertEqual(
            drafts[1]["work_per_call"]["bf16_flops"],
            2 * 248320 * 5120 + 2 * 5120 * 10240 + 4 * 5 * 24 * 256,
        )
        samples = [
            item for item in result["components"] if item["operation"] == "sample"
        ]
        self.assertEqual([item["rows"] for item in samples], [1, 3])
        self.assertEqual(samples[-1]["work_per_call"]["fp64_flops"], 6)
