"""Replay one Qwen HTTP archive and check all 24 cases and 12 efficiency gates."""

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from oh_my_vllm.models.qwen38.cache_budget import cache_layout
from oh_my_vllm.models.qwen38.tokenizer import Tokenization

from .accounting import RATES
from .benchmark_options import CONFIGURATIONS
from .benchmark_trace import join_events, trace_events
from .qwen_fp8_evidence import VERSION, digest, require
from .qwen_fp8_identity import calculator_identity
from .qwen_fp8_roofline import lower_bound, meets_efficiency_target

SCOPE = "qwen-fp8-http-24-long-12"
ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError)
STREAMS = {"worker": "execution.jsonl", "rust": "rust-execution.jsonl"}


@dataclass(frozen=True, order=True)
class CaseKey:
    mode: str
    batch: int
    input_tokens: int
    output_tokens: int

    @classmethod
    def parse(cls, value):
        require(value["mode"] in ("ordinary", "mtp4"), "unknown HTTP mode")
        for name in ("batch", "input_tokens", "output_tokens"):
            require(type(value[name]) is int and value[name] > 0, f"invalid {name}")
        return cls(*(value[name] for name in cls.__dataclass_fields__))

    @property
    def filename(self):
        return f"{self.mode}-{self.batch}-{self.input_tokens}-{self.output_tokens}.json"


@dataclass(frozen=True)
class LoadedCase:
    key: CaseKey
    result_path: Path
    artifact_id: str
    manifest: dict
    responses: list
    events: list
    stored_result: dict
    reference: dict
    http_seconds: float
    efficiency: float


@dataclass(frozen=True)
class Result:
    status: str
    artifact_id: str
    observed_cases: int
    observed_long_cases: int
    coverage_complete: bool
    rows: tuple[LoadedCase, ...]
    missing: tuple[CaseKey, ...]
    duplicates: tuple[CaseKey, ...]
    extra: tuple[CaseKey, ...]
    failed_threshold_cases: tuple[CaseKey, ...]
    errors: tuple[str, ...]
    scope: str = SCOPE
    accounting_version: str = VERSION
    required_cases: int = 24
    required_long_cases: int = 12
    threshold: float = 0.40

    @property
    def target_passed(self):
        return self.status == "passed"

    def report(self):
        fields = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in ("rows", "missing", "duplicates", "extra", "failed_threshold_cases")
        }
        fields.update(target_passed=self.target_passed)
        for name in ("missing", "duplicates", "extra", "failed_threshold_cases"):
            fields[name] = [asdict(key) for key in getattr(self, name)]
        fields["rows"] = [
            dict(
                **asdict(row.key),
                result_file=row.result_path.name,
                artifact_id=row.artifact_id,
                http_seconds=row.http_seconds,
                reference_seconds=row.reference["seconds"],
                efficiency=row.efficiency,
            )
            for row in self.rows
        ]
        return fields


def _object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _finite(value):
    number = float(value)
    require(math.isfinite(number), "nonfinite JSON number")
    return number


def _json(data):
    return json.loads(
        data, object_pairs_hook=_object, parse_float=_finite, parse_constant=_finite
    )


def _canonical(value):
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _inventory(root, errors):
    records = []
    metadata = {"identity.json", "summary.json"} | {
        f"{mode}-tool-check.json" for mode in ("ordinary", "mtp4")
    }
    for path in sorted(root.glob("*.json")):
        if path.name in metadata:
            continue
        try:
            value = _json(path.read_bytes())
            key = CaseKey.parse(value)
            records.append((path, key, value))
            require(path.name == key.filename, "case filename differs from its key")
        except ERRORS as error:
            errors.append(f"{path.name}: {error}")
    return records


def _run_identity(root, tokenizer_dir):
    artifacts = _json((root / "identity.json").read_bytes())
    accounting = artifacts["accounting"]
    require(accounting["version"] == VERSION, "requires the formal FP8 reference")
    require(accounting["rates"] == RATES, "accounting rates differ")
    require(
        accounting.get("calculator") == calculator_identity(),
        "calculator content binding is missing or differs",
    )
    try:
        tokenizer = Tokenization(tokenizer_dir)
        tokenizer.verify_identity(artifacts["tokenizer"])
    except Exception as error:
        raise ValueError(f"invalid tokenizer assets: {error}") from error
    return artifacts, tokenizer


def _raw_streams(root, modes):
    streams = {}
    for mode in modes:
        for stream, suffix in STREAMS.items():
            path = root / f"{mode}-{suffix}"
            require(path.resolve().parent == root, "raw trace escapes the archive")
            raw = path.read_bytes()
            require(raw.endswith(b"\n"), "raw trace has an incomplete final line")
            for line in raw.splitlines():
                event = _json(line)
                require(isinstance(event, dict), "raw trace event is not an object")
            streams[mode, stream] = raw
    return streams


def _case_events(key, value, streams):
    events = {}
    for stream, suffix in STREAMS.items():
        interval = value["trace_ranges"][stream]
        require(interval["file"] == f"{key.mode}-{suffix}", "raw trace mode differs")
        start, end = interval["start"], interval["end"]
        raw = streams[key.mode, stream]
        require(
            type(start) is int and type(end) is int and 0 <= start < end <= len(raw),
            "invalid raw trace range",
        )
        require(
            (start == 0 or raw[start - 1 : start] == b"\n")
            and raw[end - 1 : end] == b"\n",
            "raw trace range cuts a JSONL line",
        )
        events[stream] = trace_events(raw[:start], raw[:end])
    return events


def _loaded_configuration(identity, artifacts, mode):
    requested, actual = artifacts["config"], identity["config"]
    require(
        type(requested["mamba_blocks"]) is int
        and requested["mamba_blocks"] > 0
        and actual["mamba_blocks"] == requested["mamba_blocks"],
        "loaded checkpoint blocks differ from the request",
    )
    for name in ("mamba_blocks", "num_gpu_blocks", "num_speculative_tokens"):
        require(type(actual[name]) is int, f"invalid loaded {name}")
    require(actual["model_path"] == requested["model"], "loaded model path differs")
    require(
        actual["num_speculative_tokens"] == (4 if mode == "mtp4" else 0),
        "loaded speculation differs from HTTP mode",
    )
    blocks = requested["blocks"]
    require(
        blocks is None or (type(blocks) is int and blocks == actual["num_gpu_blocks"]),
        "loaded KV blocks differ from the request",
    )
    expected = cache_layout(actual).wire()
    require(
        all(
            identity["cache_memory"].get(name) == value
            for name, value in expected.items()
        ),
        "loaded cache layout differs from its configured pools",
    )


def _shape(key, value):
    require(
        value.get("status") in (None, "completed"), "stored HTTP case did not complete"
    )
    require(
        value["lower_bound"]["status"] == "calculated", "stored bound is not calculated"
    )
    manifest, responses = value["manifest"], value["responses"]
    require(manifest["mode"] == key.mode, "manifest mode differs")
    require(
        len(manifest["requests"]) == len(responses) == key.batch,
        "HTTP batch inventory differs",
    )
    for row in manifest["requests"]:
        stream = row["body"].get("stream", False)
        require(
            type(stream) is bool and not stream, "requires nonstreamed HTTP responses"
        )
        require(
            all(
                type(row[name]) is int and row[name] == getattr(key, name)
                for name in ("input_tokens", "output_tokens")
            ),
            "manifest request shape differs",
        )
    return manifest, responses


def _http_outputs(events, manifest, responses):
    completed = {}
    for event in events:
        if event["operation"] == "complete":
            for row in event["requests"]:
                identifier = row["id"]
                require(
                    type(identifier) is int and identifier not in completed,
                    "duplicate complete request",
                )
                completed[identifier] = row["response"]
    require(
        completed.keys() == {row["id"] for row in manifest["requests"]},
        "complete response inventory differs",
    )
    mapped = {response["id"]: response for response in responses}
    for row in manifest["requests"]:
        response, prepared = mapped[row["response_id"]], completed[row["id"]]
        require(response.get("error") is None, "HTTP response contains an error")
        choices = response["choices"]
        require(
            len(choices) == 1 and choices[0].get("error") is None,
            "invalid HTTP choices",
        )
        message = choices[0]["message"]
        require(
            message.get("role") == "assistant" and "content" in message,
            "missing assistant output",
        )
        require(prepared["id"] == row["response_id"], "prepared response ID differs")
        require(
            _canonical(prepared) == _canonical(response),
            "HTTP response differs from Rust completion",
        )


def _load_case(path, key, value, artifacts, tokenizer, streams):
    manifest, responses = _shape(key, value)
    raw = _case_events(key, value, streams)
    events = join_events(raw["worker"], raw["rust"], manifest, artifacts)
    _loaded_configuration(events[0], artifacts, key.mode)
    reference = lower_bound(events, manifest, artifacts, responses, tokenizer=tokenizer)
    require(
        reference["status"] == "calculated" and reference["accounting"] == VERSION,
        f"replayed bound is blocked: {reference.get('reasons')}",
    )
    _http_outputs(events, manifest, responses)
    for name in ("seconds", "http_seconds", "efficiency"):
        number = reference[name]
        require(
            type(number) in (int, float) and math.isfinite(number) and number > 0,
            f"invalid recalculated {name}",
        )
    seconds = manifest["timer"]["finished"] - manifest["timer"]["started"]
    require(
        type(value["seconds"]) in (int, float)
        and type(value["efficiency"]) in (int, float)
        and value["seconds"] == seconds == reference["http_seconds"]
        and value["efficiency"] == reference["efficiency"],
        "stored HTTP seconds or efficiency differs from replay",
    )
    for name, evidence in (("manifest_id", manifest), ("trace_id", events)):
        require(
            value[name] == reference[name] == digest(evidence),
            f"stored {name} differs from raw evidence",
        )
    require(
        _canonical(value["lower_bound"]) == _canonical(reference),
        "stored complete bound differs from recomputation",
    )
    return LoadedCase(
        key,
        path,
        digest(artifacts),
        manifest,
        responses,
        events,
        value,
        reference,
        seconds,
        reference["efficiency"],
    )


def _consistent_rows(rows):
    identities, ranges, executions = {}, defaultdict(list), set()
    for row in rows:
        mode = row.key.mode
        loaded = _canonical(row.events[0])
        require(
            identities.setdefault(mode, loaded) == loaded,
            "loaded identity changes within mode",
        )
        for stream in STREAMS:
            interval = row.stored_result["trace_ranges"][stream]
            ranges[mode, stream].append((interval["start"], interval["end"]))
        for event in row.events:
            if event["operation"] != "accounting_call":
                continue
            key = (mode, event["worker_epoch"], event["step_id"])
            require(key not in executions, "execution evidence is reused within mode")
            executions.add(key)
    for intervals in ranges.values():
        ordered = sorted(intervals)
        require(
            all(first[1] <= second[0] for first, second in pairwise(ordered)),
            "formal case raw ranges overlap",
        )


def evaluate_archive(archive_dir, *, tokenizer_dir):
    root = Path(archive_dir).resolve()
    errors, rows, artifact_id = [], [], ""
    records = _inventory(root, errors)
    expected = {CaseKey(*case) for case in CONFIGURATIONS}
    counts = Counter(key for _, key, _ in records)
    missing = tuple(sorted(expected - counts.keys()))
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    extra = tuple(sorted(counts.keys() - expected))
    coverage = counts == Counter(expected)
    if not coverage:
        errors.append("the 24-case HTTP matrix is missing, duplicated or extra")
    try:
        require(
            len(expected) == 24
            and sum(key.output_tokens == 8192 for key in expected) == 12,
            "the frozen HTTP matrix differs",
        )
        artifacts, tokenizer = _run_identity(root, tokenizer_dir)
        artifact_id = digest(artifacts)
        if (root / "summary.json").exists():
            summary = _json((root / "summary.json").read_bytes())
            if "configurations" in summary:
                count = summary["configurations"]
                require(
                    type(count) is int and count == len(records),
                    "summary count differs",
                )
        streams = _raw_streams(root, {key.mode for _, key, _ in records})
        for path, key, value in records:
            try:
                rows.append(_load_case(path, key, value, artifacts, tokenizer, streams))
            except ERRORS as error:
                errors.append(f"{path.name}: {error}")
        _consistent_rows(rows)
        tokenizer.verify_identity(artifacts["tokenizer"])
        require(
            artifacts["accounting"]["calculator"] == calculator_identity(),
            "calculator changed during replay",
        )
    except ERRORS as error:
        errors.append(str(error))
    failed = tuple(
        sorted(
            row.key
            for row in rows
            if row.key.output_tokens == 8192
            and not meets_efficiency_target(row.reference["seconds"], row.http_seconds)
        )
    )
    status = "invalid" if errors else "failed" if failed else "passed"
    return Result(
        status,
        artifact_id,
        len(records),
        sum(key.output_tokens == 8192 for _, key, _ in records),
        coverage,
        tuple(rows),
        missing,
        duplicates,
        extra,
        failed,
        tuple(errors),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.archive.resolve()
    if arguments.output and arguments.output.resolve().is_relative_to(root):
        parser.error("write the report outside the read-only archive")
    result = evaluate_archive(root, tokenizer_dir=arguments.tokenizer)
    text = json.dumps(result.report(), indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(text)
    print(text, end="")
    return 0 if result.target_passed else 2 if result.status == "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
