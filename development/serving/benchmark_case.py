from .benchmark_trace import (
    join_events,
    prompt_tokens,
    read_trace,
    trace_events,
    wait_for_completions,
)
from .qwen_fp8_evidence import VERSION, Blocked, digest
from .qwen_fp8_roofline import lower_bound as fp8_lower_bound
from .roofline import lower_bound
from .workload import PROFILE, validate_tools


def formal_reference(arguments):
    return getattr(arguments, "reference_version", None) == VERSION


def tool_evidence(response, tokenizer, formal):
    try:
        return validate_tools(response, tokenizer)
    except ValueError:
        if not formal:
            raise
        return None


def case_manifest(
    service,
    tokenizer,
    bodies,
    responses,
    sampling,
    length,
    output,
    started,
    finished,
    events,
):
    rows = []
    for index, response in enumerate(responses):
        request_id = int(response["id"].rsplit("_", 1)[1])
        row = dict(
            ordinal=index,
            id=request_id,
            response_id=response["id"],
            input_tokens=length,
            output_tokens=output,
            seed=index,
            body_sha256=digest(bodies[index]),
            body=bodies[index],
            tools=tool_evidence(
                response, tokenizer, formal_reference(service.arguments)
            ),
        )
        if formal_reference(service.arguments):
            try:
                row["prompt_token_ids"] = prompt_tokens(events, request_id)
            except Blocked:
                row["prompt_token_ids"] = None
        rows.append(row)
    return dict(
        artifact_id=service.arguments.artifact_id,
        mode=service.mode,
        sampling=sampling,
        workload=PROFILE,
        requests=rows,
        timer=dict(clock="perf_counter", started=started, finished=finished),
    )


def blocked(reason):
    return dict(
        accounting=VERSION,
        status="blocked",
        reasons=[str(reason)],
        seconds=None,
        efficiency=None,
    )


def case_bound(service, tokenizer, events, manifest, responses, rust_previous):
    rust_current = rust_previous
    if formal_reference(service.arguments):
        try:
            rust_current = wait_for_completions(
                service.rust_trace,
                rust_previous,
                {row["id"] for row in manifest["requests"]},
            )
            rust = trace_events(rust_previous, rust_current)
            events = join_events(events, rust, manifest, service.arguments.run_identity)
            bound = fp8_lower_bound(
                events,
                manifest,
                service.arguments.run_identity,
                responses,
                tokenizer=tokenizer,
            )
        except (Blocked, ValueError, KeyError, TypeError, IndexError, OSError) as error:
            bound = blocked(error)
    elif any(
        event.get("cache_layout", {}).get("kind") == "qwen_radix_v1"
        for event in events
        if event["operation"] == "identity"
    ):
        bound = blocked(
            "new Qwen cache layout requires explicit --reference-version " + VERSION
        )
    else:
        bound = lower_bound(events, manifest)
    return events, bound, rust_current


def archive_case(
    service,
    tokenizer,
    batch,
    length,
    output,
    bodies,
    responses,
    sampling,
    started,
    finished,
    previous,
    rust_previous,
):
    trace_error = None
    current = b""
    try:
        current = read_trace(service.trace)
        events = trace_events(previous, current)
    except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
        if not formal_reference(service.arguments):
            raise
        events, trace_error = [], error
    manifest = case_manifest(
        service,
        tokenizer,
        bodies,
        responses,
        sampling,
        length,
        output,
        started,
        finished,
        events,
    )
    new_reference = formal_reference(service.arguments)
    events, bound, rust_current = case_bound(
        service,
        tokenizer,
        events,
        manifest,
        responses,
        rust_previous,
    )
    if trace_error is not None:
        bound = blocked(trace_error)
    seconds = finished - started
    reference = bound["seconds"]
    result = dict(
        mode=service.mode,
        batch=batch,
        input_tokens=length,
        output_tokens=output,
        seconds=seconds,
        lower_bound=bound,
        manifest=manifest,
        manifest_id=digest(manifest),
        trace_id=digest(events),
        efficiency=reference / seconds if reference is not None else None,
        responses=responses,
    )
    if new_reference:
        result["trace_ranges"] = dict(
            worker=dict(file=service.trace.name, start=len(previous), end=len(current)),
            rust=dict(
                file=service.rust_trace.name,
                start=len(rust_previous),
                end=len(rust_current),
            ),
        )
    return result
