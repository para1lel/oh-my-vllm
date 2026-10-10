"""Actual read files bind citations; reported ranges supply diagnostics."""

import pytest
from test_speculative_service_evidence import agent_evidence

from benchmarks.agentic_soak import agentic
from benchmarks.agentic_soak_analysis import source_citations


def reads():
    return [
        {"path": "crates/scheduler/src/lib.rs", "line_range": [240, 330]},
        {
            "path": "python/oh_my_vllm/worker/model_runner.py",
            "line_range": [85, 215],
        },
    ]


@pytest.mark.parametrize("style", ["colon", "lines", "full", "basenames"])
def test_citations_bind_to_the_original_read_files(style):
    answer = {
        "colon": "lib.rs:240-330; model_runner.py:85-215.",
        "lines": "`lib.rs` (lines 240\u2013330); `model_runner.py` (lines 85-215).",
        "full": (
            "crates/scheduler/src/lib.rs; python/oh_my_vllm/worker/model_runner.py."
        ),
        "basenames": "lib.rs and model_runner.py.",
    }[style]
    result = source_citations(answer, reads())
    assert [citation["path"] for citation in result] == [r["path"] for r in reads()]
    assert all(citation["reference"] in answer for citation in result)
    assert [citation["actual_line_ranges"] for citation in result] == [
        [read["line_range"]] for read in reads()
    ]
    expected_match = True if style in ("colon", "lines") else None
    assert all(c["line_range_matches_read"] is expected_match for c in result)


@pytest.mark.parametrize("full_paths", [False, True])
def test_approximate_ranges_are_retained_without_rejecting_valid_files(full_paths):
    scheduler, worker = (
        (read["path"] if full_paths else read["path"].rsplit("/", 1)[-1])
        for read in reads()
    )
    # Endpoints reproduce the first formal MTP attempt's genuine tool answer.
    answer = (
        f"`{scheduler}` (lines 240\u2013333): `schedule()` builds a batch. "
        f"`{worker}` (lines 85\u2013218): the worker creates graph caches."
    )
    result = source_citations(answer, reads())
    assert [c["line_range"] for c in result] == [[240, 333], [85, 218]]
    assert [c["line_range_matches_read"] for c in result] == [False, False]
    assert [c["actual_line_ranges"] for c in result] == [[[240, 330]], [[85, 215]]]
    expected_style = "full_path" if full_paths else "unique_basename"
    assert all(c["style"] == expected_style for c in result)


def test_citation_ranges_remain_diagnostic_for_whole_and_multiple_reads():
    actual = [*reads(), dict(reads()[0], line_range=[334, 450])]
    result = source_citations("lib.rs:334-450; model_runner.py:85-218", actual)
    assert result[0]["actual_line_ranges"] == [[240, 330], [334, 450]]
    assert result[0]["line_range_matches_read"] is True
    actual[0]["line_range"] = None
    result = source_citations("lib.rs:1-9; model_runner.py", actual)
    assert result[0]["actual_line_ranges"] == [None, [334, 450]]
    assert result[0]["line_range_matches_read"] is None


@pytest.mark.parametrize("full_paths", [False, True])
def test_unrelated_reported_ranges_do_not_override_correct_file_identities(full_paths):
    names = [
        read["path"] if full_paths else read["path"].rsplit("/", 1)[-1]
        for read in reads()
    ]
    result = source_citations("; ".join(name + ":1-9" for name in names), reads())
    assert all(c["line_range"] == [1, 9] for c in result)
    assert all(c["line_range_matches_read"] is False for c in result)


@pytest.mark.parametrize(
    "answer",
    [
        "crates/kv-cache/src/lib.rs:240-330; model_runner.py:85-215.",
        "some_model_runner.py:85-215; lib.rs:240-330.",
        "../crates/scheduler/src/lib.rs:240-330; model_runner.py:85-215.",
        "lib.rs alone.",
        "No source filenames are cited.",
    ],
)
def test_missing_or_wrong_file_citations_are_rejected(answer):
    with pytest.raises(ValueError, match="unambiguous read citation"):
        source_citations(answer, reads())


@pytest.mark.parametrize("reference", ["lib.rs", "lib.rs:240-330", "lib.rs:1-9"])
def test_duplicate_read_basenames_require_full_paths(reference):
    actual = [
        *reads(),
        {"path": "crates/kv-cache/src/lib.rs", "line_range": [240, 330]},
    ]
    with pytest.raises(ValueError, match="unambiguous read citation"):
        source_citations(reference + "; model_runner.py:85-215", actual)
    assert (
        len(
            source_citations(
                "crates/scheduler/src/lib.rs:240-330; "
                "crates/kv-cache/src/lib.rs:240-330; "
                "model_runner.py:85-215",
                actual,
            )
        )
        == 3
    )


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_abbreviated_citation_still_requires_actual_tool_result_forwarding(api):
    events, requests = agent_evidence(api)
    for event in events:
        if event["type"] == "tool_execution_start":
            path = event["args"]["path"]
            event["args"]["path"] += (
                ":240-330" if path.endswith("lib.rs") else ":85-215"
            )
    final = events[-1]["message"]
    final["content"][0]["text"] = "lib.rs:240-333; model_runner.py:85-218"
    with pytest.raises(ValueError, match="both required source filenames"):
        agentic.validate_agent_evidence(events, requests, api)
    result = agentic.validate_agent_evidence(
        events, requests, api, citation_check=source_citations
    )
    assert result["tool_results_in_final_model_request"]
    assert len(result["final_source_citations"]) == 2
    assert all(
        c["line_range_matches_read"] is False for c in result["final_source_citations"]
    )
    history = requests[0]["request"]["messages" if api == "chat" else "input"]
    history.pop(0)
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(
            events, requests, api, citation_check=source_citations
        )
