"""Actual read identity and ranges disambiguate abbreviated model citations."""

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


@pytest.mark.parametrize("style", ["colon", "lines", "full"])
def test_citations_bind_to_the_original_read_files_and_ranges(style):
    answer = {
        "colon": "lib.rs:240-330; model_runner.py:85-215.",
        "lines": "`lib.rs` (lines 240\u2013330); `model_runner.py` (lines 85-215).",
        "full": (
            "crates/scheduler/src/lib.rs; python/oh_my_vllm/worker/model_runner.py."
        ),
    }[style]
    result = source_citations(answer, reads())
    assert [citation["path"] for citation in result] == [r["path"] for r in reads()]
    assert all(citation["reference"] in answer for citation in result)


@pytest.mark.parametrize(
    "answer",
    [
        "lib.rs and model_runner.py.",  # No range; lib.rs exists in many crates.
        "lib.rs:1-9; model_runner.py:85-215.",
        "crates/kv-cache/src/lib.rs:240-330; model_runner.py:85-215.",
        "some_model_runner.py:85-215; lib.rs:240-330.",
        "../crates/scheduler/src/lib.rs:240-330; model_runner.py:85-215.",
        "crates/scheduler/src/lib.rs:1-9; model_runner.py:85-215.",
    ],
)
def test_ambiguous_or_wrong_file_and_range_citations_are_rejected(answer):
    with pytest.raises(ValueError, match="unambiguous read citation"):
        source_citations(answer, reads())


def test_duplicate_read_basenames_require_full_paths():
    actual = [
        *reads(),
        {"path": "crates/kv-cache/src/lib.rs", "line_range": [240, 330]},
    ]
    with pytest.raises(ValueError, match="unambiguous read citation"):
        source_citations("lib.rs:240-330; model_runner.py:85-215", actual)
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
    final["content"][0]["text"] = "lib.rs:240-330; model_runner.py:85-215"
    with pytest.raises(ValueError, match="both required source filenames"):
        agentic.validate_agent_evidence(events, requests, api)
    result = agentic.validate_agent_evidence(
        events, requests, api, citation_check=source_citations
    )
    assert result["tool_results_in_final_model_request"]
    assert len(result["final_source_citations"]) == 2
    history = requests[0]["request"]["messages" if api == "chat" else "input"]
    history.pop(0)
    with pytest.raises(ValueError, match="not forwarded"):
        agentic.validate_agent_evidence(
            events, requests, api, citation_check=source_citations
        )
