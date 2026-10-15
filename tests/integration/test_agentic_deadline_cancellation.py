"""Real sealed native-goal lanes cancelled at the collection deadline.

Each lane runs the pinned OMP binary inside a rootless container with
``--network none`` and only its workspace plus the bridge socket directory.
A local API streams an unterminated SSE response, so the request is still in
flight when the native goal reaches the collection deadline. The controller then
pauses the goal and sends the native RPC abort, which cancels the request. The
test binds the original evidence: RPC pause/abort command coverage, the aborted
native result, the unchanged request bytes, the canonical downstream disconnect
at the proxy, and the recorded failure on both mounted hops. It runs the
accepted collector path, so the framework audit produces the deadline proof and
the transport audit binds each hop's own failed attempt. The audit still applies
the deadline rules unchanged, including main's two-millisecond native/SDK
completion bound, whose measured jitter is reported in Issue 19.
"""

import gzip
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.agentic_corpus import Corpus
from benchmarks.agentic_deadline_evidence import action, control_records
from benchmarks.agentic_framework_evidence import audit_framework_session
from benchmarks.agentic_lane_transport import lane_transport_evidence
from benchmarks.agentic_native_goals import run_goal
from benchmarks.agentic_sandbox import docker
from benchmarks.agentic_sdk_trace import read_sdk_spans
from benchmarks.agentic_sessions import RpcClient
from benchmarks.agentic_soak import explained_aborts
from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy
from benchmarks.agentic_translation import objective
from tests.support.corpus import synthetic_book

pytestmark = pytest.mark.omp

DURATION = 8.0
CHAT_IDENTITY = "chatcmpl_deadline_1"
RESPONSES_IDENTITY = "resp_deadline_1"


def fixed_preflight(body: dict, path: str) -> dict:
    """Fixed CPU probe data for the smoke lane; the audit still binds it."""
    budget = (
        body.get("max_completion_tokens", body.get("max_tokens"))
        if path.endswith("/chat/completions")
        else body.get("max_output_tokens")
    )
    return {
        "input_tokens": 1024,
        "output_budget": budget,
        "total_budget": 1024 + budget,
        "reasoning_effort": "xhigh",
    }


def fixed_metric(api: str, identity: str, wire: dict) -> dict:
    """Fixed CPU API data for the smoke lane; the audit still binds it."""
    received = wire["client_received_unix_s"]
    completed = wire["transport_error_unix_s"]
    return {
        "api": api,
        "epoch": 1,
        "request_id": int(identity.rsplit("_", 1)[1]),
        "response_id": identity,
        "status": "cancelled",
        "output_tokens": 8,
        "received_unix_s": received,
        "processing_started_unix_s": received + 0.01,
        "generation_completed_unix_s": completed,
        "queue_seconds": 0.01,
        "processing_seconds": completed - received,
    }


def chat_frames(identity: str, model: str, index: int) -> bytes:
    delta = {"role": "assistant"} if index == 0 else {"content": f"passage {index} "}
    event = {
        "id": identity,
        "model": model,
        "choices": [{"index": 0, "delta": delta}],
    }
    return b"data: " + json.dumps(event).encode() + b"\n\n"


def responses_frames(identity: str, model: str, index: int) -> bytes:
    if index == 0:
        created = {
            "type": "response.created",
            "response": {"id": identity, "model": model, "status": "in_progress"},
        }
        item = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_deadline_1",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }
        return b"".join(
            b"event: "
            + event["type"].encode()
            + b"\ndata: "
            + json.dumps(event).encode()
            + b"\n\n"
            for event in (created, item)
        )
    delta = {
        "type": "response.output_text.delta",
        "response": {"id": identity, "model": model, "status": "in_progress"},
        "output_index": 0,
        "content_index": 0,
        "delta": f"passage {index} ",
    }
    return (
        b"event: "
        + delta["type"].encode()
        + b"\ndata: "
        + json.dumps(delta).encode()
        + b"\n\n"
    )


class StreamingHandler(BaseHTTPRequestHandler):
    """Stream an unterminated SSE response until the client disappears."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def chunk(self, payload: bytes):
        self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        api = "chat" if "messages" in body else "responses"
        identity = CHAT_IDENTITY if api == "chat" else RESPONSES_IDENTITY
        frames = chat_frames if api == "chat" else responses_frames
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for index in range(400):
                self.chunk(frames(identity, body["model"], index))
                self.server.streamed.set()
                time.sleep(0.25)
        except OSError:
            # The aborted client closed the connection: the stream ends here.
            return


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_deadline_abort_cancels_an_in_flight_sealed_request(
    tmp_path, agentic_sandbox_image, api
):
    book, identity, _manifest = synthetic_book(tmp_path)
    corpus = Corpus(book, identity)
    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
    server.bodies, server.streamed = [], threading.Event()
    handler_thread = threading.Thread(target=server.serve_forever, daemon=True)
    handler_thread.start()
    directory = tmp_path / "lane"
    directory.mkdir()
    client = None
    closed = False
    try:
        with disk_proxy(
            f"http://127.0.0.1:{server.server_port}/v1",
            directory / "wire",
            ModelSlots(),
            request_timeout=180,
            preflight=fixed_preflight,
            request_context=lambda: client.context() if client else {"lane": 0},
        ) as (proxy, records):
            client = RpcClient(
                "omp",
                api,
                0,
                directory,
                proxy,
                threading.Event(),
                120,
                workload="translation",
                sandbox_image=agentic_sandbox_image,
                corpus=corpus,
            )
            client.initialize()
            assert client.sandbox.inspect()["network_mode"] == "none"
            clock = {"monotonic": time.monotonic(), "unix": time.time()}
            collection = run_goal(client, clock, DURATION, objective)
            result = client.collect()
            # The soak runner records these two fields for the lane evidence.
            result["workload_collection"] = collection
            result["active_seconds"] = (
                collection["collection_deadline_monotonic_s"] - clock["monotonic"]
            )
            # Every journal and archive is finalized once the fixed process exits.
            try:
                client.close()
            finally:
                closed = True
        assert result["workload_collection"] == collection
        deadline = collection["collection_deadline_monotonic_s"]
        assert (
            abs(collection["collection_deadline_unix_s"] - clock["unix"] - DURATION)
            < 0.01
        )
        assert server.streamed.is_set() and len(server.bodies) == 1
        assert len(records) == 1

        turns = json.loads((directory / "turns.json").read_text())
        assert (
            len(turns) == 1 and turns[0]["transition_reason"] == "collection_deadline"
        )
        assert turns[0]["goal_pause"]["goal"]["id"] == turns[0]["goal_id"]
        ended = collection["controller_ended_monotonic_s"]
        assert deadline <= turns[0]["active_ended_monotonic_s"] <= ended

        # Original RPC coverage: one successful pause and abort after the deadline.
        control = control_records(directory)
        pause = action(control, "goal", deadline)
        abort = action(control, "abort", deadline)
        paused_goal = pause["acknowledgement"]["message"]["data"]["goal"]
        assert paused_goal["id"] == turns[0]["goal_id"]
        assert paused_goal["status"] == "paused"
        assert abort["acknowledgement"]["message"]["success"] is True
        assert (
            pause["acknowledgement"]["monotonic_s"]
            <= abort["start"]["monotonic_s"]
            <= abort["finish"]["monotonic_s"]
            <= ended
        )
        assert abort["acknowledgement"]["monotonic_s"] <= ended
        offset = collection["collection_deadline_unix_s"] - deadline
        for record in control:
            if record["monotonic_s"] >= deadline:
                assert abs(record["unix_s"] - record["monotonic_s"] - offset) < 0.002

        # Original model-request bytes stay unchanged and the native request aborted.
        wire = json.loads((directory / "wire/http.jsonl").read_text().splitlines()[0])
        assert wire["transport_error_kind"] in (
            "BrokenPipeError",
            "ConnectionResetError",
        )
        assert wire["error_direction"] == "downstream_write"
        assert wire["transport_outcome"] == "failed"
        assert wire["error"].startswith(wire["transport_error_kind"] + ":")
        assert (
            wire["client_received_unix_s"]
            <= collection["collection_deadline_unix_s"]
            <= wire["transport_error_unix_s"]
        )
        request_bytes = (directory / "wire" / wire["request_file"]).read_bytes()
        assert [json.loads(request_bytes)["model"]] == ["qwen3.8-27b-fp8"]
        with gzip.open(directory / "rpc.jsonl.gz", "rt") as stream:
            aborted = [
                record["message"]
                for record in map(json.loads, stream)
                if record.get("type") == "message_end"
                and record.get("message", {}).get("role") == "assistant"
                and record["message"].get("stopReason") == "aborted"
            ]
        assert len(aborted) == 1
        assert aborted[0]["responseId"] == wire["response_ids"][0]
        assert len(wire["response_ids"]) == 1

        # The collector path binds the abort: the framework audit produces the
        # deadline proof from the original evidence, the runner derives one
        # canonical attempt bound, and the transport audit binds the hops.
        session = dict(result, lane=0, api=api)
        audit = audit_framework_session(
            directory,
            session,
            {wire["response_ids"][0]: fixed_metric(api, wire["response_ids"][0], wire)},
            clock["unix"],
        )[1]
        explained = explained_aborts(directory, audit)
        digest = hashlib.sha256(request_bytes).hexdigest()
        assert set(explained) == {digest}, [
            (outcome["status"], outcome.get("cancellation_evidence"))
            for outcome in audit["outcomes"]
        ]
        bound = explained[digest][0]
        assert bound["direction"] == "downstream_write"
        assert bound["response_id"] == wire["response_ids"][0]
        # The accepted window is the recorded abort action through the client
        # disconnect, and the abort is dispatched after the deadline.
        assert bound["deadline_monotonic_s"] == deadline
        assert deadline < bound["abort_monotonic_s"] <= bound["last_monotonic_s"]

        transport = lane_transport_evidence(directory, explained={"model": explained})
        assert transport["explained_aborts"] == {"model": [digest]}
        assert transport["unused_explanations"] == []
        for hop in ("bridge-model", "guest-model"):
            assert transport[hop]["explained_aborts"] == 1
            assert transport[hop]["reached_records"] == 0
        assert transport["bridge-telemetry"]["reached_records"] == 1
        spans = read_sdk_spans(directory / "sdk-trace")
        aborted_spans = [
            span
            for span in spans
            if span["attributes"].get("gen_ai.response.id") == wire["response_ids"][0]
        ]
        assert len(aborted_spans) == 1
        attributes = aborted_spans[0]["attributes"]
        assert aborted_spans[0]["status_code"] == 2
        assert attributes.get("error.type") == "aborted"
        assert attributes.get("gen_ai.response.finish_reasons") == ["error"]
    finally:
        if client is not None and not closed:
            client.close()
        server.shutdown()
        handler_thread.join(timeout=10)
        server.server_close()
    assert not client.sandbox.root.exists()
    assert not docker(
        "container",
        "ls",
        "--all",
        "--quiet",
        "--filter",
        f"name=^/{client.sandbox.name}$",
    )
    assert json.loads((directory / "sandbox-cleanup.json").read_text())["closed"]
