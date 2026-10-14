"""Actual subprocess RPC retains original bytes and binds control receipt lines."""

import gzip
import json
import subprocess
import sys
import threading

from benchmarks.agentic_deadline_evidence import control_records
from benchmarks.agentic_rpc_evidence import ControlRpcEvidence
from benchmarks.agentic_sessions import RpcClient


def test_native_control_bytes_and_receipt_lines_survive_real_rpc_io(tmp_path):
    program = """
import json,sys
for raw in sys.stdin:
    command=json.loads(raw)
    if command['type']=='abort':
        print(json.dumps({'type':'message_end','message':{
            'role':'assistant','responseId':'response-1','stopReason':'aborted'}}),flush=True)
    print(json.dumps({'id':command['id'],'type':'response',
        'command':command['type'],'success':True,'data':{'echo':command}}),flush=True)
"""
    client = object.__new__(RpcClient)
    client.directory = tmp_path
    client.control_evidence = ControlRpcEvidence(tmp_path)
    client.process = subprocess.Popen(
        [sys.executable, "-c", program], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    client.lane, client.sequence = 0, 0
    client.current_prompt, client.session_id = "goal-command", "fixed-session"
    client.condition, client.stop = threading.Condition(), threading.Event()
    client.responses, client.results, client.events = {}, {}, []
    client.error, client.compacting = None, False
    reader = threading.Thread(target=client._read)
    reader.start()
    try:
        pause = client.command("goal", op="pause")
        abort = client.command("abort")
        assert pause["echo"] == {"id": "lane-0-1", "type": "goal", "op": "pause"}
        assert abort["echo"] == {"id": "lane-0-2", "type": "abort"}
    finally:
        client.process.stdin.close()
        try:
            client.process.wait(timeout=5)
        finally:
            if client.process.poll() is None:
                client.process.kill()
                client.process.wait(timeout=5)
            reader.join(timeout=5)
            client.process.stdout.close()
    assert not reader.is_alive() and client.error is None
    records = control_records(tmp_path)
    assert len(records) == 7
    with gzip.open(tmp_path / "rpc.jsonl.gz", "rt") as source:
        raw_events = [json.loads(line) for line in source]
    assert [event["type"] for event in raw_events] == [
        "response",
        "message_end",
        "response",
    ]
    assert all("collected_unix_s" not in event for event in raw_events)
    assert {
        record["line_number"] for record in records if record["phase"] == "received"
    } == {1, 2, 3}
