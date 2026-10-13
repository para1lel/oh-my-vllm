def metric(rid=7, *, status="completed", api="chat", tokens=20):
    return {
        "request_id": rid,
        "response_id": f"{'chatcmpl' if api == 'chat' else 'resp'}_test_{rid}",
        "api": api,
        "status": status,
        "received_unix_s": 100.0,
        "processing_started_unix_s": 102.0,
        "generation_completed_unix_s": 105.0,
        "output_tokens": tokens,
        "queue_seconds": 2.0,
        "processing_seconds": 3.0,
    }


def task(rid=7, *, api="chat"):
    return {
        "task_id": "task",
        "lane": 3,
        "api": api,
        "passed": True,
        "requests": [
            {
                "method": "POST",
                "response_status": 200,
                "response_ids": [metric(rid, api=api)["response_id"]],
            }
        ],
    }
