"""UTC structured worker logs. Durations use monotonic clocks, never wall time."""

import datetime
import json
import logging
import os


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, datetime.UTC
            ).isoformat(),
            "level": record.levelname,
            "run_id": os.environ.get("OH_MY_VLLM_RUN_ID", "unassigned"),
            "pid": os.getpid(),
            "component": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("oh_my_vllm")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(os.environ.get("OH_MY_VLLM_LOG_LEVEL", "INFO").upper())
