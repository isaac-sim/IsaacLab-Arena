from __future__ import annotations

import json
import os
import socket
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def metrics_log_path() -> str | None:
    return os.getenv("ISAACLAB_ARENA_REMOTE_METRICS_LOG")


def metrics_enabled() -> bool:
    return metrics_log_path() is not None


def record_remote_metric(event: str, **fields: Any) -> None:
    path = metrics_log_path()
    if path is None:
        return

    record: dict[str, Any] = {
        "event": event,
        "ts_ns": time.time_ns(),
        "perf_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "role": os.getenv("ISAACLAB_ARENA_REMOTE_METRICS_ROLE"),
        "case": os.getenv("ISAACLAB_ARENA_REMOTE_METRICS_CASE"),
    }
    record.update({key: _jsonable(value) for key, value in fields.items()})

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")

    with _WRITE_LOCK:
        fd = os.open(path_obj, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
