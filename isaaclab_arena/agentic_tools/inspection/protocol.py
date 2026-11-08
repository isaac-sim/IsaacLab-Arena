# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schemas and validation for the local inspection JSON-lines protocol."""

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any, TypedDict

MAX_RECORD_BYTES = 16 * 1024 * 1024


class InspectionError(RuntimeError):
    """An inspection transport, backend, or artifact operation failed."""


class Artifact(TypedDict):
    path: str
    media_type: str


class InspectionResult(TypedDict):
    artifacts: dict[str, Artifact]
    metadata: dict[str, Any]
    warnings: list[str]
    temporary_dir: str | None


def encode_record(payload: dict[str, Any]) -> bytes:
    """Encode one finite JSON mapping as a bounded UTF-8 record."""
    record = (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(record) > MAX_RECORD_BYTES:
        raise ValueError("inspection record exceeds size limit")
    return record


def read_record(sock: socket.socket, *, timeout_s: float) -> dict[str, Any]:
    """Read one bounded JSON record within a total deadline."""
    deadline = time.monotonic() + timeout_s
    data = bytearray()
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("inspection record deadline exceeded")
        sock.settimeout(remaining)
        chunk = sock.recv(min(65536, MAX_RECORD_BYTES + 1 - len(data)))
        if not chunk:
            raise InspectionError("inspection peer closed the connection before a complete record")
        data.extend(chunk)
        if len(data) > MAX_RECORD_BYTES:
            raise ValueError("inspection record exceeds size limit")
    value = json.loads(data.split(b"\n", 1)[0], parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("inspection record must be a JSON object")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def validate_request(request: dict[str, Any]) -> None:
    """Validate command, graph mapping, and option types before dispatch."""
    cmd = request.get("cmd")
    if cmd in ("ping", "shutdown"):
        return
    if cmd not in ("render_spec", "run_sim_preview"):
        raise ValueError(f"unknown cmd: {cmd!r}")
    if not isinstance(request.get("spec"), dict):
        raise ValueError(f"{cmd} requires a 'spec' mapping")
    options = request.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be a mapping")
    if cmd == "render_spec":
        allowed = {"background_panorama", "ignore_cache"}
    else:
        allowed = {"num_envs", "num_steps", "env_spacing", "show_placement_overlay"}
        for key in ("num_envs", "num_steps"):
            value = options.get(key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        spacing = options.get("env_spacing")
        if type(spacing) not in (int, float) or not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("env_spacing must be a positive finite number")
    if options.keys() - allowed:
        raise ValueError(f"unknown options: {sorted(options.keys() - allowed)}")
    for key in ("background_panorama", "ignore_cache", "show_placement_overlay"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")


def validate_result(value: Any) -> InspectionResult:
    """Check the shared artifact result envelope received from a backend."""
    if not isinstance(value, dict):
        raise InspectionError("inspection result must be an object")
    if {"artifacts", "metadata", "warnings", "temporary_dir"} - value.keys():
        raise InspectionError("inspection result is missing required fields")
    if not isinstance(value.get("artifacts"), dict) or not isinstance(value.get("metadata"), dict):
        raise InspectionError("inspection result requires artifacts and metadata mappings")
    if not isinstance(value.get("warnings"), list) or not all(isinstance(w, str) for w in value["warnings"]):
        raise InspectionError("inspection warnings must be a list of strings")
    if value.get("temporary_dir") is not None and not isinstance(value["temporary_dir"], str):
        raise InspectionError("inspection temporary_dir must be a path or null")
    for artifact in value["artifacts"].values():
        if not isinstance(artifact, dict) or not all(isinstance(artifact.get(k), str) for k in ("path", "media_type")):
            raise InspectionError("invalid inspection artifact")
    return value
