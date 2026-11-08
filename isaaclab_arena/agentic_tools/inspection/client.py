# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless client for the local inspection backend."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab_arena.agentic_tools.inspection.protocol import (
    InspectionError,
    InspectionResult,
    encode_record,
    read_record,
    validate_request,
    validate_result,
)

if TYPE_CHECKING:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec


class InspectorClient:
    """Send each inspection request on its own finite-lived connection."""

    def __init__(
        self,
        socket_path: Path,
        *,
        connect_timeout_s: float = 5.0,
        request_timeout_s: float = 1800.0,
    ) -> None:
        assert 0 < connect_timeout_s < float("inf") and 0 < request_timeout_s < float("inf")
        self.socket_path = Path(socket_path)
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s

    def _request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        validate_request(payload)
        record = encode_record(payload)
        deadline = self.request_timeout_s if timeout_s is None else timeout_s
        assert 0 < deadline < float("inf")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(min(self.connect_timeout_s, deadline))
                sock.connect(str(self.socket_path))
                sock.settimeout(deadline)
                sock.sendall(record)
                response = read_record(sock, timeout_s=deadline)
        except (OSError, ValueError, InspectionError) as exc:
            raise InspectionError(
                f"Inspection request {payload['cmd']} at {self.socket_path} failed: {exc}. "
                "If a deadline expired, the outcome is unknown; work may still finish."
            ) from exc
        if response.get("ok") is not True:
            raise InspectionError(
                f"{response.get('error', 'invalid backend response')}\n{response.get('traceback', '')}"
            )
        return response

    def ping(self, *, timeout_s: float = 5.0) -> None:
        """Return after a successful health check, or raise InspectionError."""
        self._request({"cmd": "ping"}, timeout_s=timeout_s)

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        """Ask the selected backend to exit."""
        self._request({"cmd": "shutdown"}, timeout_s=timeout_s)

    def render_spec(
        self,
        spec: ArenaEnvGraphSpec,
        *,
        background_panorama: bool = False,
        ignore_cache: bool = False,
    ) -> InspectionResult:
        """Return completed thumbnail paths, dimensions, and capture warnings."""
        response = self._request({
            "cmd": "render_spec",
            "spec": spec.to_dict(),
            "options": {"background_panorama": background_panorama, "ignore_cache": ignore_cache},
        })
        return validate_result(response.get("result"))

    def run_sim_preview(
        self,
        spec: ArenaEnvGraphSpec,
        *,
        num_envs: int,
        num_steps: int,
        env_spacing: float,
        show_placement_overlay: bool = False,
    ) -> InspectionResult:
        """Return a completed preview video and rollout metadata."""
        response = self._request({
            "cmd": "run_sim_preview",
            "spec": spec.to_dict(),
            "options": {
                "num_envs": num_envs,
                "num_steps": num_steps,
                "env_spacing": env_spacing,
                "show_placement_overlay": show_placement_overlay,
            },
        })
        return validate_result(response.get("result"))
