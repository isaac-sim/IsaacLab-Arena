# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON-RPC client for the persistent :mod:`isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar` process."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from typing import Any


class SimAppSidecarError(RuntimeError):
    """Raised when the SimApp sidecar process can't fulfil a request."""


class SimAppSidecar:
    """Long-lived Kit/SimApp host process exposed as a validation service.

    See ``simapp_sidecar.py`` for the protocol. Cached via ``@st.cache_resource``
    in the Streamlit app; pipe access is serialized with an internal lock.
    """

    def __init__(self, *, boot_timeout_s: float = 180.0, shutdown_timeout_s: float = 10.0) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._boot_timeout_s = boot_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s

    def start(self) -> None:
        """Spawn the sidecar process and wait for its ``{"ready": true}`` handshake."""
        if self._proc is not None and self._proc.poll() is None:
            return

        cmd = [
            sys.executable,
            "-m",
            "isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        line = self._readline_or_die()
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self._terminate()
            raise SimAppSidecarError(f"Sidecar emitted non-JSON handshake: {line!r}") from exc

        if not msg.get("ready"):
            self._terminate()
            raise SimAppSidecarError(
                f"Sidecar boot failed: {msg.get('error', 'unknown error')}\n{msg.get('traceback', '')}"
            )

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        """Send ``shutdown``, then terminate/kill if the process doesn't exit."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            try:
                proc.wait(timeout=self._shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=5)

        with contextlib.suppress(Exception):
            if proc.stdin:
                proc.stdin.close()
        with contextlib.suppress(Exception):
            if proc.stdout:
                proc.stdout.close()

    def validate_yaml_text(self, yaml_text: str) -> dict[str, Any]:
        """Run full spec validation (including registry lookups) in the sidecar."""
        if not self.is_alive():
            raise SimAppSidecarError("SimApp sidecar is not running — start it first")

        with self._lock:
            return self._request({"cmd": "validate_spec", "yaml_text": yaml_text})

    def ping(self) -> bool:
        """Cheap liveness check round-trip — returns True on a healthy reply."""
        if not self.is_alive():
            return False
        with self._lock:
            try:
                response = self._request({"cmd": "ping"})
            except SimAppSidecarError:
                return False
        return bool(response.get("ok"))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        line = json.dumps(payload) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise SimAppSidecarError("sidecar pipe closed unexpectedly") from exc

        reply_line = self._readline_or_die()
        try:
            return json.loads(reply_line)
        except json.JSONDecodeError as exc:
            raise SimAppSidecarError(f"sidecar replied with non-JSON: {reply_line!r}") from exc

    def _readline_or_die(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if line == "":
            exit_code = self._proc.poll()
            raise SimAppSidecarError(
                f"sidecar exited prematurely (exit code: {exit_code}). "
                "See its stderr output above for the underlying cause."
            )
        return line

    def _terminate(self) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(Exception):
            self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                self._proc.kill()
        self._proc = None
