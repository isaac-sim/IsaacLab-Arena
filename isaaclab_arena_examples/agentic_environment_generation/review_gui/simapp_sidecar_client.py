# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JSON-RPC client for the review GUI SimApp sidecar (Unix domain socket)."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import yaml
from pathlib import Path
from typing import Any, TextIO

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec

SIDECAR_SOCKET_ENV = "ARENA_REVIEW_SIDECAR_SOCKET"


class SimAppSidecarError(RuntimeError):
    """Raised when the SimApp sidecar process can't fulfil a request."""


class SimAppSidecarClient:
    """Client for a Kit sidecar listening on a Unix domain socket."""

    def __init__(self, socket_path: str, reader: TextIO, writer: TextIO) -> None:
        self._socket_path = socket_path
        self._reader = reader
        self._writer = writer
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    @classmethod
    def connect(cls, socket_path: str) -> SimAppSidecarClient:
        """Open a persistent JSON-RPC session to the sidecar at ``socket_path``."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")
        client = cls(socket_path, reader, writer)
        client._sock = sock
        return client

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def is_alive(self) -> bool:
        return self.ping()

    def close(self) -> None:
        """Send ``shutdown`` when possible, then close the socket."""
        with self._lock:
            try:
                self._writer.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._writer.flush()
            except OSError:
                pass
            self._close_handles()

    def validate_yaml_text(self, yaml_text: str) -> dict[str, Any]:
        """Run full spec validation (including registry lookups) in the sidecar."""
        with self._lock:
            return self._request({"cmd": "validate_spec", "yaml_text": yaml_text})

    def build_catalogues(self) -> dict[str, Any]:
        """Build asset/relation/task catalogues from warm registries in the sidecar."""
        with self._lock:
            return self._request({"cmd": "build_catalogues"})

    def compile_intent(self, intent_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate and compile an EnvironmentIntentSpec in the sidecar."""
        with self._lock:
            return self._request({"cmd": "compile_intent", "intent_dict": intent_dict})

    def render_spec(self, spec: ArenaEnvInitialGraphSpec) -> dict[str, bytes]:
        """Ask the sidecar to render thumbnails for ``spec``."""
        yaml_text = yaml.safe_dump(spec.to_dict(), sort_keys=False)
        with self._lock:
            response = self._request({"cmd": "render_spec", "yaml_text": yaml_text})

        if not response.get("ok"):
            raise SimAppSidecarError(
                f"sidecar render failed: {response.get('error', 'unknown')}\n{response.get('traceback', '')}"
            )

        paths: dict[str, str] = response.get("paths", {}) or {}
        results: dict[str, bytes] = {}
        for node_id, path_str in paths.items():
            path = Path(path_str)
            if path.exists() and path.stat().st_size > 0:
                results[node_id] = path.read_bytes()
            else:
                print(
                    f"[review_gui]   sidecar reported {node_id} -> {path_str} but file is missing.",
                    file=sys.stderr,
                )
        return results

    def ping(self) -> bool:
        """Cheap liveness check round-trip — returns True on a healthy reply."""
        with self._lock:
            try:
                response = self._request({"cmd": "ping"})
            except SimAppSidecarError:
                return False
        return bool(response.get("ok"))

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            raise SimAppSidecarError("sidecar socket is closed")
        line = json.dumps(payload) + "\n"
        try:
            self._writer.write(line)
            self._writer.flush()
        except OSError as exc:
            raise SimAppSidecarError("sidecar socket closed unexpectedly") from exc

        reply_line = self._readline_or_die()
        try:
            return json.loads(reply_line)
        except json.JSONDecodeError as exc:
            raise SimAppSidecarError(f"sidecar replied with non-JSON: {reply_line!r}") from exc

    def _readline_or_die(self) -> str:
        line = self._reader.readline()
        if line == "":
            raise SimAppSidecarError(
                f"sidecar at {self._socket_path} closed the connection unexpectedly. "
                "Restart the review GUI via gui_runner."
            )
        return line

    def _close_handles(self) -> None:
        for handle in (self._reader, self._writer):
            with contextlib.suppress(OSError):
                handle.close()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._sock = None


def sidecar_socket_from_env() -> str | None:
    """Return the sidecar socket path from ``ARENA_REVIEW_SIDECAR_SOCKET``, if set."""
    path = os.environ.get(SIDECAR_SOCKET_ENV, "").strip()
    return path or None


def wait_for_sidecar_socket(
    socket_path: str,
    proc: subprocess.Popen[Any],
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until ``socket_path`` accepts connections and responds to ``ping``."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise SimAppSidecarError(
                f"Sidecar exited during boot (exit code: {exit_code}). See stderr above for details."
            )
        try:
            client = SimAppSidecarClient.connect(socket_path)
        except OSError as exc:
            last_error = exc
            time.sleep(poll_interval_s)
            continue
        try:
            if client.ping():
                return
        except SimAppSidecarError as exc:
            last_error = exc
        finally:
            client.close()
        time.sleep(poll_interval_s)

    detail = f" Last error: {last_error}" if last_error is not None else ""
    raise SimAppSidecarError(f"Timed out after {timeout_s:.0f}s waiting for sidecar at {socket_path}.{detail}")


def spawn_sidecar_process(socket_path: str) -> subprocess.Popen[Any]:
    """Launch the SimApp sidecar subprocess bound to ``socket_path``."""
    if Path(socket_path).exists():
        Path(socket_path).unlink()
    cmd = [
        sys.executable,
        "-m",
        "isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp_sidecar",
        "--socket",
        socket_path,
    ]
    return subprocess.Popen(cmd, env=os.environ.copy())


def stop_sidecar_process(proc: subprocess.Popen[Any] | None, socket_path: str) -> None:
    """Ask the sidecar to shut down cleanly, then terminate the process and remove the socket."""
    try:
        client = SimAppSidecarClient.connect(socket_path)
    except OSError:
        client = None
    if client is not None:
        with contextlib.suppress(SimAppSidecarError):
            client.close()

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    Path(socket_path).unlink(missing_ok=True)
