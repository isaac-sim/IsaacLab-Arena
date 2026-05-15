# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wait for a GR00T policy server to answer its ping endpoint in CI."""

from __future__ import annotations

import argparse
import socket
import time

import zmq
from gr00t.policy.server_client import MsgSerializer


def _diagnostics(host: str, port: int, tcp_timeout_sec: float) -> str:
    """Return cheap network diagnostics for CI logs.

    A TCP-open check alone is not enough for readiness because the server can
    bind its socket before the model is loaded. These diagnostics are only for
    debugging failed waits; readiness is decided by the GR00T ping endpoint.
    """
    try:
        dns_status = f"resolved -> {socket.gethostbyname_ex(host)}"
    except Exception as exc:
        dns_status = f"FAILED: {type(exc).__name__}: {exc}"
    try:
        with socket.create_connection((host, port), timeout=tcp_timeout_sec) as sock:
            tcp_status = f"OK ({sock.getpeername()})"
    except Exception as exc:
        tcp_status = f"FAILED: {type(exc).__name__}: {exc}"
    return f"DNS {host}: {dns_status}; TCP {host}:{port}: {tcp_status}"


def _ping(host: str, port: int, timeout_ms: int) -> tuple[bool, object]:
    """Call the GR00T ping endpoint with explicit socket timeouts."""
    # PolicyClient accepts a timeout_ms constructor argument but does not apply
    # it to the socket in the GR00T version used by this image. Use ZMQ directly
    # so recv() cannot block forever while the sidecar is still booting.
    context = zmq.Context.instance()
    socket_req = context.socket(zmq.REQ)
    socket_req.setsockopt(zmq.LINGER, 0)
    socket_req.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket_req.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        socket_req.connect(f"tcp://{host}:{port}")
        socket_req.send(MsgSerializer.to_bytes({"endpoint": "ping"}))
        return True, MsgSerializer.from_bytes(socket_req.recv())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        socket_req.close()


def wait_for_gr00t_server(
    host: str,
    port: int,
    timeout_sec: float,
    poll_interval_sec: float,
    request_timeout_ms: int,
    tcp_probe_timeout_sec: float,
) -> bool:
    """Poll a GR00T server until it answers ping or the timeout expires."""
    started_at = time.monotonic()
    deadline = started_at + timeout_sec

    print(f"[init] {_diagnostics(host, port, tcp_probe_timeout_sec)}", flush=True)

    while time.monotonic() < deadline:
        elapsed = int(time.monotonic() - started_at)
        ok, detail = _ping(host, port, request_timeout_ms)
        if ok:
            print(f"[t={elapsed}s] GR00T server ready (ping={detail})", flush=True)
            return True

        # Include DNS/TCP diagnostics on every attempt. The poll interval is
        # long enough that this is low-noise, and it makes service-startup
        # failures obvious in GitHub Actions logs.
        print(f"[t={elapsed}s] not ready: {detail}; {_diagnostics(host, port, tcp_probe_timeout_sec)}", flush=True)

        sleep_sec = min(poll_interval_sec, max(0.0, deadline - time.monotonic()))
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    print("---- final diagnostics ----", flush=True)
    print(_diagnostics(host, port, tcp_probe_timeout_sec), flush=True)
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a GR00T policy server to become ready.")
    parser.add_argument("--host", default="gr00t", help="GR00T policy server hostname.")
    parser.add_argument("--port", type=int, default=5555, help="GR00T policy server port.")
    parser.add_argument("--timeout-sec", type=float, default=600.0, help="Maximum time to wait.")
    parser.add_argument("--poll-interval-sec", type=float, default=15.0, help="Delay between ping attempts.")
    parser.add_argument("--request-timeout-ms", type=int, default=5000, help="ZMQ send/receive timeout per ping.")
    parser.add_argument("--tcp-probe-timeout-sec", type=float, default=3.0, help="TCP diagnostic probe timeout.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ready = wait_for_gr00t_server(
        host=args.host,
        port=args.port,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
        request_timeout_ms=args.request_timeout_ms,
        tcp_probe_timeout_sec=args.tcp_probe_timeout_sec,
    )
    if ready:
        return 0
    raise SystemExit("timed out waiting for GR00T server")


if __name__ == "__main__":
    raise SystemExit(main())
