# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_UNSPECIFIED_HOSTS = {"", "*", "0.0.0.0", "::"}


def _looks_loopback(host: str) -> bool:
    return host.startswith("127.") or host == "::1" or host == "localhost"


def _detect_local_ip_via_route(target_host: str) -> str | None:
    try:
        infos = socket.getaddrinfo(target_host, 1, type=socket.SOCK_DGRAM)
    except socket.gaierror:
        return None

    for family, socktype, proto, _, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.connect(sockaddr)
                ip = sock.getsockname()[0]
        except OSError:
            continue
        if ip and not _looks_loopback(ip):
            return ip
    return None


def autodetect_local_hostname(preferred_remote_host: str | None = None) -> str | None:
    """Best-effort local IP detection for Mooncake peer advertisement.

    Strategy:
    - Prefer the local source IP chosen for routing toward the remote peer.
    - Fall back to the default outbound route if no peer host is available.
    - Finally fall back to hostname resolution if it yields a non-loopback IP.
    """
    normalized_remote = (preferred_remote_host or "").strip()
    if normalized_remote and normalized_remote not in _UNSPECIFIED_HOSTS and not _looks_loopback(normalized_remote):
        detected = _detect_local_ip_via_route(normalized_remote)
        if detected is not None:
            return detected

    for fallback_target in ("192.0.2.1", "2001:db8::1"):
        detected = _detect_local_ip_via_route(fallback_target)
        if detected is not None:
            return detected

    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    if hostname_ip and not _looks_loopback(hostname_ip):
        return hostname_ip
    return None


@dataclass
class MooncakeTransportConfig:
    """Mooncake-specific transport configuration.

    Public API intent:
    - Keep only the small set of user-facing parameters in the shared CLI.
    - Move Mooncake-specific tuning knobs behind environment variables or
      advanced internal defaults rather than polluting the generic remote
      policy interface.

    Environment variables used by ``from_public_args()``:
    - `ISAACLAB_ARENA_MOONCAKE_PROTOCOL`
      Values: `rdma`, `tcp`
      Default: `rdma`
    - `ISAACLAB_ARENA_MOONCAKE_DEVICE_NAME`
      Optional explicit RDMA device name.
    - `ISAACLAB_ARENA_MOONCAKE_METADATA_BACKEND`
      Default: `P2PHANDSHAKE`
    - `ISAACLAB_ARENA_MOONCAKE_BUFFER_BYTES`
      Initial/default staging capacity in bytes. The runtime may grow beyond
      this to fit a larger single payload.
    - `ISAACLAB_ARENA_MOONCAKE_FORCE_REGISTER`
      Whether to force memory registration even on non-RDMA protocols.
    - `ISAACLAB_ARENA_MOONCAKE_CUDA_DEVICE_OVERRIDE`
      Advanced internal override. Prefer per-process torch device inference.
    - `ISAACLAB_ARENA_MOONCAKE_LOCAL_HOSTNAME`
      Optional fallback when CLI/config does not provide a local hostname.
    """

    local_hostname: str | None = None
    protocol: str = "rdma"
    device_name: str | None = None

    # Advanced / internal settings. These are intentionally not surfaced in
    # the shared public CLI.
    metadata_backend: str = "P2PHANDSHAKE"
    # Initial/default staging capacity. The Mooncake transport may grow the
    # registered buffer beyond this when a single payload is larger.
    staging_buffer_bytes: int = 64 * 1024 * 1024
    force_register: bool = True
    cuda_device_override: str | None = None

    @classmethod
    def from_public_args(
        cls,
        *,
        local_hostname: str | None = None,
    ) -> MooncakeTransportConfig:
        resolved_local_hostname = local_hostname or os.getenv("ISAACLAB_ARENA_MOONCAKE_LOCAL_HOSTNAME")
        return cls(
            local_hostname=resolved_local_hostname,
            protocol=os.getenv("ISAACLAB_ARENA_MOONCAKE_PROTOCOL", "rdma"),
            device_name=os.getenv("ISAACLAB_ARENA_MOONCAKE_DEVICE_NAME") or None,
            metadata_backend=os.getenv("ISAACLAB_ARENA_MOONCAKE_METADATA_BACKEND", "P2PHANDSHAKE"),
            staging_buffer_bytes=_env_int("ISAACLAB_ARENA_MOONCAKE_BUFFER_BYTES", 64 * 1024 * 1024),
            force_register=_env_bool("ISAACLAB_ARENA_MOONCAKE_FORCE_REGISTER", True),
            cuda_device_override=os.getenv("ISAACLAB_ARENA_MOONCAKE_CUDA_DEVICE_OVERRIDE") or None,
        )


def add_client_mooncake_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--remote_mooncake_local_hostname",
        type=str,
        default=None,
        help="Optional override for the local hostname/IP that peers should use to reach this client's Mooncake engine. If omitted, the client will try to auto-detect a routable local address.",
    )
    return parser


def add_server_mooncake_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mooncake_local_hostname",
        type=str,
        default=None,
        help="Optional override for the local hostname/IP that peers should use to reach this server's Mooncake engine. If omitted, the server will try to auto-detect a routable local address.",
    )
    return parser
