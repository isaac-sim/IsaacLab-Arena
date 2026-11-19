# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from isaaclab_arena.policy.policy_base import PolicyCfg

DEFAULT_VARIANT = "pi05"


# TODO(cvolk, 2026-05-18): unify the remote-policy config story across arena.
# Today openpi uses a Python dataclass (this file) and gr00t uses a YAML config
# (--policy_config_yaml_path). Decide on one mechanism (likely Hydra) when the
# planned RemotePolicy base class lands; the config-loading shape belongs there.
@dataclass
class Pi0RemotePolicyCfg(PolicyCfg):
    """Connection + runtime config for ``Pi0RemotePolicy``."""

    openpi_embodiment_adapter: str = "droid"
    """Adapter used to translate Arena observations and policy actions."""

    policy_variant: str = DEFAULT_VARIANT
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000

    batch_chunk_fetches: bool = True
    """Send the chunk requests of the environments that request on the same step as one batched request.

    Takes effect when the server advertises ``batch_inference`` in its connect metadata, as the batched
    openpi server (``isaaclab_arena_openpi/server/serve_batched_policy.py``) does; against any other server
    requests go out one per environment.
    """

    max_batch_size: int | None = None
    """Largest number of environments per batched request, or None to use the server's advertised limit."""

    batch_sizes: list[int] | None = None
    """Batch shapes the server pads every batched request to, for example ``[8, 16]``, or None for powers of two.

    Read by the OSMO pi0 server task, which starts the server with ``--batch-sizes``; the largest size is the
    batch limit. A locally started server takes the same shapes through ``run_openpi_server.sh -s``.
    """

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""
