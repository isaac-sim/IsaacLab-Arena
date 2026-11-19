# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.policy.policy_base import PolicyCfg

DEFAULT_VARIANT = "pi05"


@dataclass
class Pi0ServerCfg:
    """Settings for launching the openpi server that ``Pi0RemotePolicy`` connects to.

    The client ignores them. The OSMO pi0 server task starts its server with them, and
    ``run_openpi_server.sh`` takes the same settings as ``-b``, ``-s`` and ``-u``.
    """

    max_batch_size: int | None = 32
    """Most observations the batched server runs through the model in one call, or None to serve upstream's
    one-observation ``scripts/serve_policy.py``. Ignored when ``batch_sizes`` is given."""

    batch_sizes: list[int] | None = None
    """Batch shapes the batched server pads every batch to, for example ``[8, 16]``, or None for powers of two up
    to ``max_batch_size``. The largest size is the batch limit."""

    def __post_init__(self) -> None:
        assert (
            self.max_batch_size is None or self.max_batch_size >= 1
        ), f"pi0 server max_batch_size must be at least 1 or None, got {self.max_batch_size}"
        assert self.batch_sizes is None or (
            self.max_batch_size is not None and self.batch_sizes and all(size >= 1 for size in self.batch_sizes)
        ), f"pi0 server batch_sizes needs the batched server and positive sizes, got {self.batch_sizes}"


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
    """Largest number of environments per batched request, or None to use the server's advertised limit.

    Caps the client's requests only; the server's own limit is ``server.max_batch_size``.
    """

    server: Pi0ServerCfg = field(default_factory=Pi0ServerCfg)
    """Settings for launching the policy server, such as its batch shapes; the client ignores them."""

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""
