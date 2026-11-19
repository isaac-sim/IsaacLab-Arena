# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from isaaclab_arena.policy.policy_base import PolicyCfg


@dataclass
class RemoteChunkReplayPolicyCfg(PolicyCfg):
    """Connection and chunk-request settings shared by the configs of ``RemoteChunkReplayPolicy`` subclasses.

    Concrete configs that also configure their server declare a ``server`` field of their own server config type.
    """

    policy_device: str = "cuda"
    """Torch device for the returned action tensor."""

    remote_host: str = "localhost"
    """Hostname of the policy server."""

    remote_port: int = 8000
    """Port the policy server listens on."""

    batch_chunk_fetches: bool = True
    """Send the chunk requests of the environments that request on the same step as one batched request.

    Takes effect when the server advertises ``batch_inference`` in its connect metadata, as the batched
    openpi server (``isaaclab_arena_openpi/server/serve_batched_policy.py``) does; against any other server
    requests go out one per environment.
    """

    max_batch_size: int | None = None
    """Largest number of environments per batched request, or None to use the server's advertised limit.

    Caps the client's requests only, not the server's own limit.
    """

    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""

    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""

    def __post_init__(self) -> None:
        assert (
            self.max_batch_size is None or self.max_batch_size >= 1
        ), f"max_batch_size must be at least 1 or None, got {self.max_batch_size}"
