# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab_arena.policy.remote_policy_cfg import RemoteChunkReplayPolicyCfg

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
class Pi0RemotePolicyCfg(RemoteChunkReplayPolicyCfg):
    """Connection + runtime config for ``Pi0RemotePolicy``."""

    openpi_embodiment_adapter: str = "droid"
    """Adapter used to translate Arena observations and policy actions."""

    policy_variant: str = DEFAULT_VARIANT

    server: Pi0ServerCfg = field(default_factory=Pi0ServerCfg)
    """Settings for launching the policy server, such as its batch shapes; the client ignores them."""
