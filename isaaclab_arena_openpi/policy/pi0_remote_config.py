# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from isaaclab_arena.policy.remote_policy_config import RemoteChunkReplayPolicyCfg

DEFAULT_VARIANT = "pi05"


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

    batch_sizes: list[int] | None = None
    """Batch shapes the server pads every batched request to, for example ``[8, 16]``, or None for powers of two.

    Read by the OSMO pi0 server task, which starts the server with ``--batch-sizes``; the largest size is the
    batch limit. A locally started server takes the same shapes through ``run_openpi_server.sh -s``.
    """
