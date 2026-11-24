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
    """Batch shapes the server runs the model at, for example ``[2, 8, 20]``, or None for the server default of a
    single shape of 8.

    The server splits a batched request into groups of the largest shape and pads the rest to the smallest
    shape that holds it, so padding is least when a shape is near the number of environments that request on
    the same step (about a fifteenth of the environments, and all of them on the first step). The server
    compiles every shape when it starts, and every shape costs about the same (roughly three minutes on an
    L40S, whether the shape is 1 or 20), so a single shape per run is usually best: 8 at 100 environments,
    1 or 2 at 20. Read by the OSMO pi0 server task, which starts the server with ``--batch-sizes``; a locally
    started server takes the same shapes through ``run_openpi_server.sh -b``.
    """
