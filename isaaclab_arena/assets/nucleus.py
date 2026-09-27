# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public S3 and internal Nucleus asset roots for Arena-hosted assets."""

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

ARENA_STAGING_NUCLEUS_DIR: str = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab"
"""Internal upload location, available immediately before the staging S3 mirror syncs."""

# TODO(2026.07.14, Point Arena assets to the production bucket before release)
ARENA_NUCLEUS_DIR: str = ISAACLAB_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")
# TODO(2026.08.12, Fill request to sync Replicator kitchens to the production bucket once Sim 6.1 is released)
ISAAC_STAGING_NUCLEUS_DIR: str = ISAAC_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")
