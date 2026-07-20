# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nucleus asset root for Arena-hosted assets.

Points to staging bucket before release and to production bucket after
"""

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# TODO(2026.07.14, Point Arena assets to the production bucket before release)
ARENA_NUCLEUS_DIR: str = ISAACLAB_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")
