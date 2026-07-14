# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nucleus asset root for Arena-hosted assets.

Isaac Lab's ``ISAACLAB_NUCLEUS_DIR`` moved from the Omniverse *staging* bucket to
the *production* bucket. Arena's in-development assets (e.g. ``srl_robolab_assets``)
are not yet published to production, so while Arena is under development we resolve
Arena-hosted assets from the staging bucket instead.

Arena code should use :data:`ARENA_NUCLEUS_DIR` in place of ``ISAACLAB_NUCLEUS_DIR``
for Arena-hosted assets. Only the bucket differs from the production URL; the
Isaac Sim version and path suffix are inherited from Isaac Lab's value.

TODO(amillane): Once Arena assets are published to the production bucket, set
``ARENA_NUCLEUS_DIR = ISAACLAB_NUCLEUS_DIR`` (or drop this module and import the
Isaac Lab default directly at the call sites).
"""

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ARENA_NUCLEUS_DIR: str = ISAACLAB_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")
"""Nucleus root for Arena-hosted assets (staging bucket during development)."""
