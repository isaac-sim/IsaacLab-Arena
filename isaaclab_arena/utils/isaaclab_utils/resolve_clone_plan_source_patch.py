# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible no-op for Arena's retired nested-clone workaround.

Isaac Lab now resolves nested clone ownership through its query system and no
longer provides ``isaaclab.cloner.cloner_utils``. Keep the two probes below so
external integrations that imported the old Arena helper continue to work.
"""


def installed_resolver_handles_nesting() -> bool:
    """Return whether nested clone ownership is handled by the installed Isaac Lab."""
    return True


def patch_resolve_clone_plan_source() -> bool:
    """Return without patching because current Isaac Lab handles nested clone ownership.

    Returns:
        Always ``False`` because no runtime patch is required.
    """
    return False
