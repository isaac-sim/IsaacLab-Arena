# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo build-time IK reachability extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.placement_result import PlacementResult


def make_reachability_validator(embodiment: EmbodimentBase, **kwargs) -> Callable[[PlacementResult], bool]:
    """Build the cuRobo IK-reachability layout validator (imports the solver on first use)."""
    from isaaclab_arena_curobo.standalone_ik_layout_validator import make_standalone_ik_layout_validator

    return make_standalone_ik_layout_validator(embodiment, **kwargs)
