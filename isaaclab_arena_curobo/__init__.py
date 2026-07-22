# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""cuRobo build-time IK reachability extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams


def configure_reachability_validator(placer_params: ObjectPlacerParams, embodiment: EmbodimentBase | None) -> bool:
    """Install the cuRobo IK-reachability validator on placer_params.extra_validators (imports the solver on first use).

    Returns True when the reachability check implementation was installed, False when it cannot run and is skipped -- either because the
    solver deps (torch, cuRobo) are absent, or the embodiment has no registered cuRobo config.
    """
    try:
        from isaaclab_arena_curobo.ik_reachability_validator import configure_reachability_validator as _configure
    except ImportError:
        #  Dev environment without cuRobo deps
        return False

    return _configure(placer_params, embodiment)
