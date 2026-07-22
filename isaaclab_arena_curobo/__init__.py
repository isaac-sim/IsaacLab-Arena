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


def configure_reachability_gate(placer_params: ObjectPlacerParams, embodiment: EmbodimentBase | None) -> bool:
    """Install the cuRobo IK-reachability gate on ``placer_params`` (imports the solver on first use).

    Returns True when the gate was installed, False when it cannot run and is skipped -- either because the
    solver deps (torch, cuRobo) are absent (e.g. a base image), or the embodiment is None or lacks a
    registered cuRobo config. The env builder calls this unconditionally; all skip decisions live here.
    """
    try:
        from isaaclab_arena_curobo.ik_reachability_validator import configure_reachability_gate as _configure
    except ImportError:
        # Base image without torch/cuRobo -- reachability is simply not enforced.
        return False

    return _configure(placer_params, embodiment)
