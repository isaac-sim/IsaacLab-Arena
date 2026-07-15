# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of cuRobo descriptions keyed by embodiment name.

Keeps cuRobo out of core: core ``EmbodimentBase`` stays vendor-agnostic, and this extension owns the
mapping from an embodiment to its cuRobo config. Look one up with ``get_curobo_cfg_for(embodiment)``;
register more with ``register_curobo_cfg(embodiment_name, cfg)``.

The cuRobo config is a property of the robot hardware, so register under the robot-family base name
(e.g. ``"droid"``, the class-level ``name`` on ``DroidEmbodimentBase``). Concrete action variants
override ``name`` (e.g. ``"droid_abs_joint_pos"``), so the lookup walks the embodiment's class
hierarchy — one family entry covers every variant, and a variant may still register its own override.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena_curobo.curobo_embodiment_cfg import CuroboEmbodimentCfg

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

_CUROBO_EMBODIMENT_CFGS: dict[str, CuroboEmbodimentCfg] = {}


def register_curobo_cfg(embodiment_name: str, cfg: CuroboEmbodimentCfg) -> None:
    """Register a cuRobo config under an embodiment name, erroring on a duplicate key."""
    assert (
        embodiment_name not in _CUROBO_EMBODIMENT_CFGS
    ), f"A cuRobo config is already registered for embodiment '{embodiment_name}'."
    _CUROBO_EMBODIMENT_CFGS[embodiment_name] = cfg


def get_curobo_cfg_for(embodiment: EmbodimentBase) -> CuroboEmbodimentCfg:
    """Return the cuRobo config registered for an embodiment's robot family.

    Walks the embodiment's class hierarchy so a config registered under a robot-family base name (e.g.
    ``"droid"``) also covers concrete variants that override ``name`` (e.g. ``"droid_abs_joint_pos"``).
    The most-derived match wins, so a variant may register its own override.
    """
    # Each class contributes the ``name`` defined on it directly (not inherited), most-derived first.
    for cls in type(embodiment).__mro__:
        cfg = _CUROBO_EMBODIMENT_CFGS.get(cls.__dict__.get("name"))
        if cfg is not None:
            return cfg
    raise AssertionError(
        f"No cuRobo config registered for embodiment '{embodiment.name}' or its robot family. "
        f"Register one via register_curobo_cfg(...). Known: {sorted(_CUROBO_EMBODIMENT_CFGS)}."
    )


_DROID_CUROBO_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/robot_library/droid/droid_fixed_mimic_joint"

DROID_CUROBO_CFG = CuroboEmbodimentCfg(
    robot_cfg_template=f"{_DROID_CUROBO_ASSET_DIR}/franka_robotiq_2f_85_zero_curobo.yml",
    robot_urdf=f"{_DROID_CUROBO_ASSET_DIR}/urdf/franka_robotiq_2f_85_zero.urdf",
    robot_name="franka_robotiq",
    ee_link_name="base_link",
    gripper_joint_names=["finger_joint"],
    gripper_open_joint_pos={"finger_joint": 0.0},
    gripper_closed_joint_pos={"finger_joint": float(torch.pi / 4)},
    hand_link_names=[
        "base_link",
        "left_inner_finger",
        "left_inner_knuckle",
        "left_outer_finger",
        "left_outer_knuckle",
        "right_inner_finger",
        "right_inner_knuckle",
        "right_outer_finger",
        "right_outer_knuckle",
    ],
)

register_curobo_cfg("droid", DROID_CUROBO_CFG)
