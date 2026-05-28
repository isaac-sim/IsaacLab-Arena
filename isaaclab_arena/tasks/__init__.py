# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Import concrete task modules so their @register_task decorators fire when the
# tasks package is loaded (e.g. via ensure_assets_registered()).
from isaaclab_arena.tasks import (  # noqa: F401
    assembly_task,
    close_door_task,
    goal_pose_task,
    lift_object_task,
    no_task,
    open_door_task,
    pick_and_place_task,
    place_upright_task,
    press_button_task,
    rotate_revolute_joint_task,
    sequential_task_base,
    sorting_task,
    turn_knob_task,
)
