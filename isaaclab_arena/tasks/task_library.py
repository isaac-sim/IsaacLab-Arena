# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Importing these concrete task modules fires their @register_task decorators. This must
# NOT live in tasks/__init__.py: that runs on any `from isaaclab_arena.tasks.X import Y`,
# including at pytest-collection time, and the cascade pulls in pxr/USD modules before
# SimulationApp() is created — which segfaults the sim. Instead it is imported lazily by
# ensure_assets_registered() (called after the sim app is up), mirroring object_library.
from isaaclab_arena.tasks import (  # noqa: F401
    assembly_task,
    close_door_task,
    deformable_pick_and_place_task,
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
