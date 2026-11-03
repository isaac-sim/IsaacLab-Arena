# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_franka_reach_spheres_are_centred_on_the_shoulder_joint(simulation_app) -> bool:
    """Franka and DROID read their reach centre from panda_joint2, 0.333 m above panda_link0."""

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment

    try:
        for embodiment in (DroidAbsoluteJointPositionEmbodiment(), FrankaIKEmbodiment()):
            reach_spheres = embodiment.get_reach_spheres()
            assert len(reach_spheres) == 1, f"{embodiment.name}: expected one arm, got {reach_spheres}"
            assert reach_spheres[0].reach_m == 0.855
            center_xyz_R = reach_spheres[0].center_xyz_R
            assert all(
                abs(actual - expected) < 1e-3 for actual, expected in zip(center_xyz_R, (0.0, 0.0, 0.333))
            ), f"{embodiment.name}: reach centre {center_xyz_R} is not the shoulder joint"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_franka_reach_spheres_are_centred_on_the_shoulder_joint():
    result = run_function_with_persistent_simulation_app(_test_franka_reach_spheres_are_centred_on_the_shoulder_joint)
    assert result, "Test failed"
