# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_object_initial_pose_update(simulation_app):

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    # Get a rigid object
    rigid_object = asset_registry.get_asset_by_name("cracker_box")()
    # Disable debug visualization, this is True by default.
    rigid_object.object_cfg.debug_vis = False

    # Now lets add an initial pose to the object.
    new_initial_pose = Pose(position_xyz=(5.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    rigid_object.set_initial_pose(new_initial_pose)

    # Now lets check that the initial pose has been updated and that the debug visualization is still disabled.
    assert rigid_object.get_initial_pose() == new_initial_pose
    assert rigid_object.object_cfg.debug_vis is False

    return True


def _test_articulation_pose_reset_restores_default_joints(simulation_app):
    from types import SimpleNamespace

    from isaaclab_arena.assets.object_base import RootedObjectBase
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    from isaaclab_arena.utils.pose import Pose

    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    articulation = SimpleNamespace(
        name="cabinet",
        object_type=ObjectType.ARTICULATION,
        initial_velocity=None,
        _requires_reset_pose_event=lambda: True,
        get_initial_pose=lambda: initial_pose,
    )

    event_cfg = RootedObjectBase._build_reset_event(articulation)

    assert event_cfg is not None
    assert event_cfg.func is reset_articulation_pose_and_joints
    return True


def test_object_configuration():
    result = run_function_with_persistent_simulation_app(
        _test_object_initial_pose_update,
        headless=HEADLESS,
    )
    assert result, "Test failed"


def test_articulation_pose_reset_restores_default_joints():
    result = run_function_with_persistent_simulation_app(
        _test_articulation_pose_reset_restores_default_joints,
        headless=HEADLESS,
    )
    assert result, "Articulation pose reset did not include its configured default joint state"


if __name__ == "__main__":
    test_object_configuration()
