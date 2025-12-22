# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = False


def _test_object_with_cfg_addons(simulation_app):

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.utils.pose import Pose

    class ConeWithCfgAddons(LibraryObject):
        """
        Cone with cfg addons.
        """

        name = "cone_with_cfg_addons"
        tags = ["object"]
        usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Shapes/cone.usd"
        default_prim_path = "{ENV_REGEX_NS}/target_cone_with_cfg_addons"
        object_type = ObjectType.RIGID
        spawn_cfg_addon = {"visible": False}  # By default, the object is visible.
        asset_cfg_addon = {"debug_vis": True}  # By default, the object is not debug visualized.

        def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
            super().__init__(prim_path=prim_path, initial_pose=initial_pose)

    cone = ConeWithCfgAddons()

    # Check that the settings have been applied
    assert cone.object_cfg.spawn.visible is False
    assert cone.object_cfg.debug_vis is True

    return True


def test_object_with_cfg_addons():
    result = run_simulation_app_function(
        _test_object_with_cfg_addons,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_with_cfg_addons()
