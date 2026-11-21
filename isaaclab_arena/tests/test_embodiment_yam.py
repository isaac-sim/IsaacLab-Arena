# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_yam_embodiment_is_newton_only(simulation_app) -> bool:
    """Check the YAM embodiment spawns the robot_library USD, accepts Newton, rejects PhysX and exposes its cameras."""

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.yam.yam import YAM_USD_PATH, YamEmbodiment
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    ensure_assets_registered()
    assert AssetRegistry().get_component_by_name("yam") is YamEmbodiment

    embodiment = YamEmbodiment()
    robot = embodiment.get_scene_cfg().robot
    assert robot.spawn.usd_path == YAM_USD_PATH
    assert YAM_USD_PATH.endswith("/Arena/assets/robot_library/yam/i2rt_yam_default.usda")
    assert embodiment.get_command_body_name() == "link_6"

    camera_rig = YamEmbodiment(enable_cameras=True).get_camera_cfg()
    assert camera_rig.camera_names() == ["wrist_camera", "top_camera"]
    assert camera_rig.wrist_camera.prim_path.endswith(
        "/Robot/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/wrist_camera"
    )
    assert camera_rig.top_camera.prim_path.endswith("/Robot/Geometry/arm/top_camera")

    embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    try:
        YamEmbodiment().configure_physics_backend(PhysicsBackend.PHYSX)
    except AssertionError as error:
        assert "requires the Newton physics backend" in str(error)
    else:
        raise AssertionError("YAM must reject the PhysX backend")
    return True


def test_yam_embodiment_is_newton_only():
    assert run_function_with_persistent_simulation_app(_test_yam_embodiment_is_newton_only)
