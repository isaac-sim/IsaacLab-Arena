# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_CUSTOM_STAND_HEIGHT_M = 2.0


def _test_droid_stand_height(simulation_app) -> bool:
    """Check ``stand_height_m`` (absolute meters) sets the stand z-scale and lifts the robot base."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.droid.droid import (
        _DEFAULT_STAND_HEIGHT_M,
        _STAND_FOOTPRINT_SCALE_XY,
        DroidAbsoluteJointPositionEmbodiment,
        _stand_unit_height_m,
    )
    from isaaclab_arena.utils.pose import Pose

    try:
        # ``stand_height_m`` is an absolute height in meters, converted to a z-scale via the stand's
        # native (scale=1.0) height.
        default_emb = DroidAbsoluteJointPositionEmbodiment()
        unit_height = _stand_unit_height_m(default_emb.scene_config.stand.spawn.usd_path)
        expected_default_scale = (*_STAND_FOOTPRINT_SCALE_XY, _DEFAULT_STAND_HEIGHT_M / unit_height)
        expected_custom_scale = (*_STAND_FOOTPRINT_SCALE_XY, _CUSTOM_STAND_HEIGHT_M / unit_height)

        # The default leaves the robot base at z=0 (no lift relative to the default height).
        for got, want in zip(default_emb.scene_config.stand.spawn.scale, expected_default_scale):
            assert abs(got - want) < 1e-6
        assert default_emb.scene_config.robot.init_state.pos[2] == 0.0

        # The lift is the height delta from the default, in meters.
        expected_offset = _CUSTOM_STAND_HEIGHT_M - _DEFAULT_STAND_HEIGHT_M

        # An override changes only the z-scale (x/y footprint and robot mesh untouched)...
        custom_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
        for got, want in zip(custom_emb.scene_config.stand.spawn.scale, expected_custom_scale):
            assert abs(got - want) < 1e-6
        assert custom_emb.scene_config.robot.spawn.scale in (None, (1.0, 1.0, 1.0))

        # ...and lifts the robot base and stand together so the stand's floor contact is preserved.
        assert abs(custom_emb.scene_config.robot.init_state.pos[2] - expected_offset) < 1e-6
        assert abs(custom_emb.scene_config.stand.init_state.pos[2] - expected_offset) < 1e-6

        # The lift is re-applied on top of an explicit initial_pose (which the base class would
        # otherwise overwrite): the requested z plus the stand-height offset.
        posed_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
        posed_emb.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        scene_cfg = posed_emb.get_scene_cfg()
        assert abs(scene_cfg.robot.init_state.pos[2] - (0.5 + expected_offset)) < 1e-6
        assert abs(scene_cfg.stand.init_state.pos[2] - (0.5 + expected_offset)) < 1e-6

        # The YAML-spec path instantiates the embodiment via asset_class(**params); a scalar
        # stand_height_m from YAML applies just the same.
        registry_emb = AssetRegistry().get_asset_by_name("droid_abs_joint_pos")(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
        for got, want in zip(registry_emb.scene_config.stand.spawn.scale, expected_custom_scale):
            assert abs(got - want) < 1e-6

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_droid_stand_height():
    """Pytest entry point for the Droid stand-height configuration test."""
    result = run_simulation_app_function(_test_droid_stand_height, headless=True)
    assert result, f"Test {test_droid_stand_height.__name__} failed"


if __name__ == "__main__":
    test_droid_stand_height()
