# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_DEFAULT_STAND_SCALE = (1.2, 1.2, 1.7)
_CUSTOM_STAND_SCALE = (1.5, 1.5, 2.0)


def _test_droid_stand_scale(simulation_app) -> bool:
    """Check that ``stand_scale`` threads through to the stand spawn without touching the robot."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

    try:
        # The default is applied when the caller does not set stand_scale.
        default_emb = DroidAbsoluteJointPositionEmbodiment()
        assert tuple(default_emb.scene_config.stand.spawn.scale) == _DEFAULT_STAND_SCALE

        # An override reaches the stand spawn config, leaving the robot articulation at native scale.
        custom_emb = DroidAbsoluteJointPositionEmbodiment(stand_scale=_CUSTOM_STAND_SCALE)
        assert tuple(custom_emb.scene_config.stand.spawn.scale) == _CUSTOM_STAND_SCALE
        assert custom_emb.scene_config.robot.spawn.scale in (None, (1.0, 1.0, 1.0))

        # The YAML-spec path instantiates the embodiment via asset_class(**params); stand_scale
        # arriving as a list (as it would from YAML) applies just the same.
        registry_emb = AssetRegistry().get_asset_by_name("droid_abs_joint_pos")(stand_scale=list(_CUSTOM_STAND_SCALE))
        assert tuple(registry_emb.scene_config.stand.spawn.scale) == _CUSTOM_STAND_SCALE

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_droid_stand_scale():
    """Pytest entry point for the Droid stand-scale configuration test."""
    result = run_simulation_app_function(_test_droid_stand_scale, headless=True)
    assert result, f"Test {test_droid_stand_scale.__name__} failed"


if __name__ == "__main__":
    test_droid_stand_scale()
