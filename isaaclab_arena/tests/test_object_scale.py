# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for object scale parameter (LibraryObject and subclasses)."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_object_scale_default_and_override(simulation_app):
    from isaaclab_arena.assets.asset_registry import AssetRegistry

    asset_registry = AssetRegistry()

    cracker_box_default = asset_registry.get_asset_by_name("cracker_box")()
    assert cracker_box_default.scale == (1.0, 1.0, 1.0)

    # Same object with explicit scale override
    custom_scale = (2.0, 2.0, 2.0)
    cracker_box_scaled = asset_registry.get_asset_by_name("cracker_box")(scale=custom_scale)
    assert cracker_box_scaled.scale == custom_scale

    dex_cube_default = asset_registry.get_asset_by_name("dex_cube")()
    assert dex_cube_default.scale == (0.8, 0.8, 0.8)

    # Override object's own scale
    override_scale = (0.5, 0.5, 0.5)
    dex_cube_scaled = asset_registry.get_asset_by_name("dex_cube")(scale=override_scale)
    assert dex_cube_scaled.scale == override_scale

    dex_cube_none = asset_registry.get_asset_by_name("dex_cube")(scale=None)
    assert dex_cube_none.scale == (0.8, 0.8, 0.8)

    return True


def test_object_scale_default_and_override():
    result = run_simulation_app_function(
        _test_object_scale_default_and_override,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_object_scale_default_and_override()
