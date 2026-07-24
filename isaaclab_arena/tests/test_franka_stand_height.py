# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Franka ``stand_height_m`` USD bake and maple-table env-graph wiring."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

TEST_DATA_DIR = Path(__file__).parent / "test_data"
_CUSTOM_STAND_HEIGHT_M = 0.8
_HEIGHT_ATOL = 1e-3
_CONTACT_ATOL = 5e-2


def _stand_robot_ranges(stage):
    from pxr import Usd, UsdGeom

    from isaaclab_arena.embodiments.franka.franka_stand_usd import assert_franka_on_stand_prims

    root, stand, robot_base = assert_franka_on_stand_prims(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    stand_range = cache.ComputeWorldBound(stand).ComputeAlignedRange()
    robot_range = cache.ComputeWorldBound(robot_base).ComputeAlignedRange()
    return root, stand, robot_base, stand_range, robot_range


def _assert_fixed_bottom_stand_layout(stand_range, robot_range, stand_height_m: float, native_height_m: float) -> None:
    """Stand height is target; bottom stays at ``-H0``; robot sits on stand top."""
    stand_height = float(stand_range.GetSize()[2])
    stand_min_z = float(stand_range.GetMin()[2])
    stand_max_z = float(stand_range.GetMax()[2])
    robot_min_z = float(robot_range.GetMin()[2])
    expected_top_z = stand_height_m - native_height_m

    assert abs(stand_height - stand_height_m) < _HEIGHT_ATOL, stand_height
    assert abs(stand_min_z - (-native_height_m)) < _HEIGHT_ATOL, f"stand bottom not at -H0: min_z={stand_min_z}"
    assert abs(stand_max_z - expected_top_z) < _HEIGHT_ATOL, f"stand top not at H-H0: max_z={stand_max_z}"
    assert abs(robot_min_z - stand_max_z) < _CONTACT_ATOL, (robot_min_z, stand_max_z)


def _test_franka_stand_height_bake_and_maple_env_graph(simulation_app) -> bool:
    import tempfile
    import traceback

    from pxr import Usd

    from isaaclab_arena.embodiments.franka.franka import _FRANKA_ON_STAND_USD_PATH, FrankaIKEmbodiment
    from isaaclab_arena.embodiments.franka.franka_stand_usd import (
        FRANKA_ON_STAND_ROBOT_BASE_PRIM_NAME,
        FRANKA_ON_STAND_STAND_PRIM_NAME,
        ensure_franka_stand_height_usd,
        measure_native_stand_height_m,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    try:
        native_height = measure_native_stand_height_m(_FRANKA_ON_STAND_USD_PATH)
        assert native_height > 0.0

        # Stock height (or None) keeps the Nucleus path; no bake.
        default_emb = FrankaIKEmbodiment()
        assert default_emb.scene_config.robot.spawn.usd_path == _FRANKA_ON_STAND_USD_PATH
        assert ensure_franka_stand_height_usd(_FRANKA_ON_STAND_USD_PATH, native_height) == _FRANKA_ON_STAND_USD_PATH

        with tempfile.TemporaryDirectory() as tmp:
            baked_path = ensure_franka_stand_height_usd(
                _FRANKA_ON_STAND_USD_PATH, _CUSTOM_STAND_HEIGHT_M, cache_dir=Path(tmp)
            )
            assert baked_path != _FRANKA_ON_STAND_USD_PATH
            assert Path(baked_path).is_file()

            stage = Usd.Stage.Open(baked_path)
            assert stage is not None
            _, stand, robot_base, stand_range, robot_range = _stand_robot_ranges(stage)
            assert FRANKA_ON_STAND_STAND_PRIM_NAME in str(stand.GetPath())
            assert FRANKA_ON_STAND_ROBOT_BASE_PRIM_NAME in str(robot_base.GetPath())
            _assert_fixed_bottom_stand_layout(stand_range, robot_range, _CUSTOM_STAND_HEIGHT_M, native_height)

        custom_emb = FrankaIKEmbodiment(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
        assert Path(custom_emb.scene_config.robot.spawn.usd_path).is_file()

        # Maple-table env graph declares stand_height_m and wires it through asset construction.
        spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "minimal_maple_table_env_graph.yaml")
        assert spec.embodiment.params.get("stand_height_m") == _CUSTOM_STAND_HEIGHT_M
        arena_env = spec.to_arena_env()
        assert arena_env.embodiment.stand_height_m == _CUSTOM_STAND_HEIGHT_M
        assert Path(arena_env.embodiment.scene_config.robot.spawn.usd_path).is_file()

        # Missing stand / robot prim names fail loudly at bake time.
        with tempfile.TemporaryDirectory() as tmp:
            empty_path = str(Path(tmp) / "empty_franka_bake.usda")
            empty = Usd.Stage.CreateNew(empty_path)
            empty.DefinePrim("/panda", "Xform")
            empty.SetDefaultPrim(empty.GetPrimAtPath("/panda"))
            empty.GetRootLayer().Save()
            with pytest.raises(AssertionError, match=FRANKA_ON_STAND_STAND_PRIM_NAME):
                ensure_franka_stand_height_usd(empty_path, _CUSTOM_STAND_HEIGHT_M)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_franka_stand_height_bake_and_maple_env_graph():
    """Pytest entry point for Franka stand-height bake and maple env-graph wiring."""
    result = run_simulation_app_function(_test_franka_stand_height_bake_and_maple_env_graph, headless=True)
    assert result, f"Test {test_franka_stand_height_bake_and_maple_env_graph.__name__} failed"


if __name__ == "__main__":
    test_franka_stand_height_bake_and_maple_env_graph()
