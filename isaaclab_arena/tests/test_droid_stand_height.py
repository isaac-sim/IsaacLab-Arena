# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Droid stand height, placement bbox, and reset alignment tests."""

from __future__ import annotations

import torch
import traceback
from collections.abc import Callable
from pathlib import Path

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

_CUSTOM_STAND_HEIGHT_M = 2.0
_KITCHEN_STAND_HEIGHT_M = 0.8
_KITCHEN_YAML = (
    Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "droid_pick_and_place_lightwheel_kitchen.yaml"
)
_Z_MATCH_EPS = 1e-3


def _check_stand_height_scaling(simulation_app) -> None:
    """``stand_height_m`` sets stand z-scale and lifts the default robot/stand base together."""
    from isaaclab_arena.embodiments.droid.droid import (
        _DEFAULT_STAND_HEIGHT_M,
        _STAND_FOOTPRINT_SCALE_XY,
        DroidAbsoluteJointPositionEmbodiment,
        _stand_unit_height_m,
    )

    default_emb = DroidAbsoluteJointPositionEmbodiment()
    unit_height = _stand_unit_height_m(default_emb.scene_config.stand.spawn.usd_path)
    expected_default_scale = (*_STAND_FOOTPRINT_SCALE_XY, _DEFAULT_STAND_HEIGHT_M / unit_height)
    expected_custom_scale = (*_STAND_FOOTPRINT_SCALE_XY, _CUSTOM_STAND_HEIGHT_M / unit_height)
    expected_offset = _CUSTOM_STAND_HEIGHT_M - _DEFAULT_STAND_HEIGHT_M

    for got, want in zip(default_emb.scene_config.stand.spawn.scale, expected_default_scale):
        assert abs(got - want) < 1e-6
    assert default_emb.scene_config.robot.init_state.pos[2] == 0.0

    custom_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
    for got, want in zip(custom_emb.scene_config.stand.spawn.scale, expected_custom_scale):
        assert abs(got - want) < 1e-6
    assert custom_emb.scene_config.robot.spawn.scale in (None, (1.0, 1.0, 1.0))
    assert abs(custom_emb.scene_config.robot.init_state.pos[2] - expected_offset) < 1e-6
    assert abs(custom_emb.scene_config.stand.init_state.pos[2] - expected_offset) < 1e-6


def _check_robot_stand_pose_wiring(simulation_app) -> None:
    """Lifted robot poses drive stand scene cfg and reset scene writes via the stand root offset."""
    from isaaclab_arena.embodiments.droid.droid import (
        _DEFAULT_STAND_HEIGHT_M,
        _STAND_ROOT_OFFSET_IN_ROBOT_FRAME,
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.utils.pose import Pose

    expected_offset = _CUSTOM_STAND_HEIGHT_M - _DEFAULT_STAND_HEIGHT_M
    posed_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_CUSTOM_STAND_HEIGHT_M)
    posed_emb.set_initial_pose(Pose(position_xyz=(0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    assert abs(posed_emb.initial_pose.position_xyz[2] - (0.5 + expected_offset)) < 1e-6

    scene_cfg = posed_emb.get_scene_cfg()
    assert tuple(posed_emb.initial_pose.position_xyz) == tuple(scene_cfg.robot.init_state.pos)
    assert abs(scene_cfg.stand.init_state.pos[0] - (0.3 + _STAND_ROOT_OFFSET_IN_ROBOT_FRAME[0])) < 1e-6
    assert abs(scene_cfg.stand.init_state.pos[2] - scene_cfg.robot.init_state.pos[2]) < 1e-6

    layout_pose = Pose(position_xyz=(0.3, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    scene_writes = posed_emb.layout_pose_to_scene_writes(layout_pose)
    assert scene_writes[0][0] == "robot"
    assert scene_writes[1][0] == "stand"
    robot_write_pose = scene_writes[0][1]
    stand_write_pose = scene_writes[1][1]
    assert (
        abs(
            stand_write_pose.position_xyz[0] - (robot_write_pose.position_xyz[0] + _STAND_ROOT_OFFSET_IN_ROBOT_FRAME[0])
        )
        < 1e-6
    )
    assert abs(stand_write_pose.position_xyz[2] - robot_write_pose.position_xyz[2]) < 1e-6


def _check_placement_bbox(simulation_app) -> None:
    """Placement bounds follow the stand footprint, including stand root and stand-height offsets."""
    from isaaclab_arena.embodiments.droid.droid import (
        _STAND_ROOT_OFFSET_IN_ROBOT_FRAME,
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.utils.pose import Pose, translate_by_xyz_offset
    from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_usd

    kitchen_emb = DroidAbsoluteJointPositionEmbodiment(stand_height_m=_KITCHEN_STAND_HEIGHT_M)
    bbox = kitchen_emb.get_bounding_box()
    stand = kitchen_emb.scene_config.stand
    stand_bbox = compute_local_bounding_box_from_usd(stand.spawn.usd_path, tuple(stand.spawn.scale))
    stand_offset = translate_by_xyz_offset(_STAND_ROOT_OFFSET_IN_ROBOT_FRAME, kitchen_emb._robot_base_offset)
    expected_bbox = stand_bbox.translated(stand_offset)
    assert torch.allclose(bbox.min_point, expected_bbox.min_point)
    assert torch.allclose(bbox.max_point, expected_bbox.max_point)

    robot_bbox = compute_local_bounding_box_from_usd(
        kitchen_emb.scene_config.robot.spawn.usd_path,
        tuple(kitchen_emb.scene_config.robot.spawn.scale or (1.0, 1.0, 1.0)),
    )
    assert bbox.size[0, 0].item() > robot_bbox.size[0, 0].item()
    assert bbox.size[0, 1].item() > robot_bbox.size[0, 1].item()
    assert bbox.min_point[0, 2].item() <= 0.0

    floor_top_z = 0.0
    solver_z = floor_top_z - bbox.min_point[0, 2].item()
    kitchen_emb.set_initial_pose(Pose(position_xyz=(0.0, 0.0, solver_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    spawn_z = kitchen_emb.initial_pose.position_xyz[2]
    natural_stand_bottom_z = spawn_z + (expected_bbox.min_point[0, 2].item() - stand_offset[2])
    assert abs(natural_stand_bottom_z - floor_top_z) < _Z_MATCH_EPS


def _check_kitchen_reset_alignment(simulation_app) -> None:
    """Robot and stand roots stay z-aligned after relation placement reset in the kitchen task."""
    import warp as wp

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    spec = ArenaEnvGraphSpec.from_yaml(_KITCHEN_YAML)
    params = dict(spec.embodiment.params)
    params["stand_height_m"] = _KITCHEN_STAND_HEIGHT_M
    spec.embodiment.params = params

    builder = ArenaEnvBuilder(spec.to_arena_env(), ArenaEnvBuilderCfg(num_envs=1))
    env = builder.make_registered()
    try:
        env.reset()
        robot_z = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0, 2].item()
        stand_z = env.unwrapped.scene["stand"].get_world_poses()[0].torch[0, 2].item()
        assert abs(robot_z - stand_z) < _Z_MATCH_EPS, f"robot z {robot_z} != stand z {stand_z} after reset"
    finally:
        env.close()


_DROID_STAND_CHECKS: tuple[Callable[..., None], ...] = (
    _check_stand_height_scaling,
    _check_robot_stand_pose_wiring,
    _check_placement_bbox,
    _check_kitchen_reset_alignment,
)


def _test_droid_stand_height(simulation_app) -> bool:
    """Run Droid stand/placement checks in one SimulationApp session."""
    for check in _DROID_STAND_CHECKS:
        try:
            check(simulation_app)
        except Exception as exc:
            print(f"Error in {check.__name__}: {exc}")
            traceback.print_exc()
            return False
    return True


def test_droid_stand_height():
    """Pytest entry point for Droid stand height, placement bbox, and reset alignment."""
    result = run_simulation_app_function(_test_droid_stand_height, headless=True)
    assert result, f"Test {test_droid_stand_height.__name__} failed"


if __name__ == "__main__":
    test_droid_stand_height()
