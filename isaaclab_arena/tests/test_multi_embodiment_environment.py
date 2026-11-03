# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test environments with several robots: composition, recorded identity, and refused compositions."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def make_two_robot_definition(enable_cameras=False, relations=False):
    """Build two spaced keyed Frankas with optional cameras and placement relations."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    robots = [FrankaJointPosEmbodiment(instance_key=key, enable_cameras=enable_cameras) for key in ("left", "right")]
    for robot, x in zip(robots, (-2.0, 2.0)):
        if relations:
            robot.add_relation(AtPosition(x=x, y=0.0, z=0.0))
        else:
            robot.set_initial_pose(Pose(position_xyz=(x, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        if enable_cameras:
            robot.get_variations()[0].enable()
    assets = [registry.get_asset_by_name(name)() for name in ("ground_plane", "light")]
    if relations:
        anchor = registry.get_asset_by_name("dex_cube")()
        anchor.set_initial_pose(Pose(position_xyz=(0.0, 5.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        anchor.add_relation(IsAnchor())
        assets.append(anchor)
    return IsaacLabArenaEnvironment(
        name="multi_robot_test",
        scene=Scene(assets=assets),
        embodiments=robots,
        # Allow both robots to travel from the shared anchor to their requested positions.
        placer_params=ObjectPlacerParams(
            solver_params=RelationSolverParams(max_iters=4000), min_unique_layouts_per_env=1
        ),
    )


def _test_invalid_compositions(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    # Several robots each need an instance key, even when their scene keys differ.
    for robots in (
        [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()],
        [FrankaJointPosEmbodiment(instance_key="left"), FrankaJointPosEmbodiment()],
    ):
        with pytest.raises(AssertionError, match="instance key on every robot"):
            IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=robots)
    with pytest.raises(AssertionError, match="unique"):
        IsaacLabArenaEnvironment(
            "invalid",
            Scene(assets=[]),
            embodiments=[FrankaJointPosEmbodiment(instance_key="arm"), FrankaJointPosEmbodiment(instance_key="arm")],
        )
    # No robot, or one robot alone, needs no key.
    for robots in ([], [FrankaJointPosEmbodiment()]):
        IsaacLabArenaEnvironment("valid", Scene(assets=[]), embodiments=robots)

    definition = make_two_robot_definition()
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(mimic=True)).build_env_cfg()
    definition.teleop_device = object()
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).build_env_cfg()
    definition.teleop_device = None
    with patch(
        "isaaclab_arena.environments.arena_env_builder.get_settings_manager",
        return_value=SimpleNamespace(get=lambda *args: True),
    ):
        with pytest.raises(AssertionError, match="XR require exactly one"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).build_env_cfg()
    # The builder checks robots added after construction as well.
    definition.embodiments.append(definition.embodiments[0])
    with pytest.raises(AssertionError, match="unique"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).build_env_cfg()
    return True


def test_invalid_compositions_fail_before_building():
    assert run_function_with_persistent_simulation_app(_test_invalid_compositions)
