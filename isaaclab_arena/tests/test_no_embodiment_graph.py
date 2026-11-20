# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Graph environments can select the registered robot-free embodiment."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_graph_without_robot(_simulation_app):
    from isaaclab_arena.embodiments.no_embodiment import EmptyActionsCfg, NoEmbodiment
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec

    spec = ArenaEnvGraphSpec.from_yaml(Path(__file__).parent / "test_data/minimal_maple_table_env_graph.yaml")
    spec.embodiment = AssetSpec(id="no_robot", registry_name="no_embodiment")

    # The graph converter forwards enable_cameras even when there is no robot to carry them.
    for enable_cameras in (False, True):
        arena_env = spec.to_arena_env(enable_cameras=enable_cameras)
        embodiment = arena_env.embodiment
        assert isinstance(embodiment, NoEmbodiment)
        assert isinstance(embodiment.get_action_cfg(), EmptyActionsCfg)
        assert embodiment.enable_cameras is enable_cameras
        assert embodiment.get_scene_cfg() is None
        assert embodiment.camera_config is None
        assert arena_env.task.pick_up_object is arena_env.scene.assets["rubiks_cube_hot3d_robolab"]

    return True


def test_graph_without_robot():
    assert run_function_with_persistent_simulation_app(_test_graph_without_robot)
