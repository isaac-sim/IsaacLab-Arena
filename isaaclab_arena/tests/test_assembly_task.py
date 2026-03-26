# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import gymnasium as gym
import torch

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_peg_insert_test_environment(num_envs: int, remove_events: bool = False):
    """Returns a peg insert environment for testing."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.assembly_task import AssemblyTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments import mdp

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])
    args_cli.enable_pinocchio = False

    asset_registry = AssetRegistry()

    # Create scene assets
    background = asset_registry.get_asset_by_name("table")()
    background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))

    peg = asset_registry.get_asset_by_name("peg")()
    peg.set_initial_pose(Pose(position_xyz=(0.45, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    hole = asset_registry.get_asset_by_name("hole")()
    hole.set_initial_pose(Pose(position_xyz=(0.45, 0.1, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
    light = asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

    # Create embodiment
    embodiment = FrankaEmbodiment()
    embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene = Scene(assets=[background, peg, hole, light])

    # Create assembly task
    task = AssemblyTask(
        task_description="Assemble the peg with the hole",
        fixed_asset=peg,
        held_asset=hole,
        auxiliary_asset_list=[],
        background_scene=background,
        pose_range={"x": (0.2, 0.6), "y": (-0.20, 0.20), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
        min_separation=0.1,
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_peg_insert",
        embodiment=embodiment,
        scene=scene,
        task=task,
        env_cfg_callback=mdp.assembly_env_cfg_callback,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()

    if remove_events:
        cfg.events.reset_all = None
        cfg.events.randomize_asset_positions = None

    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, peg, hole


def get_gear_mesh_test_environment(num_envs: int, remove_events: bool = False):
    """Returns a gear mesh environment for testing."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.assembly_task import AssemblyTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments import mdp

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])
    args_cli.enable_pinocchio = False

    asset_registry = AssetRegistry()

    # Create scene assets
    background = asset_registry.get_asset_by_name("table")()
    background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))

    gear_base = asset_registry.get_asset_by_name("gear_base")()
    gear_base.set_initial_pose(Pose(position_xyz=(0.6, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    medium_gear = asset_registry.get_asset_by_name("medium_gear")()
    medium_gear.set_initial_pose(Pose(position_xyz=(0.5, 0.2, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    small_gear = asset_registry.get_asset_by_name("small_gear")()
    small_gear.set_initial_pose(Pose(position_xyz=(0.6, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    large_gear = asset_registry.get_asset_by_name("large_gear")()
    large_gear.set_initial_pose(Pose(position_xyz=(0.6, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

    light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
    light = asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

    # Create embodiment
    embodiment = FrankaEmbodiment()
    embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene = Scene(assets=[background, gear_base, medium_gear, small_gear, large_gear, light])

    # Create gear mesh task
    task = AssemblyTask(
        task_description="Mesh the medium gear with the gear base",
        fixed_asset=gear_base,
        held_asset=medium_gear,
        auxiliary_asset_list=[small_gear, large_gear],
        background_scene=background,
        pose_range={"x": (0.25, 0.6), "y": (-0.20, 0.20), "z": (0.0, 0.0), "yaw": (-1.0, 1.0)},
        min_separation=0.18,
        randomization_mode="held_fixed_and_auxiliary",
    )

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="test_gear_mesh",
        embodiment=embodiment,
        scene=scene,
        task=task,
        env_cfg_callback=mdp.assembly_env_cfg_callback,
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()

    if remove_events:
        cfg.events.reset_all = None
        cfg.events.randomize_asset_positions = None

    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, gear_base, medium_gear, small_gear, large_gear


def _test_peg_insert_assembly_single(simulation_app) -> bool:
    """Test peg insert assembly with single environment."""
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, peg, hole = get_peg_insert_test_environment(num_envs=1, remove_events=True)

    def assert_assembled(env: ManagerBasedEnv, terminated: torch.Tensor):
        # Check if objects are close together (assembled)
        peg_pos = peg.get_object_pose(env, is_relative=True)[:, :3]
        hole_pos = hole.get_object_pose(env, is_relative=True)[:, :3]
        distance = torch.norm(peg_pos - hole_pos, dim=-1)

        print(f"Distance between peg and hole: {distance.item():.4f}m")
        assert distance.item() < 0.05, f"Objects not assembled, distance: {distance.item():.4f}m"

        # Check terminated
        assert terminated.shape == torch.Size([1]), "Terminated shape is not correct"
        assert terminated.item(), "Task didn't terminate when it should have"
        print("✓ Peg insert assembly task completed successfully")

    try:
        print("Testing peg insert assembly (single env)...")

        # Manually place hole on peg to simulate successful assembly - use absolute world coordinates
        peg_pose = peg.get_object_pose(env, is_relative=False)
        env.scene[hole.name].write_root_pose_to_sim(peg_pose, env_ids=torch.tensor([0], device=env.device))

        step_zeros_and_call(env, NUM_STEPS, assert_assembled)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_gear_mesh_assembly_single(simulation_app) -> bool:
    """Test gear mesh assembly with single environment."""
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, gear_base, medium_gear, small_gear, large_gear = get_gear_mesh_test_environment(num_envs=1, remove_events=True)

    def assert_assembled(env: ManagerBasedEnv, terminated: torch.Tensor):
        # Check if gears are close together (meshed)
        base_pos = gear_base.get_object_pose(env, is_relative=True)[:, :3]
        medium_pos = medium_gear.get_object_pose(env, is_relative=True)[:, :3]
        distance = torch.norm(base_pos - medium_pos, dim=-1)

        print(f"Distance between gear base and medium gear: {distance.item():.4f}m")
        assert distance.item() < 0.05, f"Gears not meshed, distance: {distance.item():.4f}m"

        # Check terminated
        assert terminated.shape == torch.Size([1]), "Terminated shape is not correct"
        assert terminated.item(), "Task didn't terminate when it should have"
        print("✓ Gear mesh assembly task completed successfully")

    try:
        print("Testing gear mesh assembly (single env)...")

        # Manually place medium gear on gear base to simulate successful assembly - use absolute world coordinates
        base_pose = gear_base.get_object_pose(env, is_relative=False)
        env.scene[medium_gear.name].write_root_pose_to_sim(base_pose, env_ids=torch.tensor([0], device=env.device))

        step_zeros_and_call(env, NUM_STEPS, assert_assembled)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_peg_insert_assembly_multi(simulation_app) -> bool:
    """Test peg insert assembly with multiple environments."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, peg, hole = get_peg_insert_test_environment(num_envs=2, remove_events=True)

    try:
        with torch.inference_mode():
            print("Testing peg insert assembly (multi env)...")

            # Assemble in both environments - use absolute world coordinates
            peg_poses = peg.get_object_pose(env, is_relative=False)
            env.scene[hole.name].write_root_pose_to_sim(peg_poses, env_ids=None)

            step_zeros_and_call(env, NUM_STEPS)

            # Check distances
            peg_pos = peg.get_object_pose(env, is_relative=True)[:, :3]
            hole_pos = hole.get_object_pose(env, is_relative=True)[:, :3]
            distances = torch.norm(peg_pos - hole_pos, dim=-1)

            print(f"Distances in both envs: {distances}")
            assert torch.all(distances < 0.05), f"Not all environments assembled: {distances}"
            print("✓ Multi-environment peg insert assembly successful")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_gear_mesh_assembly_multi(simulation_app) -> bool:
    """Test gear mesh assembly with multiple environments."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    env, gear_base, medium_gear, small_gear, large_gear = get_gear_mesh_test_environment(num_envs=2, remove_events=True)

    try:
        with torch.inference_mode():
            print("Testing gear mesh assembly (multi env)...")

            # Assemble in both environments - use absolute world coordinates
            base_poses = gear_base.get_object_pose(env, is_relative=False)
            env.scene[medium_gear.name].write_root_pose_to_sim(base_poses, env_ids=None)

            step_zeros_and_call(env, NUM_STEPS)

            # Check distances
            base_pos = gear_base.get_object_pose(env, is_relative=True)[:, :3]
            medium_pos = medium_gear.get_object_pose(env, is_relative=True)[:, :3]
            distances = torch.norm(base_pos - medium_pos, dim=-1)

            print(f"Distances in both envs: {distances}")
            assert torch.all(distances < 0.05), f"Not all environments assembled: {distances}"
            print("✓ Multi-environment gear mesh assembly successful")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        env.close()

    return True


def _test_peg_insert_initialization(simulation_app) -> bool:
    """Test that peg insert task initializes correctly with proper asset configurations."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    try:
        print("Testing peg insert task initialization...")

        # Test peg insert initialization
        env, peg, hole = get_peg_insert_test_environment(num_envs=1, remove_events=False)

        # Check that objects exist in scene
        assert peg.name in env.scene.keys(), f"Peg '{peg.name}' not found in scene"
        assert hole.name in env.scene.keys(), f"Hole '{hole.name}' not found in scene"

        # Run a few steps to ensure stability
        step_zeros_and_call(env, NUM_STEPS)

        print("✓ Peg insert task initialized successfully")
        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def _test_gear_mesh_initialization(simulation_app) -> bool:
    """Test that gear mesh task initializes correctly with proper asset configurations."""
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    try:
        print("Testing gear mesh task initialization...")

        # Test gear mesh initialization
        env, gear_base, medium_gear, small_gear, large_gear = get_gear_mesh_test_environment(
            num_envs=1, remove_events=False
        )

        # Check that all gears exist in scene
        assert gear_base.name in env.scene.keys(), f"Gear base '{gear_base.name}' not found in scene"
        assert medium_gear.name in env.scene.keys(), f"Medium gear '{medium_gear.name}' not found in scene"
        assert small_gear.name in env.scene.keys(), f"Small gear '{small_gear.name}' not found in scene"
        assert large_gear.name in env.scene.keys(), f"Large gear '{large_gear.name}' not found in scene"

        # Run a few steps to ensure stability
        step_zeros_and_call(env, NUM_STEPS)

        print("✓ Gear mesh task initialized successfully")
        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


# Test functions that will be called by pytest
@pytest.mark.skip(
    reason=(
        "Requires enable_pinocchio=False. Run separately: pytest -sv"
        " isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_assembly_single"
    )
)
def test_peg_insert_assembly_single():
    result = run_simulation_app_function(_test_peg_insert_assembly_single, headless=HEADLESS, enable_pinocchio=False)
    assert result, f"Test {_test_peg_insert_assembly_single.__name__} failed"


def test_gear_mesh_assembly_single():
    result = run_simulation_app_function(_test_gear_mesh_assembly_single, headless=HEADLESS)
    assert result, f"Test {_test_gear_mesh_assembly_single.__name__} failed"


@pytest.mark.skip(
    reason=(
        "Requires enable_pinocchio=False. Run separately: pytest -sv"
        " isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_assembly_multi"
    )
)
def test_peg_insert_assembly_multi():
    result = run_simulation_app_function(_test_peg_insert_assembly_multi, headless=HEADLESS, enable_pinocchio=False)
    assert result, f"Test {_test_peg_insert_assembly_multi.__name__} failed"


def test_gear_mesh_assembly_multi():
    result = run_simulation_app_function(_test_gear_mesh_assembly_multi, headless=HEADLESS)
    assert result, f"Test {_test_gear_mesh_assembly_multi.__name__} failed"


@pytest.mark.skip(
    reason=(
        "Requires enable_pinocchio=False. Run separately: pytest -sv"
        " isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_initialization"
    )
)
def test_peg_insert_initialization():
    """
    For peg insert task, we need to test the task with pinocchio disabled due to the "peg" and "hole" assets are not compatible with pinocchio.
    """
    result = run_simulation_app_function(_test_peg_insert_initialization, headless=HEADLESS, enable_pinocchio=False)
    assert result, f"Test {_test_peg_insert_initialization.__name__} failed"


def test_gear_mesh_initialization():
    result = run_simulation_app_function(_test_gear_mesh_initialization, headless=HEADLESS)
    assert result, f"Test {_test_gear_mesh_initialization.__name__} failed"


if __name__ == "__main__":
    """
    Peg insert tests are commented out because they require enable_pinocchio=False,
    but the current test session's SimulationApp was initialized with enable_pinocchio=True.
    Due to limitations in subprocess.py, the SimulationApp cannot be restarted with different
    parameters during a single pytest session. Run peg insert tests separately with:
      pytest -sv isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_assembly_single --disable_pinocchio
      pytest -sv isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_assembly_multi --disable_pinocchio
      pytest -sv isaaclab_arena/tests/test_assembly_task.py::test_peg_insert_initialization --disable_pinocchio
    """
    test_gear_mesh_initialization()
    test_gear_mesh_assembly_single()
    test_gear_mesh_assembly_multi()
