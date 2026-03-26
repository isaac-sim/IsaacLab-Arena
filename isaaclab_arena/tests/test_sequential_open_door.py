# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True


def get_test_environment(remove_reset_door_state_event: bool, num_envs: int):
    """Returns a scene which we use for these tests."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.utils.pose import Pose

    class SequentialOpenDoorTask(SequentialTaskBase):
        def __init__(
            self,
            subtasks,
            episode_length_s=None,
        ):
            super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

        def get_metrics(self):
            return []

        def get_prompt(self):
            return ""

        def get_mimic_env_cfg(self, embodiment_name: str):
            return None

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(num_envs)])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    microwave_0 = asset_registry.get_asset_by_name("microwave")(prim_path="{ENV_REGEX_NS}/microwave_0")
    microwave_1 = asset_registry.get_asset_by_name("microwave")(prim_path="{ENV_REGEX_NS}/microwave_1")

    microwave_0.name = "microwave_0"
    microwave_1.name = "microwave_1"

    # Put the microwave on the packing table.
    microwave_0.set_initial_pose(
        Pose(
            position_xyz=(0.6, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )
    microwave_1.set_initial_pose(
        Pose(
            position_xyz=(0.6, 0.70586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
    )

    subtask_1 = OpenDoorTask(microwave_0)
    subtask_2 = OpenDoorTask(microwave_1)

    scene = Scene(assets=[background, microwave_0, microwave_1])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="sequential_open_door",
        embodiment=FrankaEmbodiment(),
        scene=scene,
        task=SequentialOpenDoorTask([subtask_1, subtask_2]),
    )

    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    if remove_reset_door_state_event:
        # Remove the reset door and subtask state events to allow us to inspect the scene without having it reset.
        cfg.events.reset_door_state_subtask_0 = None
        cfg.events.reset_door_state_subtask_1 = None
        cfg.events.reset_subtask_success_state = None
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    return env, microwave_0, microwave_1


def _test_sequential_open_door_microwave(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave_0, microwave_1 = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_composite_task_incomplete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert not terminated.item()
        if not terminated.item():
            print("Composite task is not completed")

    def assert_composite_task_complete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert terminated.item()
        if terminated.item():
            print("Composite task is completed")

    try:
        print("Closing both microwaves")
        microwave_0.close(env, env_ids=None)
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (completing subtask 0)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1 (completing subtask 1, composite task should be complete)")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_complete)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_out_of_order_sequential_open_door_microwave(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave_0, microwave_1 = get_test_environment(remove_reset_door_state_event=True, num_envs=1)

    def assert_composite_task_incomplete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert not terminated.item()
        if not terminated.item():
            print("Composite task is not completed")

    def assert_composite_task_complete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([1])
        assert terminated.item()
        if terminated.item():
            print("Composite task is completed")

    try:
        print("Closing both microwaves")
        microwave_0.close(env, env_ids=None)
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Closing microwave 1")
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (out of order, composite task should remain incomplete)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Closing microwave 0")
        microwave_0.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (completing subtask 0)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1 (completing subtask 1, composite task should be complete)")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_complete)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sequential_open_door_microwave_multiple_envs(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave_0, microwave_1 = get_test_environment(remove_reset_door_state_event=True, num_envs=2)

    def assert_composite_task_incomplete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([2])
        assert not torch.any(terminated)
        if not torch.any(terminated):
            print("Composite task is not completed")

    def assert_composite_task_complete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([2])
        assert torch.all(terminated)
        if torch.all(terminated):
            print("Composite task is completed")

    try:
        print("Closing both microwaves")
        microwave_0.close(env, env_ids=None)
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (completing subtask 0)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1 (completing subtask 1, composite task should be complete)")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_complete)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_out_of_order_sequential_open_door_microwave_multiple_envs(simulation_app) -> bool:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave_0, microwave_1 = get_test_environment(remove_reset_door_state_event=True, num_envs=2)

    def assert_composite_task_incomplete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([2])
        assert not torch.any(terminated)
        if not torch.any(terminated):
            print("Composite task is not completed")

    def assert_composite_task_complete(env: ManagerBasedEnv, terminated: torch.Tensor):
        assert terminated.shape == torch.Size([2])
        assert torch.all(terminated)
        if torch.all(terminated):
            print("Composite task is completed")

    try:
        print("Closing both microwaves")
        microwave_0.close(env, env_ids=None)
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Closing microwave 1")
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (out of order, composite task should remain incomplete)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Closing microwave 0")
        microwave_0.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 0 (completing subtask 0)")
        microwave_0.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_incomplete)

        print("Opening microwave 1 (completing subtask 1, composite task should be complete)")
        microwave_1.open(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS, assert_composite_task_complete)

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_sequential_open_door_microwave_reset_condition(simulation_app) -> bool:
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call

    # Get the scene
    env, microwave_0, microwave_1 = get_test_environment(remove_reset_door_state_event=False, num_envs=2)

    try:
        print("Closing both microwaves")
        microwave_0.close(env, env_ids=None)
        microwave_1.close(env, env_ids=None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open_0 = microwave_0.is_open(env)
        is_open_1 = microwave_1.is_open(env)
        print(f"expected: [False, False], [False, False]: got: {is_open_0}, {is_open_1}")
        assert torch.all(is_open_0 == torch.tensor([False], device=env.device))
        assert torch.all(is_open_1 == torch.tensor([False], device=env.device))

        print("Opening microwave 0(completing subtask 0)")
        microwave_0.open(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open_0 = microwave_0.is_open(env)
        is_open_1 = microwave_1.is_open(env)
        print(f"expected: [True, True], [False, False]: got: {is_open_0}, {is_open_1}")
        assert torch.all(is_open_0 == torch.tensor([True], device=env.device))
        assert torch.all(is_open_1 == torch.tensor([False], device=env.device))

        # Check that envs automatically reset to closed.
        print("Opening microwave (completing subtask 1)")
        microwave_1.open(env, None)
        step_zeros_and_call(env, NUM_STEPS)
        is_open_0 = microwave_0.is_open(env)
        is_open_1 = microwave_1.is_open(env)
        print(f"expected: [False, False], [False, False]: got: {is_open_0}, {is_open_1}")
        assert torch.all(is_open_0 == torch.tensor([False], device=env.device))
        assert torch.all(is_open_1 == torch.tensor([False], device=env.device))

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_sequential_open_door_microwave():
    result = run_simulation_app_function(
        _test_sequential_open_door_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_open_door_microwave.__name__} failed"


def test_out_of_order_sequential_open_door_microwave():
    result = run_simulation_app_function(
        _test_out_of_order_sequential_open_door_microwave,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_out_of_order_sequential_open_door_microwave.__name__} failed"


def test_sequential_open_door_microwave_multiple_envs():
    result = run_simulation_app_function(
        _test_sequential_open_door_microwave_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_open_door_microwave_multiple_envs.__name__} failed"


def test_out_of_order_sequential_open_door_microwave_multiple_envs():
    result = run_simulation_app_function(
        _test_out_of_order_sequential_open_door_microwave_multiple_envs,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_out_of_order_sequential_open_door_microwave_multiple_envs.__name__} failed"


def test_sequential_open_door_microwave_reset_condition():
    result = run_simulation_app_function(
        _test_sequential_open_door_microwave_reset_condition,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_sequential_open_door_microwave_reset_condition.__name__} failed"


if __name__ == "__main__":
    test_sequential_open_door_microwave()
    test_out_of_order_sequential_open_door_microwave()
    test_sequential_open_door_microwave_multiple_envs()
    test_out_of_order_sequential_open_door_microwave_multiple_envs()
    test_sequential_open_door_microwave_reset_condition()
