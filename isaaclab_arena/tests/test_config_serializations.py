# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import gymnasium as gym
from tqdm import tqdm
import torch

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

CFG_YAML_PATH = "/tmp/test_config_serializations.yaml"
NUM_STEPS = 100
HEADLESS = True

def _test_config_serializations(simulation_app) -> bool:

    from isaaclab_arena.utils.config_serialization import load_env_cfg_from_yaml
    from isaaclab_arena.metrics.metrics import compute_metrics
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    embodiment = asset_registry.get_asset_by_name("gr1_joint")()

    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(0.4, 0.0, 0.1),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )
    destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
            parent_asset=background,
            object_type=ObjectType.RIGID,
    )

    scene = Scene(assets=[background, cracker_box, destination_location])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="kitchen_pick_and_place",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
        )

    try:
        env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
        env_builder.serialization_file_path = CFG_YAML_PATH
        cfg_entry_from_cli = env_builder.build_cfg_entry(serialize=True)
    except Exception as e:
        print(f"Error: {e}")
        return False

    assert Path(CFG_YAML_PATH).exists()

    cfg_entry_from_yaml = load_env_cfg_from_yaml(CFG_YAML_PATH)
    # test env can be created from the yaml file
    name = "kitchen_pick_and_place"
    entry_point = "isaaclab.envs:ManagerBasedRLEnv"
    try:
        gym.register(
            id=name,
            entry_point=entry_point,
            kwargs={"env_cfg_entry_point": cfg_entry_from_yaml},
            disable_env_checker=True,
        )

        cfg = parse_env_cfg(
            name,
            device="cuda:0",
            num_envs=1,
            use_fabric=False,
        )

        # Create environment
        print("[INFO] Creating environment...")
        env = gym.make(name, cfg=cfg).unwrapped
        env.reset()
    except Exception as e:
        print(f"Error: {e}")
        return False

    try:

        # Run some zero actions.
        for _ in tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        metrics = compute_metrics(env)
        assert metrics is not None
        assert "num_episodes" in metrics
        assert "success_rate" in metrics
        assert "object_moved_rate" in metrics

    finally:
        env.close()

    return True


def test_config_serializations():
    result = run_simulation_app_function(
        _test_config_serializations,
        headless=HEADLESS,
    )
    assert result, f"Test {test_config_serializations.__name__} failed"


if __name__ == "__main__":
    test_config_serializations()
