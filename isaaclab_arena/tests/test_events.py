# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 10
HEADLESS = True
INITIAL_POSITION_EPS = 0.1  # The cracker box falls slightly.

# For object initial velocity test: min displacement in velocity direction
OBJECT_VELOCITY_MIN_DISPLACEMENT = 0.05  # minimum movement in meters


def _test_set_object_pose_per_env_event(simulation_app):
    """Returns a scene which we use for these tests."""

    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.terms.events import set_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()

    background = asset_registry.get_asset_by_name("kitchen_with_open_drawer")()
    embodiment = asset_registry.get_asset_by_name("franka")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()
    destination_location = ObjectReference(
        name="destination_location",
        prim_path="{ENV_REGEX_NS}/kitchen_with_open_drawer/Cabinet_B_02",
        parent_asset=background,
    )

    scene = Scene(assets=[background, cracker_box])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="robot_initial_position",
        embodiment=embodiment,
        scene=scene,
        task=PickAndPlaceTask(cracker_box, destination_location, background),
        teleop_device=None,
    )

    # Build the cfg, but dont register so we can make some adjustments.
    NUM_ENVS = 2
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = NUM_ENVS
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env_cfg = env_builder.compose_manager_cfg()

    # Replace the pose reset term:
    # - from: constant per env,
    # - to: per env pose
    pose_list = [
        Pose(position_xyz=(0.4, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        Pose(position_xyz=(0.4, 0.4, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    ]
    env_cfg.events.reset_pick_up_object_pose = EventTermCfg(
        func=set_object_pose_per_env,
        mode="reset",
        params={
            "pose_list": pose_list,
            "asset_cfg": SceneEntityCfg(cracker_box.name),
        },
    )

    env = env_builder.make_registered(env_cfg)
    env.reset()

    try:

        # Run some zero actions.
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        # Check that the cracker box ended up in the correct position.
        cracker_box_poses = cracker_box.get_object_pose(env)
        initial_poses = torch.cat(
            (
                pose_list[0].to_tensor(device=env.unwrapped.device).unsqueeze(0),
                pose_list[1].to_tensor(device=env.unwrapped.device).unsqueeze(0),
            ),
            dim=0,
        )
        position_errors = torch.norm(cracker_box_poses[:, :3] - initial_poses[:, :3], dim=1)
        print(f"Cranker box poses: {cracker_box_poses}")
        print(f"Initial poses: {initial_poses}")
        print(f"Position errors: {position_errors}")
        assert torch.all(position_errors < INITIAL_POSITION_EPS), "Position errors are too large"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def _test_object_moves_with_initial_velocity(simulation_app):
    """Test that a sphere moves with the given initial velocity after reset."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    asset_registry = AssetRegistry()
    no_gravity_cfg = sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
    )
    sphere = asset_registry.get_asset_by_name("sphere")(spawner_cfg=no_gravity_cfg)

    initial_velocity = Velocity(linear_xyz=(-0.5, 0.0, 0.0))  # There is a wall in +x
    sphere.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.5), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
    sphere.set_initial_velocity(initial_velocity)

    scene = Scene(assets=[sphere])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="object_initial_velocity_test",
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = 1
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = env_builder.make_registered()
    env.reset()

    try:
        initial_position = env.unwrapped.scene[sphere.name].data.root_pose_w[0, :3].clone()
        initial_position[:3] -= env.unwrapped.scene.env_origins[0]

        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)

        final_position = env.unwrapped.scene[sphere.name].data.root_pose_w[0, :3].clone()
        final_position[:3] -= env.unwrapped.scene.env_origins[0]
        displacement = final_position - initial_position
        assert (
            displacement[0].item() < -OBJECT_VELOCITY_MIN_DISPLACEMENT  # - x because initial velocity is in -x
        ), f"Object did not move with given velocity: displacement in x was {displacement[0].item()}"

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        env.close()

    return True


def test_set_object_post_per_env_event():
    result = run_simulation_app_function(
        _test_set_object_pose_per_env_event,
        headless=HEADLESS,
    )
    assert result, f"Test {test_set_object_post_per_env_event.__name__} failed"


def test_object_moves_with_initial_velocity():
    result = run_simulation_app_function(
        _test_object_moves_with_initial_velocity,
        headless=HEADLESS,
    )
    assert result, f"Test {test_object_moves_with_initial_velocity.__name__} failed"


if __name__ == "__main__":
    test_set_object_post_per_env_event()
    test_object_moves_with_initial_velocity()
