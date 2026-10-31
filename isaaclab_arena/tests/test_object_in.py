# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify outside-to-inside containment through a simulated apple drop."""

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _ContactAssetDouble:
    """Asset double that exposes contact-filter paths by object name."""

    def __init__(self, name, object_type):
        self.name = name
        self.object_type = object_type

    def get_contact_sensor_cfg(self, contact_against_objects=()):
        return SimpleNamespace(
            prim_path=f"/World/{self.name}",
            filter_prim_paths_expr=[f"/World/{obj.name}" for obj in contact_against_objects],
        )


def test_parallel_object_in_tasks_count_objects_in_the_same_target_as_support():
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.object_in_task import ObjectInTask

    drawer = _ContactAssetDouble("drawer", ObjectType.BASE)
    other_drawer = _ContactAssetDouble("other_drawer", ObjectType.BASE)
    scissors = _ContactAssetDouble("scissors", ObjectType.RIGID)
    marker = _ContactAssetDouble("marker", ObjectType.RIGID)
    unrelated_tool = _ContactAssetDouble("unrelated_tool", ObjectType.RIGID)
    scissors_task = ObjectInTask(scissors, drawer)
    marker_task = ObjectInTask(marker, drawer)
    unrelated_task = ObjectInTask(unrelated_tool, other_drawer)

    CompositeTaskBase([scissors_task, marker_task, unrelated_task])

    scene_cfg = scissors_task.get_scene_cfg()
    sensor_cfg = getattr(scene_cfg, scissors_task.contact_sensor_cfg.name)
    assert sensor_cfg.filter_prim_paths_expr == ["/World/drawer", "/World/marker"]


def _test_stacked_objects_in_shared_target(_simulation_app):
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.object_in_task import ObjectInTask
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    background = registry.get_asset_by_name("maple_table_robolab")()
    bowl = registry.get_asset_by_name("bowl_ycb_robolab")(initial_pose=Pose(position_xyz=(0.55, 0.0, 0.03)))
    lower_cube = registry.get_asset_by_name("dex_cube")(
        name="lower_cube",
        initial_pose=Pose(position_xyz=(0.55, 0.0, 0.035)),
    )
    upper_cube = registry.get_asset_by_name("dex_cube")(
        name="upper_cube",
        initial_pose=Pose(position_xyz=(0.55, 0.0, 0.085)),
    )
    tasks = [
        ObjectInTask(
            lower_cube,
            bowl,
            consecutive_success_steps=1,
            linear_velocity_threshold=0.35,
            angular_velocity_threshold=12.0,
            minimum_contained_fraction=0.5,
        ),
        ObjectInTask(
            upper_cube,
            bowl,
            consecutive_success_steps=1,
            linear_velocity_threshold=0.35,
            angular_velocity_threshold=12.0,
            minimum_contained_fraction=0.5,
        ),
    ]
    arena_env = IsaacLabArenaEnvironment(
        name="stacked_objects_in_shared_target",
        embodiment=FrankaIKEmbodiment(),
        scene=Scene(assets=[background, bowl, lower_cube, upper_cube]),
        task=CompositeTaskBase(tasks),
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
    try:
        env.reset()
        with torch.inference_mode():
            succeeded = False
            for _ in range(50):
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
                if env.unwrapped.termination_manager.get_term("success").all():
                    succeeded = True
                    break

            lower_cube_z = env.unwrapped.arena_world.get_position_w(lower_cube.name)[:, 2]
            upper_cube_z = env.unwrapped.arena_world.get_position_w(upper_cube.name)[:, 2]
            assert torch.all(upper_cube_z > lower_cube_z + 0.04), "The cubes no longer form a stack."
            assert succeeded, "The stacked ObjectInTasks did not report success."
        return True
    finally:
        env.close()


def test_stacked_objects_in_shared_target_count_as_success():
    assert run_function_with_persistent_simulation_app(_test_stacked_objects_in_shared_target)


def get_pose_at_midpoint(env, object_name: str, target_name: str):
    """Return an identity-orientation pose centering the object's geometry in the target bounds.

    Args:
        env: Wrapped Arena environment.
        object_name: Scene key for the object to position.
        target_name: Scene key for the destination.

    Returns:
        Batched world-frame poses with XYZW quaternions.
    """
    world = env.unwrapped.arena_world
    target_bounds = world.get_aabb_w(target_name)
    object_bounds = world.get_aabb_in_local_frame(object_name)
    T_W_O = world.get_pose_w(object_name).clone()
    T_W_O[:, 3:] = T_W_O.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    T_W_O[:, :3] = target_bounds.center - object_bounds.center
    return T_W_O


def test_mesh_point_fraction_uses_target_local_frame():
    import torch

    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.tasks.predicates.spatial import mesh_point_fraction_in_target
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    half_sqrt_two = 2**-0.5
    T_W_T = torch.tensor([
        [2.0, -1.0, 0.5, 0.0, 0.0, half_sqrt_two, half_sqrt_two],
        [-3.0, 4.0, 1.0, 0.0, 0.0, -half_sqrt_two, half_sqrt_two],
    ])
    object_points_T = torch.tensor([
        [-0.9, -0.1, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 0.05],
        [1.2, 0.0, 0.0],
    ]).expand(2, -1, -1)
    q_W_T = T_W_T[:, None, 3:].expand(-1, object_points_T.shape[1], -1)
    object_points_W = quat_apply(q_W_T, object_points_T) + T_W_T[:, None, :3]
    valid_points = torch.tensor([[True, True, True, False], [True, True, True, True]])
    target_bounds_T = AxisAlignedBoundingBox((-1.0, -0.2, -0.1), (1.0, 0.2, 0.1))

    contained_fraction = mesh_point_fraction_in_target(
        object_points_W,
        valid_points,
        T_W_T,
        target_bounds_T,
    )

    torch.testing.assert_close(contained_fraction, torch.tensor([1.0, 0.75]))


def _test_apple_in_microwave(_simulation_app):
    import torch

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.object_in_task import ObjectInTask
    from isaaclab_arena.tasks.predicates.spatial import object_in_contact_with_target, object_in_target
    from isaaclab_arena.tests.utils.simulation import step_zeros_and_call
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    # Keep the door at its authored closed pose. The background root provides a
    # fixed geometry frame while Scene manages the microwave's nested physics.
    microwave = Background(
        name="microwave",
        usd_path=registry.get_asset_by_name("microwave").usd_path,
        object_min_z=-1.0,
        initial_pose=Pose(position_xyz=(2.0, 0.0, 1.0)),
    )
    apple = registry.get_asset_by_name("apple_02_objaverse_robolab")(initial_pose=Pose(position_xyz=(3.0, 0.0, 1.0)))
    arena_env = IsaacLabArenaEnvironment(
        name="apple_in_microwave_test",
        scene=Scene(assets=[microwave, apple]),
        embodiment=FrankaJointPosEmbodiment(),
        # Sim 6.2 leaves this round apple with a small residual spin after the drop.
        task=ObjectInTask(apple, microwave, angular_velocity_threshold=0.2, episode_length_s=30.0),
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
    try:
        obs, _ = env.reset()
        # Gym wrappers do not forward Arena/Isaac Lab attributes; keep env.step wrapped.
        base = env.unwrapped
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        with torch.inference_mode():

            def check_outside(env, terminated):
                assert not object_in_target(env.unwrapped, apple.name, microwave.name).any()
                assert not env.unwrapped.termination_manager.get_term("success").any()
                assert not terminated.any()

            step_zeros_and_call(env, num_steps=10, function=check_outside)
            T_W_A = get_pose_at_midpoint(env, apple.name, microwave.name)
            body = base.scene[apple.name]
            body.write_root_pose_to_sim(T_W_A)
            body.write_root_velocity_to_sim(torch.zeros((base.num_envs, 6), device=base.device))
            assert object_in_target(base, apple.name, microwave.name).all()
            initial_height = T_W_A[:, 2].clone()
            fell = False
            made_contact = False
            for step in range(500):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                assert not truncated.any(), "Apple drop timed out"
                success = base.termination_manager.get_term("success")
                if terminated.any():
                    assert success.all(), "Apple episode terminated without task success"
                    assert made_contact, "Success requires contact with the microwave"
                    assert fell, "Apple should fall under gravity before success"
                    return True
                contact = object_in_contact_with_target(
                    base, arena_env.task.contact_sensor_cfg, arena_env.task.contact_force_threshold
                )
                if step == 0:
                    assert not contact.any(), "The apple should initially be airborne inside the microwave"
                    assert not success.any(), "Containment without contact must not count as success"
                made_contact |= bool(contact.all())
                fell |= bool((base.arena_world.get_pose_w(apple.name)[:, 2] < initial_height - 0.01).all())
            assert False, "Apple did not settle inside the microwave and trigger success"
    finally:
        env.close()


def test_apple_in_microwave():
    assert run_function_with_persistent_simulation_app(_test_apple_in_microwave)
