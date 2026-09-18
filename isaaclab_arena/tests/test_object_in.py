# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify outside-to-inside containment through a simulated apple drop."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


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
    from isaaclab_arena.tasks.predicates.spatial import object_in_contact_with_target, object_in_target_aabb
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
        task=ObjectInTask(apple, microwave, episode_length_s=30.0),
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
    try:
        obs, _ = env.reset()
        # Gym wrappers do not forward Arena/Isaac Lab attributes; keep env.step wrapped.
        base = env.unwrapped
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        with torch.inference_mode():

            def check_outside(env, terminated):
                assert not object_in_target_aabb(env.unwrapped, apple.name, microwave.name).any()
                assert not env.unwrapped.termination_manager.get_term("success").any()
                assert not terminated.any()

            step_zeros_and_call(env, num_steps=10, function=check_outside)
            T_W_A = get_pose_at_midpoint(env, apple.name, microwave.name)
            body = base.scene[apple.name]
            body.write_root_pose_to_sim(T_W_A)
            body.write_root_velocity_to_sim(torch.zeros((base.num_envs, 6), device=base.device))
            assert object_in_target_aabb(base, apple.name, microwave.name).all()
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
