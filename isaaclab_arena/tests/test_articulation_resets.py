# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test that robots and articulated scene objects each reset only themselves."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_fake_articulation(num_envs: int, num_joints: int):
    """Return an articulation stand-in that applies every write to its own state tensors."""
    import torch
    from types import SimpleNamespace

    import warp as wp

    default_root_state = torch.arange(13 * num_envs, dtype=torch.float32).reshape(num_envs, 13)
    default_joint_pos = torch.arange(num_joints * num_envs, dtype=torch.float32).reshape(num_envs, num_joints) + 1.0
    articulation = SimpleNamespace(
        data=SimpleNamespace(
            default_root_state=wp.from_torch(default_root_state),
            default_joint_pos=SimpleNamespace(torch=default_joint_pos),
            default_joint_vel=SimpleNamespace(torch=torch.zeros(num_envs, num_joints)),
        ),
        root_pose=torch.full((num_envs, 7), -1.0),
        root_velocity=torch.full((num_envs, 6), -1.0),
        joint_pos=torch.full((num_envs, num_joints), -1.0),
        joint_vel=torch.full((num_envs, num_joints), -1.0),
    )

    def write(state):
        def apply(values, env_ids):
            getattr(articulation, state)[env_ids] = values

        return apply

    articulation.write_root_pose_to_sim = write("root_pose")
    articulation.write_root_velocity_to_sim = write("root_velocity")
    articulation.write_root_pose_to_sim_index = lambda root_pose, env_ids: write("root_pose")(root_pose, env_ids)
    articulation.write_root_velocity_to_sim_index = lambda root_velocity, env_ids: write("root_velocity")(
        root_velocity, env_ids
    )
    articulation.write_joint_position_to_sim_index = lambda position, env_ids: write("joint_pos")(position, env_ids)
    articulation.write_joint_velocity_to_sim_index = lambda velocity, env_ids: write("joint_vel")(velocity, env_ids)
    return articulation


def _make_fake_env(**articulations):
    import torch
    from types import SimpleNamespace

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])

    return SimpleNamespace(scene=Scene(articulations), device="cpu")


def _test_robot_reset_restores_only_its_articulation(simulation_app):
    import torch

    import warp as wp
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.terms.events import reset_articulation_to_default

    robot = _make_fake_articulation(num_envs=2, num_joints=3)
    drawer = _make_fake_articulation(num_envs=2, num_joints=1)
    env = _make_fake_env(robot=robot, drawer=drawer)

    reset_articulation_to_default(env, torch.tensor([1]), SceneEntityCfg("robot"))

    default_root_state = wp.to_torch(robot.data.default_root_state)[1]
    expected_root_pose = default_root_state[:7].clone()
    expected_root_pose[:3] += torch.tensor([10.0, 20.0, 30.0])
    torch.testing.assert_close(robot.root_pose[1], expected_root_pose)
    torch.testing.assert_close(robot.root_velocity[1], default_root_state[7:])
    torch.testing.assert_close(robot.joint_pos[1], robot.data.default_joint_pos.torch[1])
    torch.testing.assert_close(robot.joint_vel[1], torch.zeros(3))
    # Other environments and other articulations keep their state.
    assert torch.all(robot.root_pose[0] == -1.0) and torch.all(robot.joint_pos[0] == -1.0)
    for state in (drawer.root_pose, drawer.root_velocity, drawer.joint_pos, drawer.joint_vel):
        assert torch.all(state == -1.0)
    return True


def test_robot_reset_restores_only_its_articulation():
    assert run_function_with_persistent_simulation_app(_test_robot_reset_restores_only_its_articulation)


def _test_articulated_objects_reset_their_joints(simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.placement_events import make_cached_placement_event
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange

    def reset_drawer(drawer: Object):
        """Run the drawer's own reset event for environment 1 and return the written state."""
        articulation = _make_fake_articulation(num_envs=2, num_joints=2)
        other = _make_fake_articulation(num_envs=2, num_joints=3)
        name, event = drawer.get_event_cfg()
        event.func(_make_fake_env(**{name: articulation, "other": other}), torch.tensor([1]), **event.params)
        for state in (
            articulation.root_pose,
            articulation.root_velocity,
            articulation.joint_pos,
            articulation.joint_vel,
        ):
            assert torch.all(state[0] == -1.0)
        for state in (other.root_pose, other.root_velocity, other.joint_pos, other.joint_vel):
            assert torch.all(state == -1.0)
        torch.testing.assert_close(articulation.joint_pos[1], articulation.data.default_joint_pos.torch[1])
        torch.testing.assert_close(articulation.joint_vel[1], articulation.data.default_joint_vel.torch[1])
        return articulation

    def make_drawer():
        return Object(name="drawer", usd_path="/unused/drawer.usd", object_type=ObjectType.ARTICULATION)

    # Without a configured pose, the drawer restores its default root and joints.
    articulation = reset_drawer(make_drawer())
    default_root_state = wp.to_torch(articulation.data.default_root_state)[1]
    expected_root_pose = default_root_state[:7].clone()
    expected_root_pose[:3] += torch.tensor([10.0, 20.0, 30.0])
    torch.testing.assert_close(articulation.root_pose[1], expected_root_pose)
    torch.testing.assert_close(articulation.root_velocity[1], default_root_state[7:])

    # Explicitly disabling pose resets keeps the displaced root.
    drawer = make_drawer()
    drawer.disable_reset_pose()
    articulation = reset_drawer(drawer)
    assert torch.all(articulation.root_pose == -1.0)
    assert torch.all(articulation.root_velocity == -1.0)

    # A construction-only pose leaves root resets to dynamic relation placement.
    drawer = make_drawer()
    drawer.set_initial_pose(Pose.identity(), create_reset_event=False)
    assert not drawer.has_pose_reset_event()
    articulation = reset_drawer(drawer)
    assert torch.all(articulation.root_pose == -1.0)
    assert torch.all(articulation.root_velocity == -1.0)

    # With a pose, the drawer restores its root pose and its joints.
    drawer = make_drawer()
    drawer.set_initial_pose(Pose(position_xyz=(1.0, 2.0, 3.0)))
    articulation = reset_drawer(drawer)
    torch.testing.assert_close(articulation.root_pose[1], torch.tensor([11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 1.0]))

    # With a per-environment pose, as static relation placement stores, the drawer restores that
    # environment's root pose and its joints.
    drawer = make_drawer()
    drawer.set_initial_pose(PosePerEnv(poses=[Pose.identity(), Pose(position_xyz=(1.0, 2.0, 3.0))]))
    articulation = reset_drawer(drawer)
    torch.testing.assert_close(articulation.root_pose[1], torch.tensor([11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 1.0]))

    # With a pose range, the drawer samples its root pose and restores its joints. The range holds
    # one position, so the sampled position is known.
    drawer = make_drawer()
    drawer.set_initial_pose(PoseRange(position_xyz_min=(1.0, 2.0, 3.0), position_xyz_max=(1.0, 2.0, 3.0)))
    articulation = reset_drawer(drawer)
    torch.testing.assert_close(articulation.root_pose[1, :3], torch.tensor([11.0, 22.0, 33.0]))

    # When placement owns the root pose, the drawer keeps only its joint reset.
    drawer = make_drawer()
    drawer.set_initial_pose(Pose.identity())
    make_cached_placement_event(PlacementLayouts({"drawer": [Pose((1.0, 0.0, 0.0))]}), [drawer], num_envs=2)
    assert not drawer.has_pose_reset_event()
    articulation = reset_drawer(drawer)
    assert torch.all(articulation.root_pose == -1.0)
    return True


def test_articulated_objects_reset_their_joints():
    assert run_function_with_persistent_simulation_app(_test_articulated_objects_reset_their_joints)
