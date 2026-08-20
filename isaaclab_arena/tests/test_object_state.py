# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled tests for Arena rigid and deformable runtime state."""

import torch
import types


class _Scene:
    def __init__(self, entities: dict[str, object], num_envs: int):
        self._entities = entities
        self.env_origins = torch.zeros(num_envs, 3)

    def __getitem__(self, name: str):
        return self._entities[name]

    def keys(self):
        return self._entities.keys()


class _Env:
    def __init__(self, entities, object_types, object_bounds=None):
        from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

        first = next(iter(entities.values()))
        root_pos_w = getattr(first.data, "root_pos_w")
        self.num_envs = root_pos_w.shape[0]
        self.device = root_pos_w.device
        self.scene = _Scene(entities, self.num_envs)
        self.arena_object_types = object_types
        self.arena_object_bounds = object_bounds or {}
        self.object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(self.num_envs, self.device)
        self.unwrapped = self


def test_simulated_deformable_cube_supported_and_settled():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_simulated_deformable_cube_supported_and_settled,
        headless=True,
    )


def test_object_state_dispatches_from_arena_object_type():
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.object_state import object_state

    rigid_pos = torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.3]])
    rigid = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=rigid_pos,
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, 4),
            root_lin_vel_w=torch.ones(2, 3),
            root_ang_vel_w=torch.full((2, 3), 2.0),
        )
    )
    nodal_pos = torch.tensor([
        [[0.0, 0.0, 0.1], [0.2, 0.0, 0.3]],
        [[1.0, 0.0, 0.2], [1.2, 0.0, 0.4]],
    ])
    nodal_vel = torch.tensor([
        [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
    ])
    soft = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=nodal_pos.mean(dim=1),
            root_vel_w=nodal_vel.mean(dim=1),
            nodal_pos_w=nodal_pos,
            nodal_vel_w=nodal_vel,
        )
    )
    env = _Env({"rigid": rigid, "soft": soft}, {"rigid": ObjectType.RIGID, "soft": ObjectType.DEFORMABLE})

    rigid_state = object_state(env, "rigid")
    soft_state = object_state(env, "soft")
    assert rigid_state.points is None
    assert torch.equal(rigid_state.aggregate.angular_velocity_w, rigid.data.root_ang_vel_w)
    assert soft_state.aggregate.orientation_w is None
    assert soft_state.points is not None
    assert torch.equal(soft_state.points.position_w, nodal_pos)
    assert torch.equal(soft_state.points.linear_velocity_w, nodal_vel)


def test_deformable_supported_and_settled_use_nodal_state():
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.spatial import object_supported_by
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    soft_pos = torch.tensor([
        [[-0.02, 0.00, 0.10], [0.02, 0.00, 0.10], [0.00, 0.00, 0.20]],
        [[0.48, 0.00, 0.10], [0.52, 0.00, 0.10], [0.50, 0.00, 0.20]],
    ])
    # Both centroids are stationary; env 1 still has fast internal nodal motion.
    soft_vel = torch.tensor([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    soft = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=soft_pos.mean(dim=1),
            root_vel_w=soft_vel.mean(dim=1),
            nodal_pos_w=soft_pos,
            nodal_vel_w=soft_vel,
        )
    )
    destination_pos = torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, 0.05]])
    destination = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=destination_pos,
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, 4),
            root_lin_vel_w=torch.zeros(2, 3),
            root_ang_vel_w=torch.zeros(2, 3),
        )
    )
    env = _Env(
        {"soft": soft, "destination": destination},
        {"soft": ObjectType.DEFORMABLE, "destination": ObjectType.RIGID},
        {
            "destination": AxisAlignedBoundingBox(
                min_point=(-0.1, -0.1, -0.05),
                max_point=(0.1, 0.1, 0.05),
            )
        },
    )

    supported = object_supported_by(
        env,
        SceneEntityCfg("soft"),
        SceneEntityCfg("destination"),
        support_tolerance=1.0e-4,
    )
    settled = objects_settled(env, ["soft"], lin_vel_threshold=0.1)
    assert supported.tolist() == [True, False]
    assert settled.tolist() == [True, False]


def test_objects_settled_preserves_rigid_angular_speed_check():
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled

    position = torch.zeros(2, 3)
    rigid = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=position,
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, 4),
            root_lin_vel_w=torch.zeros(2, 3),
            root_ang_vel_w=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        )
    )
    env = _Env({"cube": rigid}, {"cube": ObjectType.RIGID})
    assert objects_settled(env, ["cube"]).tolist() == [True, False]


def _test_simulated_deformable_cube_supported_and_settled(simulation_app):
    import isaaclab.sim as sim_utils
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.deformable_spawn import VolumeDeformableMaterial
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.spatial import object_supported_by
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    destination = Object(
        name="destination",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(0.4, 0.4, 0.1)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05)),
    )
    soft_cube = DeformableObject(
        name="soft_cube",
        spawner_cfg=sim_utils.MeshCuboidCfg(size=(0.1, 0.1, 0.1)),
        material=VolumeDeformableMaterial(
            youngs_modulus=8.0e4,
            poissons_ratio=0.4,
            density=300.0,
            particle_radius=0.01,
        ),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.15)),
    )
    arena_env = IsaacLabArenaEnvironment(
        name="object_state_deformable_cube",
        scene=Scene(assets=[destination, soft_cube]),
    )
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_kwargs["arena_object_bounds"]["destination"] = AxisAlignedBoundingBox(
        min_point=(-0.2, -0.2, -0.05),
        max_point=(0.2, 0.2, 0.05),
    )
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()

    try:
        nodal_state = env.unwrapped.scene["soft_cube"].data.nodal_state_w.torch.clone()
        nodal_state[..., 3:] = 0.0
        env.unwrapped.scene["soft_cube"].write_nodal_state_to_sim_index(nodal_state)
        assert object_supported_by(
            env,
            SceneEntityCfg("soft_cube"),
            SceneEntityCfg("destination"),
        ).item()
        assert objects_settled(env, ["soft_cube"]).item()
    finally:
        env.close()
    return True
