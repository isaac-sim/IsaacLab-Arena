# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration and backend smoke tests for deformables."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.physics_backend import PhysicsBackend


def _make_soft_cube(physics_backend: PhysicsBackend, initial_pose=None):
    import isaaclab.sim as sim_utils
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    from isaaclab_arena.assets.deformable_object import DeformableObject

    if physics_backend is PhysicsBackend.PHYSX:
        deformable_props = PhysxDeformableBodyPropertiesCfg()
        collision_props = [PhysxCollisionCfg(rest_offset=0.0025, contact_offset=0.01)]
        physics_material = PhysxDeformableBodyMaterialCfg()
    else:
        deformable_props = NewtonDeformableBodyPropertiesCfg()
        collision_props = None
        physics_material = NewtonDeformableBodyMaterialCfg()
    return DeformableObject(
        name="soft_cube",
        spawner_cfg=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.1),
            deformable_props=deformable_props,
            collision_props=collision_props,
            physics_material=physics_material,
        ),
        initial_pose=initial_pose,
    )


def _test_backend_specific_deformable_config(simulation_app) -> bool:
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.utils.pose import Pose, PoseRange

    pose = Pose(position_xyz=(0.1, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    soft_cube = _make_soft_cube(PhysicsBackend.PHYSX, initial_pose=pose)
    _, physx_cfg = soft_cube.get_object_cfg()

    assert isinstance(physx_cfg.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(physx_cfg.spawn.physics_material, PhysxDeformableBodyMaterialCfg)
    assert physx_cfg.init_state.pos == pose.position_xyz
    assert physx_cfg.init_state.rot == pose.rotation_xyzw

    with pytest.raises(AssertionError, match="backend-specific deformable_props"):
        DeformableObject(
            name="authored_soft_body",
            spawner_cfg=UsdFileCfg(usd_path="/tmp/authored_soft_body.usd"),
        )

    updated_pose = Pose(position_xyz=(0.4, 0.0, 0.5))
    soft_cube.set_initial_pose(updated_pose)
    assert physx_cfg.init_state.pos == updated_pose.position_xyz
    with pytest.raises(AssertionError, match="fixed Pose or PosePerEnv"):
        soft_cube.set_initial_pose(PoseRange())
    return True


def _test_deformable_validates_simulation_cfg(simulation_app) -> bool:
    from isaaclab.sim import SimulationCfg
    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, VBDSolverCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.scene.scene import Scene

    physx_scene = Scene(assets=[_make_soft_cube(PhysicsBackend.PHYSX)])
    physx_scene.validate_simulation_cfg(SimulationCfg(physics=PhysxCfg()))

    newton_scene = Scene(assets=[_make_soft_cube(PhysicsBackend.NEWTON)])
    newton_scene.validate_simulation_cfg(SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg())))
    coupled_solver_cfg = CouplerProxyCfg(
        entries=[CouplerEntryCfg(name="soft", solver_cfg=VBDSolverCfg(), all_particles=True)]
    )
    newton_scene.validate_simulation_cfg(SimulationCfg(physics=NewtonCfg(solver_cfg=coupled_solver_cfg)))

    with pytest.raises(AssertionError, match="requires a Newton VBD solver.*MJWarpSolverCfg"):
        newton_scene.validate_simulation_cfg(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

    with pytest.raises(AssertionError, match="requires a Newton VBD solver.*CouplerProxyCfg"):
        unassigned_vbd_cfg = CouplerProxyCfg(entries=[CouplerEntryCfg(name="soft", solver_cfg=VBDSolverCfg())])
        newton_scene.validate_simulation_cfg(SimulationCfg(physics=NewtonCfg(solver_cfg=unassigned_vbd_cfg)))

    for scene, wrong_backend_cfg in (
        (physx_scene, SimulationCfg(physics=NewtonCfg(solver_cfg=VBDSolverCfg()))),
        (newton_scene, SimulationCfg(physics=PhysxCfg())),
    ):
        with pytest.raises(AssertionError, match="configured for.*not the selected backend"):
            scene.validate_simulation_cfg(wrong_backend_cfg)
    return True


def _test_deformable_nodal_reset_terms(simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils import math as math_utils

    from isaaclab_arena.terms.events import set_deformable_object_pose, set_deformable_object_pose_per_env
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.velocity import Velocity

    class FakeAsset:
        def __init__(self, usd_backed: bool = False):
            default_state = torch.zeros((2, 2, 6))
            default_state[0, :, :3] = torch.tensor([[-0.05, 0.0, 0.5], [0.05, 0.0, 0.5]])
            default_state[1, :, :3] = torch.tensor([[9.95, 0.0, 0.5], [10.05, 0.0, 0.5]])
            self.data = SimpleNamespace(default_nodal_state_w=SimpleNamespace(torch=default_state))
            spawn = UsdFileCfg(usd_path="/tmp/fake.usd") if usd_backed else SimpleNamespace()
            self.cfg = SimpleNamespace(
                init_state=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
                spawn=spawn,
            )
            self.written_state = default_state.clone()
            self.reset_env_ids = None

        def write_nodal_state_to_sim_index(self, nodal_state, env_ids):
            self.written_state[env_ids] = nodal_state

        def transform_nodal_pos(self, nodal_pos, pos, quat):
            mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
            return math_utils.transform_points(nodal_pos - mean_nodal_pos, pos, quat) + mean_nodal_pos

        def reset(self, env_ids):
            self.reset_env_ids = env_ids.clone()

    class FakeScene(dict):
        def __init__(self, asset):
            super().__init__(soft_cube=asset)
            self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    asset = FakeAsset()
    env = SimpleNamespace(scene=FakeScene(asset), device=torch.device("cpu"))
    asset_cfg = SimpleNamespace(name="soft_cube")
    env_ids = torch.tensor([0, 1])

    fixed_pose = Pose(position_xyz=(1.0, 2.0, 3.0))
    set_deformable_object_pose(
        env,
        env_ids,
        asset_cfg,
        fixed_pose,
        Velocity(linear_xyz=(0.1, 0.2, 0.3)),
    )
    torch.testing.assert_close(
        asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[1.0, 2.0, 3.0], [11.0, 2.0, 3.0]]),
    )
    torch.testing.assert_close(asset.written_state[..., 3:], torch.tensor([0.1, 0.2, 0.3]).expand(2, 2, 3))

    poses = [Pose(position_xyz=(0.0, 1.0, 2.0)), Pose(position_xyz=(3.0, 4.0, 5.0))]
    set_deformable_object_pose_per_env(env, env_ids, asset_cfg, poses)
    torch.testing.assert_close(
        asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[0.0, 1.0, 2.0], [13.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(asset.written_state[..., 3:], torch.zeros((2, 2, 3)))
    torch.testing.assert_close(asset.reset_env_ids, env_ids)

    usd_asset = FakeAsset(usd_backed=True)
    usd_env = SimpleNamespace(scene=FakeScene(usd_asset), device=torch.device("cpu"))
    set_deformable_object_pose(usd_env, env_ids, asset_cfg, fixed_pose)
    torch.testing.assert_close(
        usd_asset.written_state[..., :3].mean(dim=1),
        torch.tensor([[1.0, 2.0, 3.5], [11.0, 2.0, 3.5]]),
    )
    return True


def _test_deformable_reset_and_initial_pose(simulation_app) -> bool:
    import torch
    from unittest.mock import Mock

    from isaaclab.assets import DeformableObjectCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.offline_placement.pool_validation import validate_pool_layouts
    from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
    from isaaclab_arena.relations.placement_result import PlacementResult
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.validation.types import PlacementCheck, PlacementValidationResults
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    poses = PosePerEnv(
        poses=[
            Pose(position_xyz=(0.0, 0.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            Pose(position_xyz=(0.2, 0.0, 0.6), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        ]
    )
    physics_backend = PhysicsBackend.PHYSX
    soft_cube = AssetRegistry().get_asset_by_name("deformable_cube_physx")(name="soft_cube")
    soft_cube.set_initial_pose(poses)
    arena_env = IsaacLabArenaEnvironment(
        name=f"{physics_backend}_deformable_reset_and_initial_pose",
        scene=Scene(assets=[soft_cube]),
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=2, presets=physics_backend, solve_relations=False),
    )
    env_cfg, env_kwargs = builder.build_env_cfg()
    assert isinstance(env_cfg.scene.soft_cube, DeformableObjectCfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    env.reset()

    try:
        deformable_asset = env.unwrapped.scene[soft_cube.name]
        initial_nodal_state = deformable_asset.data.nodal_state_w.torch.clone()
        assert torch.isfinite(initial_nodal_state).all()
        expected = torch.tensor([pose.position_xyz for pose in poses.poses], device=env.unwrapped.device)
        aggregate_positions = deformable_asset.data.root_pos_w.torch - env.unwrapped.scene.env_origins
        torch.testing.assert_close(aggregate_positions, expected, atol=2.0e-3, rtol=0.0)

        displaced = initial_nodal_state.clone()
        displaced[..., 0] += 0.25
        displaced[..., 1] += torch.linspace(
            0.0,
            0.05,
            displaced.shape[1],
            device=displaced.device,
        )
        displaced[..., 3:] = 0.5
        deformable_asset.write_nodal_state_to_sim_index(displaced)
        env.reset()
        restored_nodal_state = deformable_asset.data.nodal_state_w.torch.clone()
        torch.testing.assert_close(restored_nodal_state, initial_nodal_state)

        # Pool validation supports nodal bodies without requiring a rigid root pose.
        pool = Mock(spec=PooledObjectPlacer)
        pool.objects = [soft_cube]
        pool.layouts_per_env.return_value = [
            [PlacementResult(PlacementValidationResults(), {soft_cube: pose.position_xyz}, 0.0, 1)]
            for pose in poses.poses
        ]
        results = validate_pool_layouts(env, pool, PhysicsSettleParams(num_steps=5))
        assert len(results) == 2
        assert all(not checks.validation_results[PlacementCheck.PHYSICS_SETTLED] for _, _, checks in results)
    finally:
        env.close()
    return True


def _test_deformable_placement_is_not_immediate_task_success(simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    soft_cube = AssetRegistry().get_asset_by_name("deformable_cube_physx")(name="soft_cube")
    soft_cube.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.05)))
    table = AssetRegistry().get_asset_by_name("procedural_table")(
        name="destination",
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0)),
    )
    task = PickAndPlaceTask(
        pick_up_object=soft_cube,
        destination_location=table,
        background_scene=table,
    )
    arena_env = IsaacLabArenaEnvironment(
        name="deformable_pick_and_place_success",
        scene=Scene(assets=[soft_cube, table]),
        task=task,
    )
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, presets="physx", solve_relations=False),
    )
    env_cfg, env_kwargs = builder.build_env_cfg()
    env = builder.make_registered(env_cfg, env_kwargs)

    try:
        env.reset()
        success_objective = task.get_termination_cfg().success[0]
        placement_requirement = success_objective.predicate_sequence[-1]
        placement_predicate = placement_requirement.predicate
        placed = placement_predicate(env.unwrapped)
        torch.testing.assert_close(placed, torch.ones(1, dtype=torch.bool, device=env.unwrapped.device))

        # Starting on the destination must not skip the required lift.
        env.unwrapped.termination_manager.compute()
        success = env.unwrapped.termination_manager.get_term("success")
        torch.testing.assert_close(success, torch.zeros(1, dtype=torch.bool, device=env.unwrapped.device))
    finally:
        env.close()
    return True


def _test_droid_deformable_environment(simulation_app, physics_backend: str) -> bool:
    """Spawn the requested backend and complete pick-and-place after teleporting the object."""
    import torch

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    args = parser.parse_args([
        "--num_envs",
        "4",
        "--presets",
        physics_backend,
        "droid_deformable_pick_and_place",
        "--pick_object",
        f"deformable_cube_{physics_backend}",
        "--embodiment",
        "droid_differential_ik",
        "--teleop_device",
        "keyboard",
    ])
    builder = get_arena_builder_from_cli(args)
    assert builder.arena_env.task.get_task_description() == "Pick up the deformable cube and place it on the plate."
    env_cfg, env_kwargs = builder.build_env_cfg()
    assert env_cfg.teleop_devices is not None
    assert "keyboard" in env_cfg.teleop_devices.devices
    if physics_backend == "newton":
        assert isinstance(env_cfg.sim.physics, NewtonCfg)
        expected_deformable_props_type = NewtonDeformableBodyPropertiesCfg
    else:
        assert isinstance(env_cfg.sim.physics, PhysxCfg)
        expected_deformable_props_type = PhysxDeformableBodyPropertiesCfg

    env = builder.make_registered(env_cfg, env_kwargs)
    try:
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for step in range(100):
            _, _, terminated, truncated, _ = env.step(actions)
            assert not terminated.any() and not truncated.any(), f"Environment reset after zero-action step {step + 1}."
            _, settled = get_object_initial_rest_state(env.unwrapped, "pick_object")
            if settled.all():
                break
        assert settled.all(), "Deformable objects did not settle under zero actions."

        deformable = env.unwrapped.scene.deformable_objects["pick_object"]
        assert isinstance(deformable.cfg.spawn.deformable_props, expected_deformable_props_type)
        lifted_state = deformable.data.nodal_state_w.torch.clone()
        lifted_state[..., 2] += 0.1
        lifted_state[..., 3:] = 0.0
        deformable.write_nodal_state_to_sim_index(lifted_state)
        _, _, terminated, truncated, _ = env.step(actions)
        assert not terminated.any() and not truncated.any(), "Lifting should advance progress without ending the task."

        destination_bounds = env.unwrapped.arena_world.get_aabb_w("plate")
        placed_state = deformable.data.nodal_state_w.torch.clone()
        node_positions = placed_state[..., :3]
        destination_center = destination_bounds.center
        translation = torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
        translation[:, :2] = destination_center[:, :2] - node_positions.mean(dim=1)[:, :2]
        translation[:, 2] = destination_bounds.max_point[:, 2] - node_positions.amin(dim=1)[:, 2]
        placed_state[..., :3] += translation[:, None, :]
        placed_state[..., 3:] = 0.0
        deformable.write_nodal_state_to_sim_index(placed_state)

        completed = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for _ in range(200):
            _, _, terminated, truncated, extras = env.step(actions)
            assert not truncated.any(), "The task timed out before the placement could register."
            progress_states = extras["progress_tracking"]["states"]
            for env_id in terminated.nonzero(as_tuple=False).flatten().tolist():
                assert progress_states[env_id].all_complete
            completed |= terminated
            if completed.all():
                break
        assert completed.all(), (
            "Teleporting the lifted deformable objects onto the plate did not terminate every environment: "
            f"{completed.tolist()}"
        )
    finally:
        env.close()
    return True


def test_backend_specific_deformable_config():
    assert run_function_with_persistent_simulation_app(_test_backend_specific_deformable_config)


def test_deformable_validates_simulation_cfg():
    assert run_function_with_persistent_simulation_app(_test_deformable_validates_simulation_cfg)


def test_deformable_nodal_reset_terms():
    assert run_function_with_persistent_simulation_app(_test_deformable_nodal_reset_terms)


def test_deformable_reset_and_initial_pose():
    assert run_function_with_persistent_simulation_app(_test_deformable_reset_and_initial_pose)


def test_deformable_placement_is_not_immediate_task_success():
    assert run_function_with_persistent_simulation_app(_test_deformable_placement_is_not_immediate_task_success)


@pytest.mark.parametrize(
    "physics_backend",
    ["physx", pytest.param("newton", marks=pytest.mark.with_newton)],
)
def test_droid_deformable_environment(physics_backend):
    assert run_function_with_persistent_simulation_app(
        _test_droid_deformable_environment,
        physics_backend=physics_backend,
    )
