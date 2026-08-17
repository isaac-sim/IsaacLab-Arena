# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Replicator kitchen background physics."""

from __future__ import annotations

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_L_SHAPE_SPEC = _REPO_ROOT / "isaaclab_arena_environments/kitchen_bench/replicator_kitchen_l_shape_mustard_bowl.yaml"
_TRAY_RELATIVE_PATH = "TrayA01_01/TrayA01_OnlineVisual/Geometry/sm_kitchenware_tray_a01_obj_00"


def _test_replicator_kitchen_background_physics(simulation_app) -> bool:
    import torch

    import warp as wp
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.utils.usd.rigid_bodies import get_joint_connected_rigid_body_paths

    env = None
    try:
        spec = ArenaEnvGraphSpec.from_yaml(_L_SHAPE_SPEC)
        arena_env = spec.to_arena_env()
        background = arena_env.scene.assets["replicator_kitchen_l_shape"]
        assert not background.has_pose_reset_event(), "Fixture reset must not masquerade as a root-pose reset"
        args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "2"])
        env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
        env.reset()
        base_env = env.unwrapped

        stage = base_env.sim.stage
        physics_view = base_env.sim.physics_manager.get_physics_sim_view()
        indices = wp.from_torch(torch.tensor([0], dtype=torch.int32, device=base_env.device))
        tray_views = []
        initial_transforms = []
        for env_id in range(2):
            background_path = f"/World/envs/env_{env_id}/replicator_kitchen_l_shape"
            background_prim = stage.GetPrimAtPath(background_path)
            assert background_prim.IsValid(), f"Missing Replicator background at {background_path}"
            joint_connected_paths = set()
            descendant_target_count = 0
            for prim in Usd.PrimRange(background_prim):
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                    for target in relationship.GetTargets():
                        target_prim = stage.GetPrimAtPath(target)
                        body_prim = target_prim
                        while body_prim.IsValid() and not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            body_prim = body_prim.GetParent()
                        if not body_prim.IsValid():
                            continue
                        if not target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            descendant_target_count += 1
                        joint_connected_paths.add(str(body_prim.GetPath()))
            assert joint_connected_paths, "Expected joint-connected cabinet rigid bodies in the L-shaped kitchen"
            assert descendant_target_count, "Expected joints that target children of their owning rigid bodies"
            for path in joint_connected_paths:
                rigid_body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
                assert rigid_body.GetKinematicEnabledAttr().Get() is not True, f"Jointed rigid body was frozen: {path}"

            tray_path = f"{background_path}/{_TRAY_RELATIVE_PATH}"
            tray_prim = stage.GetPrimAtPath(tray_path)
            assert tray_prim.IsValid(), f"Replicator kitchen tray path changed: {tray_path}"
            assert (
                UsdPhysics.RigidBodyAPI(tray_prim).GetKinematicEnabledAttr().Get() is True
            ), f"Tray was not made kinematic: {tray_path}"
            tray_view = physics_view.create_rigid_body_view(tray_path)
            assert tray_view.count == 1
            tray_views.append(tray_view)
            initial_transforms.append(wp.to_torch(tray_view.get_transforms()).clone())

            velocity = torch.zeros((1, 6), device=base_env.device)
            velocity[0, 0] = 2.0
            tray_view.set_velocities(wp.from_torch(velocity.contiguous()), indices=indices)

        for _ in range(30):
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)

        for tray_view, initial_transform in zip(tray_views, initial_transforms, strict=True):
            final_transform = wp.to_torch(tray_view.get_transforms()).clone()
            position_error = torch.linalg.vector_norm(final_transform[0, :3] - initial_transform[0, :3]).item()
            assert position_error < 1.0e-6, f"Kinematic tray moved {position_error:.6f} m"

        background_prim = stage.GetPrimAtPath("/World/envs/env_0/replicator_kitchen_l_shape")
        fridge_door_path = next(
            path
            for path in get_joint_connected_rigid_body_paths(background_prim)
            if "refrigerator_a01_door_right_01_obj_00" in path
        )
        fridge_door_path_pattern = fridge_door_path.replace("/env_0/", "/env_*/", 1)
        fridge_door_view = physics_view.create_rigid_body_view(fridge_door_path_pattern)
        assert fridge_door_view.count == 2
        authored_fridge_transforms = wp.to_torch(fridge_door_view.get_transforms()).clone()

        frozen_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(background_prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
        ]
        assert frozen_paths, "Expected frozen decorative props in the L-shaped kitchen"

        displaced_fridge_transforms = authored_fridge_transforms.clone()
        displaced_fridge_transforms[:, 0] += 0.1
        all_indices = wp.from_torch(torch.tensor([0, 1], dtype=torch.int32, device=base_env.device))
        fridge_door_view.set_transforms(wp.from_torch(displaced_fridge_transforms), indices=all_indices)
        fridge_door_view.set_velocities(wp.from_torch(torch.ones((2, 6), device=base_env.device)), indices=all_indices)

        env_0 = torch.tensor([0], device=base_env.device, dtype=torch.int32)
        env_1 = torch.tensor([1], device=base_env.device, dtype=torch.int32)
        base_env.reset(env_ids=env_0)
        partially_reset_fridge_transforms = wp.to_torch(fridge_door_view.get_transforms()).clone()
        partially_reset_fridge_velocities = wp.to_torch(fridge_door_view.get_velocities()).clone()
        assert torch.allclose(
            partially_reset_fridge_transforms[0], authored_fridge_transforms[0], atol=5.0e-5, rtol=0.0
        )
        assert torch.max(torch.abs(partially_reset_fridge_velocities[0])).item() < 1.0e-5
        assert not torch.allclose(
            partially_reset_fridge_transforms[1], authored_fridge_transforms[1], atol=1.0e-3, rtol=0.0
        )

        base_env.reset(env_ids=env_1)
        reset_fridge_transforms = wp.to_torch(fridge_door_view.get_transforms()).clone()
        reset_fridge_velocities = wp.to_torch(fridge_door_view.get_velocities()).clone()
        assert torch.allclose(reset_fridge_transforms, authored_fridge_transforms, atol=5.0e-5, rtol=0.0)
        assert torch.max(torch.abs(reset_fridge_velocities)).item() < 1.0e-5

        articulation_root = next(
            prim for prim in Usd.PrimRange(background_prim) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        articulation_path_pattern = str(articulation_root.GetPath()).replace("/env_0/", "/env_*/", 1)
        articulation_view = physics_view.create_articulation_view(articulation_path_pattern)
        assert articulation_view.count == 2
        authored_root_transforms = wp.to_torch(articulation_view.get_root_transforms()).clone()
        authored_dof_positions = wp.to_torch(articulation_view.get_dof_positions()).clone()

        displaced_root_transforms = authored_root_transforms.clone()
        displaced_root_transforms[:, 0] += 0.05
        displaced_dof_positions = authored_dof_positions.clone()
        displaced_dof_positions[:, 0] += 0.1
        articulation_view.set_root_transforms(
            wp.from_torch(displaced_root_transforms.contiguous()), indices=all_indices
        )
        articulation_view.set_dof_positions(wp.from_torch(displaced_dof_positions.contiguous()), indices=all_indices)
        articulation_view.set_root_velocities(
            wp.from_torch(torch.ones((2, 6), device=base_env.device)), indices=all_indices
        )
        articulation_view.set_dof_velocities(
            wp.from_torch(torch.ones_like(displaced_dof_positions)), indices=all_indices
        )

        base_env.reset(env_ids=env_0)
        reset_root_transforms = wp.to_torch(articulation_view.get_root_transforms()).clone()
        reset_dof_positions = wp.to_torch(articulation_view.get_dof_positions()).clone()
        reset_root_velocities = wp.to_torch(articulation_view.get_root_velocities()).clone()
        reset_dof_velocities = wp.to_torch(articulation_view.get_dof_velocities()).clone()
        assert torch.allclose(reset_root_transforms[0], authored_root_transforms[0], atol=5.0e-5, rtol=0.0)
        assert torch.allclose(reset_dof_positions[0], authored_dof_positions[0], atol=1.0e-6, rtol=0.0)
        assert torch.max(torch.abs(reset_root_velocities[0])).item() < 1.0e-5
        assert torch.max(torch.abs(reset_dof_velocities[0])).item() < 1.0e-5
        assert not torch.allclose(reset_root_transforms[1], authored_root_transforms[1], atol=1.0e-3, rtol=0.0)
        assert not torch.allclose(reset_dof_positions[1], authored_dof_positions[1], atol=1.0e-3, rtol=0.0)

        base_env.reset(env_ids=env_1)
        fully_reset_root_transforms = wp.to_torch(articulation_view.get_root_transforms()).clone()
        fully_reset_dof_positions = wp.to_torch(articulation_view.get_dof_positions()).clone()
        assert torch.allclose(fully_reset_root_transforms, authored_root_transforms, atol=5.0e-5, rtol=0.0)
        assert torch.allclose(fully_reset_dof_positions, authored_dof_positions, atol=1.0e-6, rtol=0.0)
        for path in frozen_paths:
            assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get() is True
    finally:
        if env is not None:
            env.close()
    return True


def test_replicator_kitchen_background_physics():
    """Replicator props stay fixed while both fixture representations reset per environment."""
    result = run_function_with_persistent_simulation_app(
        _test_replicator_kitchen_background_physics,
        headless=True,
    )
    assert result
