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
        all_indices = wp.from_torch(torch.tensor([0, 1], dtype=torch.int32, device=base_env.device))
        for env_id in range(2):
            background_path = f"/World/envs/env_{env_id}/replicator_kitchen_l_shape"
            background_prim = stage.GetPrimAtPath(background_path)
            assert background_prim.IsValid(), f"Missing Replicator background at {background_path}"
            rigid_body_prims = [prim for prim in Usd.PrimRange(background_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert rigid_body_prims, f"Expected embedded rigid bodies under {background_path}"
            kinematic_body_paths = [
                str(prim.GetPath())
                for prim in rigid_body_prims
                if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
            ]
            assert not kinematic_body_paths, f"Background bodies must remain dynamic: {kinematic_body_paths}"

        background_prim = stage.GetPrimAtPath("/World/envs/env_0/replicator_kitchen_l_shape")
        tray_path = f"{background_prim.GetPath()}/{_TRAY_RELATIVE_PATH}"
        tray_path_pattern = tray_path.replace("/env_0/", "/env_*/", 1)
        tray_view = physics_view.create_rigid_body_view(tray_path_pattern)
        assert tray_view.count == 2
        initial_tray_transforms = wp.to_torch(tray_view.get_transforms()).clone()

        upward_velocity = torch.zeros((2, 6), device=base_env.device)
        upward_velocity[:, 2] = 2.0
        tray_view.set_velocities(wp.from_torch(upward_velocity), indices=all_indices)
        for _ in range(3):
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)
        moved_tray_transforms = wp.to_torch(tray_view.get_transforms()).clone()
        tray_displacement = torch.linalg.vector_norm(
            moved_tray_transforms[:, :3] - initial_tray_transforms[:, :3], dim=1
        )
        assert torch.all(tray_displacement > 1.0e-3), "Loose trays did not respond to applied velocity"

        env.reset()
        reset_tray_transforms = wp.to_torch(tray_view.get_transforms()).clone()
        reset_tray_velocities = wp.to_torch(tray_view.get_velocities()).clone()
        assert torch.allclose(reset_tray_transforms, initial_tray_transforms, atol=5.0e-5, rtol=0.0)
        assert torch.max(torch.abs(reset_tray_velocities)).item() < 1.0e-5

        fridge_door_path = next(
            str(prim.GetPath())
            for prim in Usd.PrimRange(background_prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "refrigerator_a01_door_right_01_obj_00" in str(prim.GetPath())
        )
        fridge_door_path_pattern = fridge_door_path.replace("/env_0/", "/env_*/", 1)
        fridge_door_view = physics_view.create_rigid_body_view(fridge_door_path_pattern)
        assert fridge_door_view.count == 2
        authored_fridge_transforms = wp.to_torch(fridge_door_view.get_transforms()).clone()

        displaced_fridge_transforms = authored_fridge_transforms.clone()
        displaced_fridge_transforms[:, 0] += 0.1
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
    finally:
        if env is not None:
            env.close()
    return True


def test_replicator_kitchen_background_physics():
    """Embedded kitchen bodies remain dynamic and reset through both PhysX representations."""
    result = run_function_with_persistent_simulation_app(
        _test_replicator_kitchen_background_physics,
        headless=True,
    )
    assert result
