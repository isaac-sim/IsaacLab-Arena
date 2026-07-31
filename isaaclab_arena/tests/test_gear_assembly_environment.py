# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the Arena Gear Assembly scene setup."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_gear_assembly_scene_and_newton_cfg(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.assets import (
        GEAR_GREEN_DIFFUSE_COLOR,
        GEAR_GREEN_VISUAL_MATERIAL_PATH,
        NEWTON_GEAR_ANGULAR_DAMPING,
        NEWTON_GEAR_BASE_MESH_APPROXIMATION,
        NEWTON_GEAR_CONTACT_OFFSET,
        NEWTON_GEAR_LINEAR_DAMPING,
        NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY,
        NEWTON_GEAR_MESH_APPROXIMATION,
        spawn_maple_table_top_collision,
        spawn_newton_maple_table_usd,
        spawn_newton_mesh_collision_usd,
    )
    from isaaclab_arena.tasks.gear_assembly.events import randomize_gears_and_base_pose_with_inactive_gear_parking
    from isaaclab_arena.tasks.gear_assembly.specs import (
        DROID_BASE_GEAR_POSE,
        GEAR_POSE_RANGE,
        GEAR_TABLETOP_ORIENTATION_XYZW,
        GEAR_TABLETOP_PARKING_POSITIONS,
        MAPLE_TABLE_POSE,
        MAPLE_TABLE_TOP_COLLISION_POSE,
        MAPLE_TABLE_TOP_COLLISION_SIZE,
        MAPLE_TABLE_TOP_COLLISION_THICKNESS,
        SELECTED_GEAR_POS_RANGE,
    )
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    arena_env = GearAssemblyEnvironment().build(GearAssemblyEnvironmentCfg())

    assert arena_env.name == "gear_assembly"
    assert arena_env.embodiment.name == "droid_abs_joint_pos"
    assert arena_env.rl_framework_entry_point is None
    assert arena_env.rl_policy_cfg is None

    assets = arena_env.scene.assets
    assert list(assets) == [
        "ground",
        "maple_table_robolab",
        "maple_table_top_collision",
        "factory_gear_base",
        "factory_gear_small",
        "factory_gear_medium",
        "factory_gear_large",
        "light",
        "table",
    ]
    assert assets["maple_table_robolab"].get_initial_pose() == MAPLE_TABLE_POSE
    assert assets["maple_table_robolab"].object_cfg.spawn.rigid_props.kinematic_enabled is True
    assert assets["maple_table_robolab"].object_cfg.spawn.func == spawn_newton_maple_table_usd
    tabletop_collider_pose = assets["maple_table_top_collision"].get_initial_pose()
    assert tabletop_collider_pose.position_xyz[:2] == MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[:2]
    assert (
        abs(
            tabletop_collider_pose.position_xyz[2]
            - (MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[2] - MAPLE_TABLE_TOP_COLLISION_THICKNESS / 2.0)
        )
        < 1e-6
    )
    assert assets["maple_table_top_collision"].object_cfg.spawn.size == (
        *MAPLE_TABLE_TOP_COLLISION_SIZE,
        MAPLE_TABLE_TOP_COLLISION_THICKNESS,
    )
    assert assets["maple_table_top_collision"].object_cfg.spawn.rigid_props.kinematic_enabled is True
    assert assets["maple_table_top_collision"].object_cfg.spawn.visible is True
    assert assets["maple_table_top_collision"].object_cfg.spawn.func == spawn_maple_table_top_collision
    assert assets["maple_table_top_collision"].object_cfg.spawn.visual_material is None
    assert assets["table"].parent_asset is assets["maple_table_robolab"]
    assert abs(assets["table"].get_initial_pose().position_xyz[2] - DROID_BASE_GEAR_POSE.position_xyz[2]) < 1e-6
    assert assets["factory_gear_base"].object_cfg.spawn.activate_contact_sensors is False
    assert assets["factory_gear_base"].object_cfg.spawn.rigid_props.kinematic_enabled is True
    assert assets["factory_gear_base"].object_cfg.spawn.visual_material is None
    assert assets["factory_gear_small"].object_cfg.spawn.activate_contact_sensors is False
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.kinematic_enabled is False
    assert assets["factory_gear_base"].object_cfg.spawn.func == spawn_newton_mesh_collision_usd
    assert assets["factory_gear_small"].object_cfg.spawn.func == spawn_newton_mesh_collision_usd
    for gear_name in (
        "factory_gear_small",
        "factory_gear_medium",
        "factory_gear_large",
    ):
        gear_spawn = assets[gear_name].object_cfg.spawn
        assert gear_spawn.visual_material.diffuse_color == GEAR_GREEN_DIFFUSE_COLOR
        assert gear_spawn.visual_material_path == GEAR_GREEN_VISUAL_MATERIAL_PATH
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.linear_damping == NEWTON_GEAR_LINEAR_DAMPING
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.angular_damping == NEWTON_GEAR_ANGULAR_DAMPING
    assert (
        assets["factory_gear_small"].object_cfg.spawn.rigid_props.max_depenetration_velocity
        == NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY
    )
    assert assets["factory_gear_small"].object_cfg.spawn.collision_props.contact_offset == NEWTON_GEAR_CONTACT_OFFSET
    assert NEWTON_GEAR_MESH_APPROXIMATION == "convexDecomposition"
    assert NEWTON_GEAR_BASE_MESH_APPROXIMATION == "convexDecomposition"
    assert GEAR_POSE_RANGE["z"] == [0.0, 0.0]
    assert GEAR_POSE_RANGE["roll"] == [0.0, 0.0]
    assert GEAR_POSE_RANGE["pitch"] == [0.0, 0.0]
    assert SELECTED_GEAR_POS_RANGE == {
        "x": [-0.02, 0.02],
        "y": [-0.02, 0.02],
        "z": [0.0575, 0.0775],
    }
    assert (
        arena_env.task.events_cfg.randomize_gears_and_base_pose.func
        == randomize_gears_and_base_pose_with_inactive_gear_parking
    )
    assert (
        arena_env.task.events_cfg.randomize_gears_and_base_pose.params["parking_positions"]
        == GEAR_TABLETOP_PARKING_POSITIONS
    )
    assert (
        arena_env.task.events_cfg.randomize_gears_and_base_pose.params["parking_orientation_xyzw"]
        == GEAR_TABLETOP_ORIENTATION_XYZW
    )
    assert "selected_parking_positions" not in arena_env.task.events_cfg.randomize_gears_and_base_pose.params
    assert "selected_orientation_xyzw" not in arena_env.task.events_cfg.randomize_gears_and_base_pose.params

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1))
    env_cfg, _ = builder.compose_manager_cfg()
    solver_cfg = env_cfg.sim.physics.solver_cfg

    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"
    assert type(solver_cfg).__name__ == "MJWarpSolverCfg"
    assert solver_cfg.solver == "newton"
    assert solver_cfg.integrator == "implicitfast"
    assert solver_cfg.use_mujoco_contacts is False
    assert env_cfg.sim.physics.default_shape_cfg.gap == 0.0
    assert env_cfg.scene.robot.spawn.usd_path.endswith("_newton_inertia.usd")
    assert env_cfg.scene.replicate_physics is True
    assert (
        env_cfg.events.randomize_gears_and_base_pose.params["selected_parking_positions"]
        == GEAR_TABLETOP_PARKING_POSITIONS
    )
    assert (
        env_cfg.events.randomize_gears_and_base_pose.params["selected_orientation_xyzw"]
        == GEAR_TABLETOP_ORIENTATION_XYZW
    )
    assert env_cfg.sim.dt == 1.0 / 120.0
    assert env_cfg.decimation == 4
    assert env_cfg.episode_length_s == 6.66
    return True


def test_gear_assembly_scene_and_newton_cfg():
    assert run_simulation_app_function(_test_gear_assembly_scene_and_newton_cfg, headless=True)


def _test_gear_assembly_newton_gears_settle(simulation_app):  # noqa: C901
    import torch

    import warp as wp
    from isaaclab.sim.utils import get_current_stage
    from isaaclab.utils.math import quat_apply
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.assets import (
        GEAR_GREEN_DIFFUSE_COLOR,
        GEAR_GREEN_VISUAL_MATERIAL_PATH,
        MAPLE_TABLE_TOP_COLLISION_COLOR,
        NEWTON_GEAR_BASE_MESH_APPROXIMATION,
        NEWTON_GEAR_MESH_APPROXIMATION,
    )
    from isaaclab_arena.tasks.gear_assembly.specs import (
        GEAR_TABLETOP_PARKING_POSITIONS,
        GEAR_TABLETOP_PARKING_Z,
        MAPLE_TABLE_TOP_COLLISION_POSE,
        MAPLE_TABLE_TOP_COLLISION_SIZE,
        MAPLE_TABLE_TOP_COLLISION_THICKNESS,
    )
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    arena_env = GearAssemblyEnvironment().build(
        GearAssemblyEnvironmentCfg(enable_cameras=False, embodiment="droid_rel_joint_pos")
    )
    arena_env.name = "gear_assembly_newton_settle_regression"
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1))
    env_cfg, env_kwargs = builder.compose_manager_cfg()

    env_cfg.scene.robot.init_state.pos = (10.0, 10.0, 0.0)
    env_cfg.episode_length_s = 30.0
    for name in (
        "init_franka_arm_pose",
        "randomize_franka_joint_state",
        "set_robot_to_grasp_pose",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    for name in ("time_out", "gear_dropped", "gear_orientation_exceeded"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    env = builder.make_registered(env_cfg, env_kwargs)
    uenv = env.unwrapped
    env.reset()
    device = uenv.device
    env_id = torch.tensor([0], device=device, dtype=torch.long)
    zero_velocity = torch.zeros((1, 6), device=device)
    far_pose = torch.tensor([[8.0, 8.0, 0.3, 1.0, 0.0, 0.0, 0.0]], device=device)
    gear_names = ("factory_gear_small", "factory_gear_medium", "factory_gear_large")
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=device)

    table_center = torch.tensor(MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz, device=device)
    table_size = torch.tensor(MAPLE_TABLE_TOP_COLLISION_SIZE, device=device)
    step_dt = uenv.step_dt
    stage = get_current_stage()
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    collision_local_corners = {}

    asset_stage_paths = {
        "factory_gear_base": (
            "/World/envs/env_0/FactoryGearBase",
            "/World/envs/env_0/FactoryGearBase/factory_gear_base",
            NEWTON_GEAR_BASE_MESH_APPROXIMATION,
        ),
        "factory_gear_small": (
            "/World/envs/env_0/FactoryGearSmall",
            "/World/envs/env_0/FactoryGearSmall/factory_gear_small",
            NEWTON_GEAR_MESH_APPROXIMATION,
        ),
        "factory_gear_medium": (
            "/World/envs/env_0/FactoryGearMedium",
            "/World/envs/env_0/FactoryGearMedium/factory_gear_medium",
            NEWTON_GEAR_MESH_APPROXIMATION,
        ),
        "factory_gear_large": (
            "/World/envs/env_0/FactoryGearLarge",
            "/World/envs/env_0/FactoryGearLarge/factory_gear_large",
            NEWTON_GEAR_MESH_APPROXIMATION,
        ),
    }

    def _box_corners(box) -> list[list[float]]:
        box_min = box.GetMin()
        box_max = box.GetMax()
        return [
            [x, y, z]
            for x in (box_min[0], box_max[0])
            for y in (box_min[1], box_max[1])
            for z in (box_min[2], box_max[2])
        ]

    def _assert_newton_collision_meshes(
        asset_name: str, prim_path: str, rigid_prim_path: str, approximation: str
    ) -> None:
        root = stage.GetPrimAtPath(prim_path)
        rigid_root = stage.GetPrimAtPath(rigid_prim_path)
        assert root.IsValid(), f"{asset_name} prim is missing from the stage"
        assert rigid_root.IsValid(), f"{asset_name} rigid-body prim is missing from the stage"
        collision_meshes = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.GetTypeName() == "Mesh" and "/collisions" in str(prim.GetPath())
        ]
        assert collision_meshes, f"{asset_name} has no concrete collision mesh leaves"
        local_corners = []
        for prim in collision_meshes:
            assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} is missing CollisionAPI"
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), f"{prim.GetPath()} is missing MeshCollisionAPI"
            authored_approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            assert authored_approximation == approximation
            local_corners.extend(_box_corners(bbox_cache.ComputeRelativeBound(prim, rigid_root).ComputeAlignedBox()))
        collision_local_corners[asset_name] = torch.tensor(local_corners, device=device, dtype=torch.float32)

    def _world_collision_z_bounds(asset_name: str, root_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corners = collision_local_corners[asset_name]
        world_corners = root_pose[:3] + quat_apply(root_pose[3:7].repeat(corners.shape[0], 1), corners)
        return world_corners[:, 2].min(), world_corners[:, 2].max()

    def _shape_indices(labels: list[str], asset_fragment: str) -> list[int]:
        matches = [
            index
            for index, label in enumerate(labels)
            if asset_fragment in label
            and "/collisions/" in label
            and "visuals" not in label
            and not label.endswith("_visual")
        ]
        assert matches, f"Expected Newton collision shapes for {asset_fragment}"
        return matches

    def _assert_newton_contact_pairs() -> None:
        model = NewtonManager._model
        labels = list(model.shape_label)
        pairs = wp.to_torch(model.shape_contact_pairs).cpu().to(torch.long)
        pair_set = {tuple(sorted(pair.tolist())) for pair in pairs}

        base_shapes = _shape_indices(labels, "FactoryGearBase")
        gear_shapes = {
            "factory_gear_small": _shape_indices(labels, "FactoryGearSmall"),
            "factory_gear_medium": _shape_indices(labels, "FactoryGearMedium"),
            "factory_gear_large": _shape_indices(labels, "FactoryGearLarge"),
        }
        for asset_name, asset_shapes in gear_shapes.items():
            for gear_shape in asset_shapes:
                assert any(
                    tuple(sorted((base_shape, gear_shape))) in pair_set for base_shape in base_shapes
                ), f"{asset_name} shape {gear_shape} is not paired with gear base"

        table_top_shapes = [index for index, label in enumerate(labels) if "/maple_table_top_collision/" in label]
        assert table_top_shapes, "Newton model is missing the finite maple tabletop collision shape"
        table_top_shape = table_top_shapes[0]
        for asset_name, asset_shapes in {
            "factory_gear_base": base_shapes,
            **gear_shapes,
        }.items():
            for shape in asset_shapes:
                assert (
                    tuple(sorted((table_top_shape, shape))) in pair_set
                ), f"{asset_name} shape {shape} is not paired with table top"

    def _assert_finite_tabletop_proxy() -> None:
        proxy_mesh = stage.GetPrimAtPath("/World/envs/env_0/maple_table_top_collision/geometry/mesh")
        assert proxy_mesh.IsValid(), "Finite tabletop proxy mesh is missing from the stage"
        assert proxy_mesh.GetTypeName() == "Cube"
        proxy_box = bbox_cache.ComputeWorldBound(proxy_mesh).ComputeAlignedBox()
        proxy_extent = proxy_box.GetSize()
        assert abs(proxy_extent[0] - MAPLE_TABLE_TOP_COLLISION_SIZE[0]) < 1e-4
        assert abs(proxy_extent[1] - MAPLE_TABLE_TOP_COLLISION_SIZE[1]) < 1e-4
        assert abs(proxy_extent[2] - MAPLE_TABLE_TOP_COLLISION_THICKNESS) < 1e-4
        display_color = UsdGeom.PrimvarsAPI(proxy_mesh).GetPrimvar("displayColor").Get()
        assert display_color is not None
        authored_color = tuple(display_color[0])
        assert all(
            abs(actual - expected) < 1e-6 for actual, expected in zip(authored_color, MAPLE_TABLE_TOP_COLLISION_COLOR)
        )

    def _assert_maple_table_top_newton_material() -> None:
        table_top = stage.GetPrimAtPath("/World/envs/env_0/maple_table_robolab/table/table_01/top")
        assert table_top.IsValid(), "Maple table top is missing from the stage"
        bound_material, _ = UsdShade.MaterialBindingAPI(table_top).ComputeBoundMaterial()
        assert (
            bound_material and bound_material.GetPrim().IsValid()
        ), "Maple table top has no bound Newton-readable material"
        material_path = str(bound_material.GetPrim().GetPath())
        assert material_path.endswith("/Looks/newton_maple_top")
        shader_prim = stage.GetPrimAtPath(f"{material_path}/OmniPBRShader")
        assert shader_prim.IsValid(), "Maple table top OmniPBR shader is missing"
        assert str(shader_prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()) == "OmniPBR"
        authored_color = tuple(shader_prim.GetAttribute("inputs:diffuse_color_constant").Get())
        assert all(
            abs(actual - expected) < 1e-6 for actual, expected in zip(authored_color, MAPLE_TABLE_TOP_COLLISION_COLOR)
        )

    def _assert_green_visual_material(prim_path: str) -> None:
        root = stage.GetPrimAtPath(prim_path)
        bound_material, _ = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial()
        assert bound_material and bound_material.GetPrim().IsValid(), f"{prim_path} has no bound visual material"
        assert str(bound_material.GetPrim().GetPath()) == f"{prim_path}/{GEAR_GREEN_VISUAL_MATERIAL_PATH}"
        shader_prim = stage.GetPrimAtPath(f"{prim_path}/{GEAR_GREEN_VISUAL_MATERIAL_PATH}/Shader")
        assert shader_prim.IsValid(), f"{prim_path} green preview shader is missing"
        authored_color = tuple(shader_prim.GetAttribute("inputs:diffuseColor").Get())
        assert all(abs(actual - expected) < 1e-6 for actual, expected in zip(authored_color, GEAR_GREEN_DIFFUSE_COLOR))

    def _park_except(active_name: str, include_base: bool) -> None:
        names = list(gear_names)
        if include_base:
            names.append("factory_gear_base")
        for asset_name in names:
            if asset_name == active_name:
                continue
            asset = uenv.scene[asset_name]
            asset.write_root_pose_to_sim_index(root_pose=far_pose, env_ids=env_id)
            asset.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_id)

    def _root_motion(
        asset_name: str, steps: int, sample_from_step: int
    ) -> tuple[torch.Tensor, float, float, float, float, float]:
        asset = uenv.scene[asset_name]
        recent_poses = []
        for step in range(steps + 1):
            root_pose = asset.data.root_link_pose_w.torch[0]
            root_velocity = asset.data.root_com_vel_w.torch[0]
            assert torch.isfinite(root_pose).all(), f"{asset_name} produced non-finite root pose at step {step}"
            assert torch.isfinite(root_velocity).all(), f"{asset_name} produced non-finite root velocity at step {step}"
            if step >= sample_from_step:
                recent_poses.append(root_pose.detach().clone())
            env.step(action)
        recent_poses = torch.stack(recent_poses)
        recent_positions = recent_poses[:, :3]
        linear_speeds, angular_speeds = _finite_difference_speeds(recent_poses)
        excursion = torch.linalg.norm(recent_positions - recent_positions.mean(dim=0), dim=1).max().item()
        return (
            asset.data.root_link_pose_w.torch[0].detach().clone(),
            excursion,
            linear_speeds.max().item(),
            angular_speeds.max().item(),
            linear_speeds[-1].item(),
            angular_speeds[-1].item(),
        )

    def _finite_difference_speeds(
        poses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_delta = poses[1:, :3] - poses[:-1, :3]
        linear_speeds = torch.linalg.norm(position_delta, dim=1) / step_dt
        quat_dot = torch.abs(torch.sum(poses[1:, 3:7] * poses[:-1, 3:7], dim=1))
        angular_speeds = 2.0 * torch.acos(torch.clamp(quat_dot, max=1.0)) / step_dt
        return linear_speeds, angular_speeds

    def _recent_pose_metrics(
        poses: list[torch.Tensor],
    ) -> tuple[float, float, float, float, float]:
        recent_poses = torch.stack(poses)
        linear_speeds, angular_speeds = _finite_difference_speeds(recent_poses)
        excursion = (
            torch.linalg.norm(
                recent_poses[:, :3] - recent_poses[:, :3].mean(dim=0),
                dim=1,
            )
            .max()
            .item()
        )
        return (
            excursion,
            linear_speeds.max().item(),
            angular_speeds.max().item(),
            linear_speeds[-1].item(),
            angular_speeds[-1].item(),
        )

    for asset_name, (
        prim_path,
        rigid_prim_path,
        approximation,
    ) in asset_stage_paths.items():
        _assert_newton_collision_meshes(asset_name, prim_path, rigid_prim_path, approximation)
        if asset_name in gear_names:
            _assert_green_visual_material(prim_path)
    _assert_finite_tabletop_proxy()
    _assert_maple_table_top_newton_material()
    _assert_newton_contact_pairs()

    for gear_name in gear_names:
        env.reset()
        _park_except(gear_name, include_base=True)
        gear = uenv.scene[gear_name]
        table_settle_pose = torch.tensor(
            [[
                table_center[0].item(),
                table_center[1].item(),
                GEAR_TABLETOP_PARKING_Z,
                1.0,
                0.0,
                0.0,
                0.0,
            ]],
            device=device,
        )
        gear.write_root_pose_to_sim_index(root_pose=table_settle_pose, env_ids=env_id)
        gear.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_id)
        uenv.sim.forward()

        (
            table_state,
            table_excursion,
            table_max_linear,
            table_max_angular,
            table_final_linear,
            table_final_angular,
        ) = _root_motion(gear_name, steps=600, sample_from_step=540)
        table_xy_error = torch.abs(table_state[:2] - table_center[:2])
        table_collision_min_z, _ = _world_collision_z_bounds(gear_name, table_state)
        assert (table_xy_error <= table_size / 2).all(), f"{gear_name} slid off the tabletop: {table_state[:3]}"
        assert (
            abs(table_collision_min_z.item() - table_center[2].item()) < 0.006
        ), f"{gear_name} collision mesh did not settle on the tabletop: {table_collision_min_z}"
        assert table_excursion < 0.003, f"{gear_name} did not settle on the tabletop; max excursion={table_excursion}"
        assert table_max_linear < 0.08, f"{gear_name} tabletop linear velocity is high"
        assert table_max_angular < 0.2, f"{gear_name} tabletop angular velocity is high"
        assert table_final_linear < 0.02, f"{gear_name} final tabletop linear velocity is high"
        assert table_final_angular < 0.2, f"{gear_name} final tabletop angular velocity is high"

    for gear_key in ("gear_small", "gear_medium", "gear_large"):
        gear_type_cfg = uenv.event_manager.get_term_cfg("randomize_gear_type")
        gear_type_cfg.params["gear_types"] = [gear_key]
        uenv.event_manager.set_term_cfg("randomize_gear_type", gear_type_cfg)
        env.reset()

        all_positions = {asset_name: [] for asset_name in gear_names}
        recent_poses = {asset_name: [] for asset_name in gear_names}
        for step in range(601):
            for asset_name in gear_names:
                asset = uenv.scene[asset_name]
                root_pose = asset.data.root_link_pose_w.torch[0]
                root_velocity = asset.data.root_com_vel_w.torch[0]
                assert torch.isfinite(root_pose).all(), f"{asset_name} reset pose became non-finite at step {step}"
                assert torch.isfinite(
                    root_velocity
                ).all(), f"{asset_name} reset velocity became non-finite at step {step}"
                all_positions[asset_name].append(root_pose[:3].detach().clone())
                if step >= 540:
                    recent_poses[asset_name].append(root_pose.detach().clone())
            env.step(action)

        base_pose_w = uenv.scene["factory_gear_base"].data.root_link_pose_w.torch[0]
        base_velocity_w = uenv.scene["factory_gear_base"].data.root_com_vel_w.torch[0]
        base_collision_min_z, _ = _world_collision_z_bounds("factory_gear_base", base_pose_w)
        assert torch.isfinite(base_pose_w).all(), "factory_gear_base reset pose became non-finite"
        assert torch.isfinite(base_velocity_w).all(), "factory_gear_base reset velocity became non-finite"
        assert (torch.abs(base_pose_w[:2] - table_center[:2]) <= table_size / 2).all()
        assert abs(base_collision_min_z.item() - table_center[2].item()) < 0.005
        assert torch.linalg.norm(base_velocity_w).item() < 1e-6
        for asset_name in gear_names:
            asset = uenv.scene[asset_name]
            root_pose = asset.data.root_link_pose_w.torch[0]
            trajectory = torch.stack(all_positions[asset_name])
            (
                excursion,
                max_linear_speed,
                max_angular_speed,
                final_linear_speed,
                final_angular_speed,
            ) = _recent_pose_metrics(recent_poses[asset_name])
            xy_error = torch.abs(root_pose[:2] - table_center[:2])
            trajectory_xy_error = torch.abs(trajectory[:, :2] - table_center[:2])
            collision_min_z, collision_max_z = _world_collision_z_bounds(asset_name, root_pose)

            assert (xy_error <= table_size / 2).all(), f"{asset_name} left the tabletop in reset: {root_pose[:3]}"
            assert (
                trajectory_xy_error <= table_size / 2
            ).all(), f"{asset_name} left the tabletop during reset: {trajectory[-1]}"
            assert collision_min_z > table_center[2] - 0.006, f"{asset_name} fell through the tabletop: {root_pose[:3]}"
            assert max_linear_speed < 0.15, f"{asset_name} reset linear velocity is high"
            assert max_angular_speed < 5.0, f"{asset_name} reset angular velocity is high"
            assert final_linear_speed < 0.1, f"{asset_name} final reset linear velocity is high"
            assert final_angular_speed < 2.0, f"{asset_name} final reset angular velocity is high"
            assert (
                collision_max_z < table_center[2] + 0.11
            ), f"{asset_name} is floating above the tabletop: {root_pose[:3]}"
            assert excursion < 0.005, f"{asset_name} reset excursion is high: {excursion}"
            parking_key = asset_name.removeprefix("factory_")
            expected_position = torch.tensor(GEAR_TABLETOP_PARKING_POSITIONS[parking_key], device=device)
            assert torch.linalg.norm(root_pose[:2] - expected_position[:2]).item() < 0.03

    env.close()
    return True


def test_gear_assembly_newton_gears_settle():
    assert run_simulation_app_function(_test_gear_assembly_newton_gears_settle, headless=True)
