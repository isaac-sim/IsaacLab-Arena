# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the Arena Gear Assembly scene setup."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_gear_assembly_scene_and_newton_cfg(simulation_app):
    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.actions import NewtonDroidDifferentialInverseKinematicsAction
    from isaaclab_arena.tasks.gear_assembly.assets import (
        GEAR_GREEN_DIFFUSE_COLOR,
        GEAR_GREEN_VISUAL_MATERIAL_PATH,
        NEWTON_GEAR_CONTACT_OFFSET,
        NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY,
        spawn_maple_table_top_collision,
        spawn_newton_maple_table_usd,
        spawn_newton_mesh_collision_usd,
    )
    from isaaclab_arena.tasks.gear_assembly.events import (
        randomize_gears_and_base_pose_with_inactive_gear_parking,
        set_newton_rigid_body_material,
    )
    from isaaclab_arena.tasks.gear_assembly.specs import (
        DROID_BASE_GEAR_POSE,
        GEAR_ASSEMBLED_ANGULAR_VELOCITY_THRESHOLD,
        GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS,
        GEAR_ASSEMBLED_LINEAR_VELOCITY_THRESHOLD,
        GEAR_ASSEMBLED_SUPPORT_Z_THRESHOLD,
        GEAR_ASSEMBLED_UPRIGHT_AXIS_THRESHOLD_DEG,
        GEAR_ASSEMBLED_XY_THRESHOLD,
        GEAR_ASSEMBLED_Z_THRESHOLD,
        GEAR_POSE_RANGE,
        GEAR_TABLETOP_PARKING_POSITIONS,
        MAPLE_TABLE_POSE,
        MAPLE_TABLE_TOP_COLLISION_POSE,
        MAPLE_TABLE_TOP_COLLISION_SIZE,
        MAPLE_TABLE_TOP_COLLISION_THICKNESS,
        NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE,
        NEWTON_GEAR_ASSEMBLED_SUPPORT_Z_OFFSET,
        NEWTON_GEAR_OFFSETS,
        NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW,
        NEWTON_GEAR_TABLETOP_PARKING_POSITIONS,
        SELECTED_GEAR_POS_RANGE,
        get_droid_robot_spec,
    )
    from isaaclab_arena.tasks.gear_assembly.terminations import selected_gear_on_base
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    def assert_stock_droid(embodiment):
        assert (
            embodiment.action_config.arm_action.class_type
            == "isaaclab.envs.mdp.actions.task_space_actions:DifferentialInverseKinematicsAction"
        )
        assert embodiment.action_config.arm_action.body_name == "panda_link0"
        assert embodiment.scene_config.ee_frame.target_frames[0].prim_path == "{ENV_REGEX_NS}/Robot/panda_link0"
        assert embodiment.scene_config.robot.actuators["gripper"].joint_names_expr == ["finger_joint"]
        assert embodiment.scene_config.robot.spawn.rigid_props.disable_gravity is True

    assert_stock_droid(DroidDifferentialIKEmbodiment())
    arena_env = GearAssemblyEnvironment().build(GearAssemblyEnvironmentCfg())

    assert arena_env.name == "gear_assembly"
    assert arena_env.embodiment.name == "droid_differential_ik"
    assert arena_env.embodiment.action_config.arm_action.class_type == NewtonDroidDifferentialInverseKinematicsAction
    assert arena_env.embodiment.action_config.arm_action.body_name == "base_link"
    assert (
        arena_env.embodiment.scene_config.ee_frame.target_frames[0].prim_path
        == "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
    )
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
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.linear_damping == 0.0
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.angular_damping == 0.0
    assert (
        assets["factory_gear_small"].object_cfg.spawn.rigid_props.max_depenetration_velocity
        == NEWTON_GEAR_MAX_DEPENETRATION_VELOCITY
    )
    assert assets["factory_gear_small"].object_cfg.spawn.collision_props.contact_offset == NEWTON_GEAR_CONTACT_OFFSET
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
    for material_term_name in (
        "small_gear_physics_material",
        "medium_gear_physics_material",
        "large_gear_physics_material",
        "gear_base_physics_material",
        "robot_physics_material",
    ):
        assert getattr(arena_env.task.events_cfg, material_term_name).func == set_newton_rigid_body_material
    assert (
        arena_env.task.events_cfg.randomize_gears_and_base_pose.params["parking_positions"]
        == NEWTON_GEAR_TABLETOP_PARKING_POSITIONS
    )
    assert (
        arena_env.task.events_cfg.randomize_gears_and_base_pose.params["parking_orientation_xyzw"]
        == NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW
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
    assert solver_cfg.njmax == 512
    assert env_cfg.sim.physics.num_substeps == 12
    assert env_cfg.sim.physics.default_shape_cfg.gap == 0.0
    assert env_cfg.scene.robot.spawn.usd_path.endswith("_newton_droid.usd")
    assert env_cfg.scene.replicate_physics is True
    assert env_cfg.events.randomize_franka_joint_state is None
    assert (
        env_cfg.events.randomize_gears_and_base_pose.params["selected_parking_positions"]
        == NEWTON_GEAR_TABLETOP_PARKING_POSITIONS
    )
    assert (
        env_cfg.events.randomize_gears_and_base_pose.params["selected_orientation_xyzw"]
        == NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW
    )
    assert env_cfg.sim.dt == 1.0 / 120.0
    assert env_cfg.decimation == 4
    assert env_cfg.viewer.eye == (1.6, 1.2, 1.0)
    assert env_cfg.episode_length_s == 6.66
    assert env_cfg.observations.policy.concatenate_terms is False
    assert env_cfg.observations.policy.enable_corruption is False
    assert env_cfg.observations.policy.gripper_pos is not None
    assert env_cfg.terminations.success is not None
    assert env_cfg.terminations.success.func == selected_gear_on_base
    assert env_cfg.terminations.success.params["root_z_above_base"] == NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE
    assert env_cfg.terminations.success.params["root_xy_offset_from_base"] == NEWTON_GEAR_OFFSETS
    assert env_cfg.terminations.success.params["xy_threshold"] == GEAR_ASSEMBLED_XY_THRESHOLD
    assert env_cfg.terminations.success.params["z_threshold"] == GEAR_ASSEMBLED_Z_THRESHOLD
    assert (
        env_cfg.terminations.success.params["upright_axis_threshold_deg"] == GEAR_ASSEMBLED_UPRIGHT_AXIS_THRESHOLD_DEG
    )
    assert env_cfg.terminations.success.params["linear_velocity_threshold"] == GEAR_ASSEMBLED_LINEAR_VELOCITY_THRESHOLD
    assert (
        env_cfg.terminations.success.params["angular_velocity_threshold"] == GEAR_ASSEMBLED_ANGULAR_VELOCITY_THRESHOLD
    )
    assert env_cfg.terminations.success.params["support_z_offset"] == NEWTON_GEAR_ASSEMBLED_SUPPORT_Z_OFFSET
    assert env_cfg.terminations.success.params["base_support_prim_name"] == "platform"
    assert env_cfg.terminations.success.params["enabled_colliders_only"] is True
    assert env_cfg.terminations.success.params["support_z_threshold"] == GEAR_ASSEMBLED_SUPPORT_Z_THRESHOLD
    assert env_cfg.terminations.success.params["consecutive_success_steps"] == GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS
    assert env_cfg.terminations.time_out is None
    assert env_cfg.terminations.gear_dropped is None
    assert env_cfg.terminations.gear_orientation_exceeded is None

    randomized_env = GearAssemblyEnvironment().build(GearAssemblyEnvironmentCfg(mode="randomized"))
    assert randomized_env.task.observation_cfg.policy.concatenate_terms is True
    assert randomized_env.task.observation_cfg.policy.gripper_pos is None
    assert randomized_env.task.termination_cfg.time_out is not None
    assert randomized_env.task.termination_cfg.gear_dropped is not None
    assert randomized_env.task.termination_cfg.gear_orientation_exceeded is not None

    # Building the Newton environment must not mutate the registered DROID defaults.
    assert_stock_droid(DroidDifferentialIKEmbodiment())

    # Keep the existing PhysX task values unchanged.
    physx_env = GearAssemblyEnvironment().build(
        GearAssemblyEnvironmentCfg(physics_backend="physx", embodiment="droid_abs_joint_pos")
    )
    physx_robot_spec = get_droid_robot_spec()
    assert physx_robot_spec.gear_offsets_grasp == {
        "gear_small": [0.0, 0.076125, -0.19],
        "gear_medium": [0.0, 0.030375, -0.19],
        "gear_large": [0.0, -0.045375, -0.19],
    }
    assert physx_robot_spec.startup_materials["robot"] == (0.75, 0.75, 0.0)
    assert physx_env.task.events_cfg.randomize_gears_and_base_pose.params["parking_positions"] == (
        GEAR_TABLETOP_PARKING_POSITIONS
    )
    assert physx_env.task.events_cfg.set_robot_to_grasp_pose.params["pos_randomization_range"] == (
        physx_robot_spec.set_grasp_pos_randomization_range
    )
    assert "root_xy_offset_from_base" not in physx_env.task.termination_cfg.success.params
    assert "base_support_prim_name" not in physx_env.task.termination_cfg.success.params
    assert "enabled_colliders_only" not in physx_env.task.termination_cfg.success.params
    return True


def test_gear_assembly_scene_and_newton_cfg():
    assert run_simulation_app_function(_test_gear_assembly_scene_and_newton_cfg, headless=True)


def _test_gear_assembly_newton_differential_ik(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.actions import NewtonDroidDifferentialInverseKinematicsAction
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    arena_env = GearAssemblyEnvironment().build(GearAssemblyEnvironmentCfg(enable_cameras=False))
    arena_env.name = "gear_assembly_newton_differential_ik"
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env = builder.make_registered(env_cfg, env_kwargs)
    uenv = env.unwrapped
    env.reset()
    robot = uenv.scene["robot"]
    arm_action = uenv.action_manager.get_term("arm_action")

    assert isinstance(arm_action, NewtonDroidDifferentialInverseKinematicsAction)
    jacobian = arm_action.jacobian_w
    assert jacobian.shape == (1, 6, 7)
    assert torch.isfinite(jacobian).all()
    assert torch.linalg.norm(jacobian).item() > 2.0

    end_effector_index = robot.body_names.index("base_link")
    start_position = robot.data.body_pos_w.torch[:, end_effector_index].clone()
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=uenv.device)
    action[:, 2] = 0.01
    for _ in range(30):
        env.step(action)
    action[:, 2] = 0.0
    for _ in range(30):
        env.step(action)

    displacement = robot.data.body_pos_w.torch[:, end_effector_index] - start_position
    assert displacement[0, 2].item() > 0.015
    assert torch.linalg.norm(displacement[0, :2]).item() < 0.002
    assert torch.isfinite(robot.data.joint_pos.torch).all()
    assert torch.isfinite(robot.data.joint_vel.torch).all()

    env.close()
    return True


def test_gear_assembly_newton_differential_ik():
    assert run_simulation_app_function(_test_gear_assembly_newton_differential_ik, headless=True)


def _test_gear_assembly_newton_gears_settle(simulation_app):
    import torch

    import isaaclab.utils.math as math_utils
    import warp as wp
    from isaaclab.sim.utils import get_current_stage
    from isaaclab_newton.physics import NewtonManager
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.gear_assembly.specs import (
        DROID_GRIPPER_MIMIC_SIGNS,
        GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS,
        MAPLE_TABLE_TOP_COLLISION_POSE,
        NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE,
        NEWTON_GEAR_OFFSETS,
    )
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    arena_env = GearAssemblyEnvironment().build(
        GearAssemblyEnvironmentCfg(enable_cameras=False, embodiment="droid_abs_joint_pos")
    )
    arena_env.name = "gear_assembly_newton_stability"
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env = builder.make_registered(env_cfg, env_kwargs)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]

    model = NewtonManager.get_model()
    shape_labels = [str(label) for label in model.shape_label]
    friction = wp.to_torch(model.shape_material_mu)
    restitution = wp.to_torch(model.shape_material_restitution)
    gear_and_base_shape_indices = torch.tensor(
        [index for index, label in enumerate(shape_labels) if "/FactoryGear" in label],
        device=uenv.device,
    )
    finger_pad_shape_indices = torch.tensor(
        [index for index, label in enumerate(shape_labels) if label.endswith("/newton_pad_collision")],
        device=uenv.device,
    )
    assert len(gear_and_base_shape_indices) == 48 * uenv.num_envs
    assert len(finger_pad_shape_indices) == 2 * uenv.num_envs
    assert torch.all(friction[gear_and_base_shape_indices] == 0.75)
    assert torch.all(friction[finger_pad_shape_indices] == 2.0)
    assert torch.all(restitution[gear_and_base_shape_indices] == 0.0)
    assert torch.all(restitution[finger_pad_shape_indices] == 0.0)

    stage = get_current_stage()
    base_collision_root = stage.GetPrimAtPath("/World/envs/env_0/FactoryGearBase/factory_gear_base/newton_collisions")
    assert base_collision_root.IsValid()
    base_collision_names = {
        prim.GetName()
        for prim in Usd.PrimRange(base_collision_root)
        if prim.IsA(UsdGeom.Mesh) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    }
    assert base_collision_names == {"platform", "gear_small_peg", "gear_medium_peg", "gear_large_peg"}

    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    robotiq_collision_meshes = [
        prim for prim in Usd.PrimRange(robot_root) if prim.IsA(UsdGeom.Mesh) and "Robotiq_2F_85" in str(prim.GetPath())
    ]
    enabled_robotiq_collision_meshes = {
        prim.GetParent().GetName(): prim.GetName()
        for prim in robotiq_collision_meshes
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    }
    assert enabled_robotiq_collision_meshes == {
        "left_inner_finger": "newton_pad_collision",
        "right_inner_finger": "newton_pad_collision",
    }
    pad_collision_meshes = [prim for prim in robotiq_collision_meshes if prim.GetName() == "newton_pad_collision"]
    assert all(UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide for prim in pad_collision_meshes)
    render_meshes = [prim for prim in robotiq_collision_meshes if prim.GetName() != "newton_pad_collision"]
    assert len(render_meshes) == 11
    assert all(UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited for prim in render_meshes)

    for asset_name in ("Small", "Medium", "Large"):
        collision_root = stage.GetPrimAtPath(
            f"/World/envs/env_0/FactoryGear{asset_name}/factory_gear_{asset_name.lower()}/newton_collisions"
        )
        assert collision_root.IsValid()
        collision_meshes = [prim for prim in Usd.PrimRange(collision_root) if prim.IsA(UsdGeom.Mesh)]
        proxy_meshes = [prim for prim in collision_meshes if prim.GetName().startswith(("plate_", "hub_"))]
        assert len(proxy_meshes) == 12
        for prim in collision_meshes:
            collision_enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            assert collision_enabled is True
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
            radial_coordinates = torch.tensor([
                (float(point[0]) ** 2 + float(point[1]) ** 2) ** 0.5
                for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
            ])
            assert torch.min(radial_coordinates).item() > 0.005

        source_collision_root = stage.GetPrimAtPath(
            f"/World/envs/env_0/FactoryGear{asset_name}/factory_gear_{asset_name.lower()}/collisions"
        )
        source_collision_meshes = [prim for prim in Usd.PrimRange(source_collision_root) if prim.IsA(UsdGeom.Mesh)]
        assert source_collision_meshes
        assert all(
            not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() for prim in source_collision_meshes
        )
        assert all(not any("SDF" in schema for schema in prim.GetAppliedSchemas()) for prim in collision_meshes)

    gripper_joint_names = tuple(DROID_GRIPPER_MIMIC_SIGNS)
    gripper_joint_indices = [robot.joint_names.index(name) for name in gripper_joint_names]
    mimic_signs = torch.tensor(tuple(DROID_GRIPPER_MIMIC_SIGNS.values()), device=uenv.device)
    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=uenv.device)
    gear_type_cfg = uenv.event_manager.get_term_cfg("randomize_gear_type")
    tabletop_z = MAPLE_TABLE_TOP_COLLISION_POSE.position_xyz[2]
    env_ids = torch.arange(uenv.num_envs, device=uenv.device)
    zero_velocity = torch.zeros((uenv.num_envs, 6), device=uenv.device)

    for gear_type in ("gear_small", "gear_medium", "gear_large"):
        gear_type_cfg.params["gear_types"] = [gear_type]
        uenv.event_manager.set_term_cfg("randomize_gear_type", gear_type_cfg)
        env.reset()

        gear = uenv.scene[f"factory_{gear_type}"]
        arm_position = robot.data.joint_pos.torch[:, :7].clone()
        end_effector_index = robot.body_names.index("base_link")
        grasp_rotation = torch.tensor(uenv.cfg.grasp_rot_offset, device=uenv.device).repeat(uenv.num_envs, 1)
        target_orientation = math_utils.quat_mul(gear.data.root_link_quat_w.torch, grasp_rotation)
        grasp_offset = torch.tensor(uenv.cfg.gear_offsets_grasp[gear_type], device=uenv.device).repeat(uenv.num_envs, 1)
        target_position = gear.data.root_link_pos_w.torch + math_utils.quat_apply(target_orientation, grasp_offset)

        position_error = torch.linalg.norm(robot.data.body_pos_w.torch[:, end_effector_index] - target_position, dim=-1)
        rotation_error = math_utils.quat_error_magnitude(
            robot.data.body_quat_w.torch[:, end_effector_index], target_orientation
        )
        assert (
            torch.max(position_error).item() < 1.0e-3
        ), f"{gear_type} IK position error={position_error}; arm={arm_position}"
        assert torch.max(rotation_error).item() < 1.0e-3, f"{gear_type} IK rotation error={rotation_error}"

        arm_limits = robot.data.joint_pos_limits.torch[:, :7]
        normalized_margin = torch.minimum(
            (arm_position - arm_limits[:, :, 0]) / (arm_limits[:, :, 1] - arm_limits[:, :, 0]),
            (arm_limits[:, :, 1] - arm_position) / (arm_limits[:, :, 1] - arm_limits[:, :, 0]),
        ).min()
        assert normalized_margin.item() > 0.03
        wrist_z = robot.data.body_pos_w.torch[:, robot.body_names.index("panda_link7"), 2]
        assert torch.min(wrist_z).item() > tabletop_z + 0.15

        action[:, :7] = arm_position
        initial_gear_position = gear.data.root_link_pos_w.torch.clone()
        max_gear_displacement = 0.0
        max_arm_drift = 0.0
        max_gripper_speed = 0.0
        max_mimic_error = 0.0
        for command, width in (
            (0.0, uenv.cfg.hand_grasp_width[gear_type]),
            (1.0, uenv.cfg.hand_close_width[gear_type]),
        ):
            action[:, -1] = command
            expected_positions = width * mimic_signs
            position_tolerance = 0.002 if command == 0.0 else 0.02
            for _ in range(90):
                env.step(action)
                positions = robot.data.joint_pos.torch[:, gripper_joint_indices]
                velocities = robot.data.joint_vel.torch[:, gripper_joint_indices]
                assert torch.isfinite(robot.data.joint_pos.torch).all()
                assert torch.isfinite(robot.data.joint_vel.torch).all()
                assert torch.isfinite(gear.data.root_link_pose_w.torch).all()
                assert torch.isfinite(gear.data.root_link_vel_w.torch).all()

                max_gear_displacement = max(
                    max_gear_displacement,
                    torch.linalg.norm(gear.data.root_link_pos_w.torch - initial_gear_position, dim=-1).max().item(),
                )
                max_arm_drift = max(
                    max_arm_drift,
                    torch.abs(robot.data.joint_pos.torch[:, :7] - arm_position).max().item(),
                )
                max_gripper_speed = max(max_gripper_speed, torch.abs(velocities).max().item())
                max_mimic_error = max(
                    max_mimic_error,
                    torch.abs(positions - positions[:, :1] * mimic_signs).max().item(),
                )
                if (
                    torch.abs(positions - expected_positions).max().item() < position_tolerance
                    and torch.abs(velocities).max().item() < 0.01
                ):
                    break

            assert torch.abs(positions - expected_positions).max().item() < position_tolerance
            assert torch.abs(velocities).max().item() < 0.01

        assert max_gear_displacement < 0.002, f"{gear_type}: displacement={max_gear_displacement}"
        assert max_arm_drift < 5.0e-4, f"{gear_type}: arm drift={max_arm_drift}"
        assert max_gripper_speed < 0.75, f"{gear_type}: gripper speed={max_gripper_speed}"
        assert max_mimic_error < 0.002, f"{gear_type}: mimic error={max_mimic_error}"
        assert torch.linalg.norm(gear.data.root_link_vel_w.torch[:, :3], dim=-1).max().item() < 0.02

        base_pose = uenv.scene["factory_gear_base"].data.root_link_pose_w.torch.detach().clone()
        root_offset = torch.tensor(NEWTON_GEAR_OFFSETS[gear_type], device=uenv.device).repeat(uenv.num_envs, 1)
        assembled_pose = base_pose.clone()
        assembled_pose[:, :3] += math_utils.quat_apply(assembled_pose[:, 3:7], root_offset)
        assembled_pose[:, 2] += NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE[gear_type]
        gear.write_root_pose_to_sim_index(root_pose=assembled_pose, env_ids=env_ids)
        gear.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)
        uenv.sim.forward()

        for step in range(GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS):
            success = uenv.termination_manager.compute()
            assert success.tolist() == [step == GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS - 1] * uenv.num_envs

    env.close()
    return True


def test_gear_assembly_newton_gears_settle():
    assert run_simulation_app_function(_test_gear_assembly_newton_gears_settle, headless=True)
