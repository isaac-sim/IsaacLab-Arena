# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Config-level coverage for the Franka soft-lift evaluation scene."""

from __future__ import annotations

import numpy as np
from dataclasses import fields

import pytest


def _cfg_field_names(cfg, value_type: type | tuple[type, ...] | None = None) -> list[str]:
    if value_type is None:
        return [field.name for field in fields(cfg)]
    return [field.name for field in fields(cfg) if isinstance(getattr(cfg, field.name), value_type)]


def test_franka_soft_lift_registered_components() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry
    from isaaclab_arena_environments.cli import ensure_environments_registered

    asset_registry = AssetRegistry()
    for asset_name in ("franka_soft_lift_panda", "franka_soft_lift_block", "franka_soft_lift_table"):
        assert asset_registry.is_registered(asset_name)

    ensure_environments_registered()
    environment_registry = EnvironmentRegistry()
    assert environment_registry.is_registered("franka_soft_lift")
    assert environment_registry.get_component_by_name("franka_soft_lift").name == "franka_soft_lift"
    assert TaskRegistry().is_registered("FrankaSoftLiftTask")


def test_franka_soft_lift_scene_assets_match_source() -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_arena.assets.deformable_spawn import SimulationBackend
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_environments.franka_soft_lift_environment import (
        FrankaSoftLiftEnvironment,
        FrankaSoftLiftEnvironmentCfg,
    )

    arena_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    assert list(arena_env.scene.assets) == ["table", "deformable", "ground", "sky_light"]

    table = arena_env.scene.assets["table"]
    deformable = arena_env.scene.assets["deformable"]
    ground = arena_env.scene.assets["ground"]
    sky_light = arena_env.scene.assets["sky_light"]

    assert table.object_type == ObjectType.BASE
    assert table.prim_path == "{ENV_REGEX_NS}/Table"
    assert table.object_cfg.init_state.pos == (0.5, 0.0, 0.0)
    assert table.object_cfg.init_state.rot == (0.0, 0.0, 0.707, 0.707)
    assert isinstance(table.object_cfg.spawn, UsdFileCfg)
    assert table.object_cfg.spawn.usd_path.endswith("/Props/Mounts/SeattleLabTable/table_instanceable.usd")

    assert deformable.prim_path == "{ENV_REGEX_NS}/Deformable"
    assert deformable.get_event_cfg()[1] is None
    assert deformable.soft_body_kinds() == frozenset({"volume"})
    bbox = deformable.get_bounding_box()
    assert bbox.min_point[0].tolist() == pytest.approx([-0.15, -0.025, -0.025])
    assert bbox.max_point[0].tolist() == pytest.approx([0.15, 0.025, 0.025])

    for preset in ("physx", "newton_mjwarp_vbd_proxy", "newton_mjwarp_vbd"):
        block_cfg = resolve_presets(deformable.object_cfg, selected=(preset,))
        assert isinstance(block_cfg, DeformableObjectCfg)
        assert isinstance(block_cfg.spawn, UsdFileCfg)
        assert block_cfg.prim_path == "{ENV_REGEX_NS}/Deformable"
        assert block_cfg.init_state.pos == (0.5, 0.0, 0.05)
        assert block_cfg.spawn.usd_path.endswith("/franka_soft_lift_block_tet.usda")
        assert block_cfg.spawn.visual_material.diffuse_color == (0.95, 0.85, 0.1)
        assert block_cfg.spawn.physics_material.density == 300.0

    physx_cfg = deformable._make_deformable_cfg(SimulationBackend.PHYSX).spawn
    assert physx_cfg.deformable_props.rest_offset is None
    assert physx_cfg.deformable_props.contact_offset is None
    assert physx_cfg.deformable_props.solver_position_iteration_count == 16
    assert physx_cfg.deformable_props.linear_damping is None
    assert physx_cfg.physics_material.youngs_modulus == 8.0e4
    assert physx_cfg.physics_material.poissons_ratio == 0.25
    assert physx_cfg.physics_material.static_friction == 10.0
    assert physx_cfg.physics_material.dynamic_friction == 5.0

    newton_cfg = deformable._make_deformable_cfg(SimulationBackend.NEWTON).spawn
    assert newton_cfg.physics_material.particle_radius == 0.01

    assert ground.prim_path == "/World/GroundPlane"
    assert ground.object_cfg.init_state.pos == (0.0, 0.0, -1.05)
    assert isinstance(ground.object_cfg.spawn, GroundPlaneCfg)

    assert isinstance(sky_light.object_cfg, AssetBaseCfg)
    assert sky_light.prim_path == "/World/skyLight"
    assert isinstance(sky_light.object_cfg.spawn, sim_utils.DomeLightCfg)
    assert sky_light.object_cfg.spawn.intensity == 750.0
    assert sky_light.object_cfg.spawn.texture_file.endswith(
        "/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
    )
    assert AssetRegistry().get_asset_by_name("franka_soft_lift_block").name == "franka_soft_lift_block"


def test_franka_soft_lift_embodiment_matches_source() -> None:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaEmbodimentBase

    embodiment = AssetRegistry().get_asset_by_name("franka_soft_lift_panda")()

    assert not isinstance(embodiment, FrankaEmbodimentBase)
    assert embodiment.observation_config is None
    assert embodiment.event_config is None
    assert embodiment.reward_config is None

    robot = embodiment.scene_config.robot
    assert robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert robot.spawn.usd_path.endswith("/Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    assert robot.spawn.rigid_props.disable_gravity is True
    assert robot.actuators["panda_hand"].effort_limit_sim == 500.0
    assert robot.actuators["panda_hand"].stiffness == 1000.0
    assert robot.actuators["panda_hand"].damping == 100.0

    ee_frame = embodiment.scene_config.ee_frame
    assert ee_frame.prim_path == "{ENV_REGEX_NS}/Robot/panda_link0"
    assert len(ee_frame.target_frames) == 1
    assert ee_frame.target_frames[0].prim_path == "{ENV_REGEX_NS}/Robot/panda_hand"
    assert ee_frame.target_frames[0].offset.pos == [0.0, 0.0, 0.1034]

    actions = embodiment.action_config
    assert _cfg_field_names(actions) == ["arm_action", "gripper_action"]
    assert actions.arm_action.controller.command_type == "pose"
    assert actions.arm_action.controller.use_relative_mode is False
    assert actions.arm_action.controller.ik_method == "dls"
    assert actions.arm_action.controller.ik_params == {"lambda_val": 0.6}
    assert actions.arm_action.body_offset.pos == [0.0, 0.0, 0.107]
    assert actions.gripper_action.open_command_expr == {"panda_finger_.*": 0.05}
    assert actions.gripper_action.close_command_expr == {"panda_finger_.*": 0.0}


def test_franka_soft_lift_task_mdp_matches_source() -> None:
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab_tasks.manager_based.manipulation.lift_franka_soft import mdp as soft_lift_mdp
    from isaaclab_tasks.manager_based.manipulation.lift_franka_soft.mdp.observations import (
        DeformableSampledPointsInRobotRootFrame,
    )

    from isaaclab_arena_environments.franka_soft_lift_environment import (
        FrankaSoftLiftEnvironment,
        FrankaSoftLiftEnvironmentCfg,
    )

    task = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg()).task

    command = task.get_commands_cfg().deformable_pose
    assert command.asset_name == "robot"
    assert command.body_name == "panda_hand"
    assert command.resampling_time_range == (5.0, 5.0)
    assert command.debug_vis is True
    assert command.ranges.pos_x == (0.4, 0.6)
    assert command.ranges.pos_y == (-0.25, 0.25)
    assert command.ranges.pos_z == (0.25, 0.5)
    assert command.goal_pose_visualizer_cfg.markers["sphere"].visual_material.diffuse_color == (0.1, 0.9, 0.2)

    policy = task.get_observation_cfg().policy
    assert _cfg_field_names(policy, ObsTerm) == [
        "joint_pos",
        "joint_vel",
        "deformable_sampled_points",
        "target_position",
        "actions",
    ]
    assert policy.enable_corruption is True
    assert policy.concatenate_terms is True
    assert policy.joint_pos.func is mdp.joint_pos_rel
    assert policy.joint_vel.func is mdp.joint_vel_rel
    assert policy.deformable_sampled_points.func is DeformableSampledPointsInRobotRootFrame
    assert policy.deformable_sampled_points.params["num_points"] == 20
    assert policy.target_position.params == {"command_name": "deformable_pose"}
    assert policy.actions.func is mdp.last_action

    rewards = task.get_rewards_cfg()
    assert _cfg_field_names(rewards) == [
        "reaching_deformable",
        "lifting_deformable",
        "deformable_goal_tracking",
        "deformable_goal_tracking_fine_grained",
        "action_rate",
        "gripper_close",
        "joint_vel",
        "joint_torque",
        "joint_acc",
    ]
    assert rewards.reaching_deformable.func is soft_lift_mdp.deformable_ee_distance
    assert rewards.reaching_deformable.weight == 5.0
    assert rewards.lifting_deformable.params["minimal_height"] == 0.04
    assert rewards.deformable_goal_tracking.weight == 16.0
    assert rewards.deformable_goal_tracking.params["minimal_height"] == 0.075
    assert rewards.deformable_goal_tracking_fine_grained.params["std"] == 0.05
    assert rewards.action_rate.weight == -1.0e-2
    assert rewards.gripper_close.func is soft_lift_mdp.gripper_close_action
    assert rewards.joint_torque.weight == -1.0e-4
    assert rewards.joint_acc.weight == -1.0e-4

    terminations = task.get_termination_cfg()
    assert _cfg_field_names(terminations) == [
        "time_out",
        "deformable_outside_table",
        "deformable_dropped",
        "ee_below_table",
    ]
    assert not hasattr(terminations, "success")
    assert terminations.deformable_outside_table.params["x_bounds"] == (0.0, 1.0)
    assert terminations.deformable_outside_table.params["y_bounds"] == (-0.5, 0.5)
    assert terminations.deformable_dropped.params["minimum_height"] == -0.1
    assert terminations.ee_below_table.params["minimum_height"] == 0.0


def test_franka_soft_lift_default_preset_and_explicit_overrides() -> None:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.physics_presets import ARENA_PHYSICS_PRESETS
    from isaaclab_arena_environments.franka_soft_lift_environment import (
        FrankaSoftLiftEnvironment,
        FrankaSoftLiftEnvironmentCfg,
    )

    arena_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    assert arena_env.default_physics_preset == "newton_mjwarp_vbd_proxy"
    assert arena_env.rl_framework_entry_point is None
    assert arena_env.rl_policy_cfg is None

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False))
    assert builder._select_backend_preset(None, needs_soft_body=True) == "newton_mjwarp_vbd_proxy"
    env_cfg, _ = builder.compose_manager_cfg()
    assert env_cfg.scene.replicate_physics is True
    assert env_cfg.sim.physics == ARENA_PHYSICS_PRESETS["newton_mjwarp_vbd_proxy"].cfg
    assert env_cfg.sim.physics.solver_cfg.rigid_solver_cfg.njmax == 40
    assert env_cfg.sim.physics.solver_cfg.rigid_solver_cfg.nconmax == 20
    assert env_cfg.sim.gravity == (0.0, 0.0, -9.81)
    assert env_cfg.sim.dt == 1.0 / 60.0
    assert env_cfg.decimation == 1
    assert env_cfg.episode_length_s == 5.0
    assert env_cfg.sync_deformable_visual_meshes_from_sim is True

    physx_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    physx_cfg, _ = ArenaEnvBuilder(
        physx_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets="physx"),
    ).compose_manager_cfg()
    assert physx_cfg.scene.replicate_physics is False
    assert physx_cfg.sim.physics == ARENA_PHYSICS_PRESETS["physx"].cfg

    vbd_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    vbd_cfg, _ = ArenaEnvBuilder(
        vbd_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets="newton_mjwarp_vbd"),
    ).compose_manager_cfg()
    assert vbd_cfg.scene.replicate_physics is True
    assert vbd_cfg.sim.physics == ARENA_PHYSICS_PRESETS["newton_mjwarp_vbd"].cfg

    surface_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    with pytest.raises(NotImplementedError, match="does not support"):
        ArenaEnvBuilder(
            surface_env,
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets="newton_mjwarp_vbd_surface"),
        ).compose_manager_cfg()


def test_franka_soft_lift_variation_defaults_hydra_and_catalogue() -> None:
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.franka_soft_lift_environment import (
        FrankaSoftLiftEnvironment,
        FrankaSoftLiftEnvironmentCfg,
    )

    arena_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    variation = arena_env.scene.assets["deformable"].get_variation("initial_pose")
    assert variation.enabled is True
    assert variation.cfg.sampler_cfg.low == [-0.05, -0.05, 0.0]
    assert variation.cfg.sampler_cfg.high == [0.05, 0.05, 0.0]

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False))
    catalogue = builder.get_variations_catalogue_as_string()
    assert "Asset: deformable" in catalogue
    assert "deformable.initial_pose.enabled=true  (default: True)" in catalogue
    assert "deformable.initial_pose.sampler_cfg.low = [-0.05,-0.05,0.0]" in catalogue

    disabled_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg(pose_variation_enabled=False))
    disabled_variation = disabled_env.scene.assets["deformable"].get_variation("initial_pose")
    assert disabled_variation.enabled is False
    disabled_cfg, _ = ArenaEnvBuilder(
        disabled_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
    ).compose_manager_cfg()
    assert not hasattr(disabled_cfg.events, "deformable_initial_pose_variation")

    override_env = FrankaSoftLiftEnvironment().build(FrankaSoftLiftEnvironmentCfg())
    override_cfg, _ = ArenaEnvBuilder(
        override_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
        hydra_overrides=[
            "deformable.initial_pose.sampler_cfg.low=[-0.01,-0.02,0.0]",
            "deformable.initial_pose.sampler_cfg.high=[0.01,0.02,0.0]",
        ],
    ).compose_manager_cfg()
    event = override_cfg.events.deformable_initial_pose_variation
    sampler = event.params["sampler"]
    assert sampler.low.tolist() == pytest.approx([-0.01, -0.02, 0.0])
    assert sampler.high.tolist() == pytest.approx([0.01, 0.02, 0.0])


def test_franka_soft_lift_metric_config_and_compute() -> None:
    from isaaclab_arena.metrics.deformable_goal_reached_rate import (
        DeformableGoalReachedRateMetric,
        compute_deformable_goal_reached_rate,
    )

    metric = DeformableGoalReachedRateMetric()
    recorder_cfg = metric.get_recorder_term_cfg()
    metric_cfg = metric.get_metric_term_cfg()

    assert metric.name == "deformable_goal_reached_rate"
    assert recorder_cfg.name == "deformable_goal_reached"
    assert recorder_cfg.command_name == "deformable_pose"
    assert recorder_cfg.minimal_height == 0.075
    assert recorder_cfg.position_tolerance == 0.05
    assert metric_cfg.recorder_term_name == "deformable_goal_reached"

    assert compute_deformable_goal_reached_rate([]) == 0.0
    assert compute_deformable_goal_reached_rate([np.array([], dtype=bool)]) == 0.0
    assert compute_deformable_goal_reached_rate([np.array([True, False]), np.array([[True]])]) == pytest.approx(2 / 3)
