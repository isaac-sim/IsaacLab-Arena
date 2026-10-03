# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Arena-native DisplayPort insertion simulation setup."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_displayport_usd_assets(simulation_app) -> bool:
    from pathlib import Path

    from pxr import PhysxSchema, Usd, UsdPhysics

    from isaaclab_arena.assets.registries import AssetRegistry

    def _schema_counts(usd_path: str) -> dict[str, object]:
        stage = Usd.Stage.Open(usd_path)
        assert stage is not None, f"Failed to open {usd_path}"
        default_prim = stage.GetDefaultPrim()
        rigid_body_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        collision_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
        physics_scene_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        sdf_prims = [prim for prim in collision_prims if prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)]
        enabled_sdf_prims = [
            prim for prim in sdf_prims if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        ]
        return {
            "default_path": str(default_prim.GetPath()),
            "default_has_rigid_body": default_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "rigid_body_count": len(rigid_body_prims),
            "collision_count": len(collision_prims),
            "physics_scene_count": len(physics_scene_prims),
            "enabled_sdf_count": len(enabled_sdf_prims),
            "sdf_resolutions": [
                PhysxSchema.PhysxSDFMeshCollisionAPI(prim).GetSdfResolutionAttr().Get() for prim in enabled_sdf_prims
            ],
        }

    registry = AssetRegistry()
    plug = registry.get_asset_by_name("displayport_plug")()
    socket = registry.get_asset_by_name("displayport_socket")()

    for asset in (plug, socket):
        usd_path = Path(asset.usd_path)
        assert usd_path.exists(), f"Missing committed USD asset: {usd_path}"
        assert not usd_path.read_bytes().startswith(b"version https://git-lfs"), f"{usd_path} is an LFS pointer"

    plug_counts = _schema_counts(plug.usd_path)
    assert plug_counts["default_path"] == "/plug"
    assert plug_counts["default_has_rigid_body"] is True
    assert plug_counts["rigid_body_count"] == 1
    assert plug_counts["collision_count"] == 5
    assert plug_counts["physics_scene_count"] == 0
    assert plug_counts["enabled_sdf_count"] == 1
    assert plug_counts["sdf_resolutions"] == [256]

    socket_counts = _schema_counts(socket.usd_path)
    assert socket_counts["default_path"] == "/socket"
    assert socket_counts["default_has_rigid_body"] is True
    assert socket_counts["rigid_body_count"] == 1
    assert socket_counts["collision_count"] == 6
    assert socket_counts["physics_scene_count"] == 0
    assert socket_counts["enabled_sdf_count"] == 5
    return True


def _test_displayport_insertion_env_composes(simulation_app) -> bool:
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.tasks.displayport_insertion_task import (
        DisplayPortInsertionSuccessRateMetric,
        ResetDisplayPortPlugCurriculum,
        ResetSampledConstantNoiseModelCfg,
    )
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.displayport_insertion_sim_environment import DisplayPortInsertionSimEnvironmentCfg

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name("displayport_insertion_sim")

    default_arena_env = factory_type().build(DisplayPortInsertionSimEnvironmentCfg())
    default_env_cfg, _ = ArenaEnvBuilder(
        default_arena_env,
        ArenaEnvBuilderCfg(num_envs=2, solve_relations=False),
    ).compose_manager_cfg()
    assert default_env_cfg.scene.env_spacing == 2.5
    assert default_env_cfg.rewards.plug_socket_keypoint_tracking.weight == -1.5
    assert default_env_cfg.rewards.plug_socket_keypoint_tracking_exp.weight == 1.5
    assert default_env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == 0.8
    assert default_env_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] == 0.0
    assert default_env_cfg.events.reset_plug_curriculum.params["anneal_end_iter"] == 500.0
    assert isinstance(default_env_cfg.observations.policy.socket_pos.noise, ResetSampledConstantNoiseModelCfg)
    assert default_env_cfg.observations.policy.enable_corruption is True
    assert not hasattr(default_env_cfg.observations.policy, "plug_pos")
    assert hasattr(default_env_cfg.observations, "critic")
    assert hasattr(default_env_cfg.observations.critic, "plug_pos")
    try:
        ArenaEnvBuilder(
            default_arena_env,
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, presets="physx"),
        ).compose_manager_cfg()
    except NotImplementedError as exc:
        assert "task-specific physics settings" in str(exc)
    else:
        raise AssertionError("DisplayPort should reject builder physics presets that override tuned PhysX settings")

    arena_env = factory_type().build(
        DisplayPortInsertionSimEnvironmentCfg(
            curriculum_mode="anneal_80_20_500",
            socket_pos_range=[0.02, 0.03, 0.04],
            socket_orn_deg=4.0,
            exp_reward_weight=3.0,
        )
    )

    assert arena_env.embodiment is None
    assert set(arena_env.scene.assets) == {"ground_plane", "light", "displayport_plug", "displayport_socket"}
    assert arena_env.scene.assets["displayport_plug"].object_cfg.spawn.activate_contact_sensors is True
    assert arena_env.scene.assets["displayport_socket"].object_cfg.spawn.activate_contact_sensors is False
    assert [variation.name for variation in arena_env.scene.assets["light"].get_variations()] == [
        "hdr_image",
        "intensity",
        "color",
        "color_temperature",
    ]
    assert any(variation.name == "mass" for variation in arena_env.scene.assets["displayport_plug"].get_variations())

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False))
    env_cfg, env_kwargs = builder.compose_manager_cfg()

    assert "variation_recorder" in env_kwargs
    assert env_cfg.scene.replicate_physics is True
    assert env_cfg.scene.env_spacing == 2.5
    assert env_cfg.decimation == 8
    assert env_cfg.sim.render_interval == 8
    assert env_cfg.sim.dt == 1.0 / 240.0
    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.sim.physics.gpu_collision_stack_size == 2**30
    assert env_cfg.scene.displayport_plug.spawn.usd_path.endswith("display_port_plug_fixed_sdf.usd")
    assert env_cfg.scene.displayport_socket.spawn.usd_path.endswith("display_port_socket_fixed_sdf_noprotrusions.usd")
    assert env_cfg.scene.displayport_plug.spawn.rigid_props.max_depenetration_velocity == 0.5
    assert env_cfg.scene.displayport_socket.spawn.rigid_props.kinematic_enabled is True
    assert env_cfg.rewards.plug_socket_keypoint_tracking.weight == -1.5
    assert env_cfg.rewards.plug_socket_keypoint_tracking_exp.weight == 3.0
    assert env_cfg.events.randomize_socket_pose.params["pose_range"]["x"] == (-0.02, 0.02)
    assert env_cfg.events.randomize_socket_pose.params["pose_range"]["y"] == (-0.03, 0.03)
    assert env_cfg.events.randomize_socket_pose.params["pose_range"]["z"] == (-0.04, 0.04)
    assert env_cfg.events.reset_plug_curriculum.func is ResetDisplayPortPlugCurriculum
    assert env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] == 0.8
    assert env_cfg.events.reset_plug_curriculum.params["at_goal_prob_final"] == 0.2
    assert env_cfg.events.reset_plug_curriculum.params["anneal_end_iter"] == 500.0
    assert isinstance(env_cfg.metrics.displayport_insertion_success_rate, MetricTermCfg)
    assert (
        env_cfg.metrics.displayport_insertion_success_rate.recorder_term_name
        == DisplayPortInsertionSuccessRateMetric.recorder_term_name
    )
    assert isinstance(env_cfg.episode_recorders.displayport_insertion, EpisodeRecorderTermCfg)
    assert env_cfg.episode_recorders.displayport_insertion.params["plug_name"] == "displayport_plug"
    assert env_cfg.recorders.displayport_insertion_success_rate.name == "displayport_insertion_success"
    assert env_cfg.recorders.displayport_insertion_success_rate.plug_name == "displayport_plug"
    return True


def _test_displayport_passive_profile_composes(simulation_app) -> bool:
    from isaaclab_arena.assets import displayport_insertion_geometry as geometry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.displayport_insertion_sim_environment import (
        DisplayPortInsertionSimEnvironment,
        DisplayPortInsertionSimEnvironmentCfg,
    )

    arena_env = DisplayPortInsertionSimEnvironment().build(
        DisplayPortInsertionSimEnvironmentCfg(profile="passive_drop_test")
    )
    env_cfg, _ = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
    ).compose_manager_cfg()

    assert env_cfg.decimation == 1
    assert env_cfg.sim.render_interval == 1
    assert env_cfg.episode_length_s == 10.0
    assert env_cfg.scene.env_spacing == 2.5
    assert not hasattr(env_cfg.rewards, "plug_socket_keypoint_tracking")
    assert not hasattr(env_cfg.observations, "critic")
    assert not hasattr(env_cfg.events, "reset_plug_curriculum")
    assert env_cfg.scene.ground_plane.init_state.pos == (0.0, 0.0, 0.0)
    assert env_cfg.scene.displayport_socket.init_state.pos == geometry.PASSIVE_DROP_SOCKET_POS
    assert env_cfg.scene.displayport_socket.init_state.rot == geometry.PASSIVE_DROP_SOCKET_ROT
    assert env_cfg.scene.displayport_plug.init_state.pos == geometry.PASSIVE_DROP_PLUG_POS
    assert env_cfg.scene.displayport_plug.init_state.rot == geometry.PASSIVE_DROP_PLUG_ROT
    return True


def _test_displayport_runtime_reset_and_metrics(simulation_app) -> bool:
    import torch

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tasks.displayport_insertion_task import displayport_mate_pos_error
    from isaaclab_arena_environments.displayport_insertion_sim_environment import (
        DisplayPortInsertionSimEnvironment,
        DisplayPortInsertionSimEnvironmentCfg,
    )

    arena_env = DisplayPortInsertionSimEnvironment().build(
        DisplayPortInsertionSimEnvironmentCfg(curriculum_mode="fixed80", episode_length_s=0.05)
    )
    env = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=2, solve_relations=False),
    ).make_registered()
    try:
        obs, _ = env.reset()
        assert "policy" in obs
        assert "critic" in obs
        assert obs["policy"].shape[-1] == 7
        assert obs["critic"].shape[-1] == 14

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(4):
            obs, reward, _, _, _ = env.step(actions)
        assert reward.shape == (2,)

        mate_error = displayport_mate_pos_error(
            env.unwrapped,
            SceneEntityCfg("displayport_socket"),
            SceneEntityCfg("displayport_plug"),
        )
        assert mate_error.shape == (2,)

        metrics = env.unwrapped.compute_metrics()
        assert "displayport_insertion_success_rate" in metrics.metric_data_entries
        assert metrics.num_episodes >= 2
        assert metrics.metric_data_entries["displayport_insertion_success_rate"].recorded_data
    finally:
        env.close()
    return True


def test_displayport_usd_assets():
    assert run_simulation_app_function(_test_displayport_usd_assets, headless=HEADLESS)


def test_displayport_insertion_env_composes():
    assert run_simulation_app_function(_test_displayport_insertion_env_composes, headless=HEADLESS)


def test_displayport_passive_profile_composes():
    assert run_simulation_app_function(_test_displayport_passive_profile_composes, headless=HEADLESS)


def test_displayport_runtime_reset_and_metrics():
    assert run_simulation_app_function(_test_displayport_runtime_reset_and_metrics, headless=HEADLESS)
