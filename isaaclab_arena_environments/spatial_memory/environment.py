# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration and builder for the registered spatial-memory environments."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic


@dataclass
class SpatialMemoryEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure a spatial-memory environment."""

    observation_variant: Literal["current", "history_k", "oracle_map", "mapped"] = "current"
    """One of ``current``, ``history_k``, ``oracle_map`` or ``mapped``."""

    include_depth: bool = False
    """Include current egocentric depth in the camera observations for a matched sensor baseline."""

    history_k: int | None = None
    """Frames stacked by ``history_k``; None covers 3 seconds of control steps."""

    difficulty: float | None = None
    """Value of the task's difficulty knob; None uses the task default."""

    robot: Literal["default", "droid"] = "default"
    """``droid`` swaps the Franka tasks' robot for DROID (Franka with a Robotiq gripper) to evaluate DROID-trained
    policies; the scripted controllers support only the default robot."""


def make_embodiment_adapter(logic_type: type[SpatialMemoryTaskLogic], robot: str):
    """Return the embodiment adapter for a task: the task's own robot, or DROID when ``robot`` is ``droid``.

    Args:
        logic_type: Task logic class.
        robot: ``default`` or ``droid`` (see ``SpatialMemoryEnvironmentCfg.robot``).

    Returns:
        A ``FrankaAdapter``, ``DroidAdapter`` or ``G1Adapter``.
    """
    from isaaclab_arena_environments.spatial_memory.embodiment_adapter import DroidAdapter, FrankaAdapter
    from isaaclab_arena_environments.spatial_memory.g1_adapter import G1Adapter

    if logic_type.robot == "g1":
        assert robot == "default", f"{logic_type.name} runs on the G1 only."
        return G1Adapter()
    if robot == "droid":
        return DroidAdapter()
    return FrankaAdapter(initial_joint_pose=logic_type.initial_joint_pose, rest_joint_pose=logic_type.rest_joint_pose)


def build_spatial_memory_environment(
    name: str, logic_type: type[SpatialMemoryTaskLogic], cfg: SpatialMemoryEnvironmentCfg
) -> IsaacLabArenaEnvironment:
    """Assemble the Arena environment of a task: primitive parts, ground, light, embodiment and task."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.spatial_memory.metrics import record_spatial_memory_episode
    from isaaclab_arena_environments.spatial_memory.observations import ObservationVariant, apply_observation_variant
    from isaaclab_arena_environments.spatial_memory.parts import make_part_object
    from isaaclab_arena_environments.spatial_memory.task import SpatialMemoryTask

    logic = logic_type(cfg.difficulty, make_embodiment_adapter(logic_type, cfg.robot))
    variant = ObservationVariant(cfg.observation_variant)
    asset_registry = AssetRegistry()
    # Spawn every part at the layout of a fixed sample so that the first frames are physically sensible; the reset
    # event writes the real per-episode layout.
    T_E_parts = logic.sample_layout(torch.Generator().manual_seed(0)).part_poses_e
    parts = [
        make_part_object(spec, tuple(T_E_parts[index, :3].tolist())) for index, spec in enumerate(logic.part_specs())
    ]
    floor_z = logic.embodiment.floor_z_m
    ground = asset_registry.get_asset_by_name("ground_plane")(initial_pose=Pose(position_xyz=(0.0, 0.0, floor_z)))
    light = asset_registry.get_asset_by_name("light")()
    embodiment = logic.embodiment.make_embodiment(enable_cameras=cfg.enable_cameras)
    assert not cfg.include_depth or cfg.enable_cameras, "Depth observations require cameras."
    uses_depth = cfg.enable_cameras and (cfg.include_depth or variant == ObservationVariant.MAPPED)
    if uses_depth:
        camera_name = logic.embodiment.camera_obs_key().removesuffix("_rgb")
        camera_cfg = getattr(embodiment.camera_config, camera_name)
        if "distance_to_image_plane" not in camera_cfg.data_types:
            camera_cfg.data_types = [*camera_cfg.data_types, "distance_to_image_plane"]
        camera_cfg.update_latest_camera_pose = True
    embodiment.set_initial_pose(Pose.identity())

    def env_cfg_callback(env_cfg):
        env_cfg = logic.embodiment.configure_env_cfg(env_cfg)
        env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, logic.minimum_env_spacing_m)
        if uses_depth:
            env_cfg.num_rerenders_on_reset = max(1, env_cfg.num_rerenders_on_reset)
        return apply_observation_variant(env_cfg, logic, variant, cfg.history_k)

    return IsaacLabArenaEnvironment(
        name=name,
        embodiment=embodiment,
        scene=Scene(assets=[ground, light, *parts]),
        task=SpatialMemoryTask(logic, variant),
        env_cfg_callback=env_cfg_callback,
        episode_recorder_terms={"spatial_memory": EpisodeRecorderTermCfg(func=record_spatial_memory_episode)},
    )
