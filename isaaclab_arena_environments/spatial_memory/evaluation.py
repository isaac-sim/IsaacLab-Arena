# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers to build a spatial-memory environment and roll out a policy for a fixed number of episodes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

ENV_SPACING_M = 5.0
"""Distance between environment origins; the longest scene, the box-carry route, spans about 4.4 m."""


def make_spatial_memory_env(
    env_name: str,
    num_envs: int,
    seed: int = 0,
    difficulty: float | None = None,
    observation_variant: str = "current",
    enable_cameras: bool = False,
    history_k: int | None = None,
    with_segmentation: bool = False,
    with_depth: bool = False,
):
    """Build and return the gym-wrapped environment registered under ``env_name``.

    With ``with_segmentation``, the egocentric camera also renders instance segmentation and keeps its pose current,
    for checking the analytic visibility model against rendered images. With ``with_depth``, the camera also
    renders optical-axis depth and reports its current pose for reconstruction.
    """
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(env_name)
    env_cfg = registry.get_environment_cfg_types()[env_name](
        enable_cameras=enable_cameras,
        include_depth=with_depth,
        observation_variant=observation_variant,
        difficulty=difficulty,
        history_k=history_k,
    )
    arena_env = factory_type().build(env_cfg)
    if with_segmentation:
        assert enable_cameras, "Segmentation requires enabled cameras."
        camera_rig = arena_env.embodiment.camera_config
        for camera_name in camera_rig.camera_names():
            camera_cfg = getattr(camera_rig, camera_name)
            camera_cfg.data_types = list(dict.fromkeys([*camera_cfg.data_types, "instance_id_segmentation_fast"]))
            camera_cfg.colorize_instance_id_segmentation = False
            camera_cfg.update_latest_camera_pose = True
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=num_envs, seed=seed, env_spacing=ENV_SPACING_M))
    return builder.make_registered()


def rollout_episodes(env, policy, num_episodes: int) -> list[dict[str, Any]]:
    """Run ``policy`` with Arena's rollout loop until ``num_episodes`` episodes finished and return their records.

    Records come from the environment's episode recorder (one JSON object per finished episode).
    """
    from isaaclab_arena.evaluation.rollout import rollout_policy

    output_path = Path(tempfile.mkdtemp(prefix="spatial_memory_")) / "episodes.jsonl"
    env.unwrapped.episode_recorder.set_output_path(output_path)
    rollout_policy(env, policy, num_steps=None, num_episodes=num_episodes)
    return [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]


def summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Return success, collision and searched-again rates and the mean time to success over episode records."""
    num_records = max(len(records), 1)
    successes = [record for record in records if record["success"]]
    times = [record["episode_length"] for record in successes]
    return {
        "episodes": len(records),
        "success_rate": len(successes) / num_records,
        "collision_rate": sum(record["spatial_memory"]["collided"] for record in records) / num_records,
        "searched_again_rate": sum(record["spatial_memory"]["searched_again"] for record in records) / num_records,
        "mean_time_to_success_steps": (sum(times) / len(times)) if times else float("nan"),
    }
