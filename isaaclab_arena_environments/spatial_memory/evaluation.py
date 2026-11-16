# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers to build a spatial-memory environment and roll out a scripted controller for a fixed number of episodes."""

from __future__ import annotations

import json
import math
import tempfile
import torch
from pathlib import Path
from typing import Any


def make_spatial_memory_env(
    env_name: str,
    num_envs: int,
    seed: int = 0,
    difficulty: float | None = None,
    observation_variant: str = "current",
    enable_cameras: bool = False,
    history_k: int | None = None,
    with_segmentation: bool = False,
):
    """Build and return the gym-wrapped environment registered under ``env_name``.

    With ``with_segmentation``, the egocentric camera also renders instance segmentation and keeps its pose current,
    for checking the analytic visibility model against rendered images.
    """
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.cli import ensure_environments_registered

    ensure_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name(env_name)
    cfg_type = factory_type._legacy_argparse_cfg_type
    env_cfg = cfg_type(
        enable_cameras=enable_cameras,
        observation_variant=observation_variant,
        difficulty=difficulty,
        history_k=history_k,
    )
    arena_env = factory_type().build(env_cfg)
    if with_segmentation:
        assert enable_cameras, "Segmentation needs the cameras to be enabled."
        camera_rig = arena_env.embodiment.camera_config
        for camera_name in camera_rig.camera_names():
            camera_cfg = getattr(camera_rig, camera_name)
            camera_cfg.data_types = ["rgb", "instance_id_segmentation_fast"]
            camera_cfg.colorize_instance_id_segmentation = False
            camera_cfg.update_latest_camera_pose = True
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=num_envs, seed=seed, env_spacing=5.0))
    return builder.make_registered()


def rollout_episodes(env, policy, num_episodes: int, max_steps: int | None = None) -> list[dict[str, Any]]:
    """Run ``policy`` until ``num_episodes`` episodes finished and return their per-episode records.

    Records come from the environment's episode recorder (one JSON object per finished episode).
    """
    base_env = env.unwrapped
    output_path = Path(tempfile.mkdtemp(prefix="spatial_memory_")) / "episodes.jsonl"
    base_env.episode_recorder.set_output_path(output_path)
    base_env.configure_episode_limit(num_episodes)
    if max_steps is None:
        max_steps = int(math.ceil(base_env.max_episode_length * math.ceil(num_episodes / base_env.num_envs))) + 5
    obs, _ = env.reset()
    policy.reset()
    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(actions)
            if base_env.reset_env_ids.numel() > 0:
                policy.reset(env_ids=base_env.reset_env_ids)
            if base_env.completed_episode_count >= num_episodes:
                break
    records = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, float]:
    """Return success, collision and searched-again rates and the mean time to success over episode records."""
    num = max(len(records), 1)
    successes = [record for record in records if record.get("success")]
    times = [record["episode_length"] for record in successes]
    return {
        "episodes": len(records),
        "success_rate": len(successes) / num,
        "collision_rate": sum(record["spatial_memory"]["collided"] for record in records) / num,
        "searched_again_rate": sum(record["spatial_memory"]["searched_again"] for record in records) / num,
        "mean_time_to_success_steps": (sum(times) / len(times)) if times else float("nan"),
    }
