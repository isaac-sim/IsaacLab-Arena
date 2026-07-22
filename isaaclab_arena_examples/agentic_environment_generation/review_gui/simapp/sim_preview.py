# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation-solver rollout preview for the review GUI SimApp server."""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    collect_garbage_and_clear_cuda_cache,
    teardown_simulation_app,
)
from isaaclab_arena.video.camera_observation_video_recorder import parse_episode_video_filename
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    CAPTURE_DONE_TAIL_UPDATES,
    pump_app,
    sim_preview_cache_dir,
)

# Placement pool size when preview uses resolve_on_reset=False (see ObjectPlacerParams).
_PREVIEW_LAYOUTS_PER_ENV = 2
_ENV_SPACING_BUFFER_M = 0.5


def parse_sim_preview_params(req: dict[str, Any]) -> tuple[int, int]:
    """Read required sim-preview rollout settings from a JSON-RPC request."""
    missing = [key for key in ("num_envs", "num_steps") if key not in req]
    if missing:
        raise ValueError(f"missing required sim preview params: {', '.join(missing)}")
    num_envs = int(req["num_envs"])
    num_steps = int(req["num_steps"])
    assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
    assert num_steps >= 1, f"num_steps must be >= 1, got {num_steps}"
    return num_envs, num_steps


def _preview_log(started_at: float, message: str) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[sim_preview] +{elapsed:.1f}s {message}", file=sys.stderr, flush=True)


def _preview_cfg(*, num_envs: int, env_spacing: float) -> ArenaEnvBuilderCfg:
    return ArenaEnvBuilderCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        resolve_on_reset=False,
    )


def _compute_env_spacing(arena_env) -> float:
    """Return the largest background XY dimension plus a safety buffer."""
    from isaaclab_arena.assets.background import Background

    backgrounds = [asset for asset in arena_env.scene.assets.values() if isinstance(asset, Background)]
    assert backgrounds, "Sim preview requires a background asset to compute environment spacing"
    max_dimension_m = max(float(background.get_bounding_box().size[0, :2].max()) for background in backgrounds)
    return max_dimension_m + _ENV_SPACING_BUFFER_M


def _collect_recorded_videos(video_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Return the viewport video and parsed per-env camera videos."""
    viewport_videos: list[Path] = []
    camera_videos: list[dict[str, Any]] = []
    for path in sorted(video_dir.glob("*.mp4")):
        parsed = parse_episode_video_filename(path.name)
        if parsed is None:
            viewport_videos.append(path)
            continue
        camera_videos.append({
            "path": str(path),
            "env_id": parsed.env_index,
            "camera_name": parsed.camera_name,
        })

    assert viewport_videos, f"Viewport recorder produced no mp4 in {video_dir}"
    return viewport_videos[0], camera_videos


def _close_env_and_reset_sim(
    env=None,
    *,
    suppress_exceptions: bool = False,
    app=None,
) -> None:
    """Close env and reset sim for the next preview."""
    error_manager = suppress(Exception) if suppress_exceptions else nullcontext()

    with error_manager:
        if env is not None and not getattr(env.unwrapped, "_is_closed", True):
            env.close()

    teardown_simulation_app(suppress_exceptions=suppress_exceptions, make_new_stage=True)

    if app is not None:
        with error_manager:
            pump_app(app, count=CAPTURE_DONE_TAIL_UPDATES)

    collect_garbage_and_clear_cuda_cache()


def run_sim_preview(
    app,
    yaml_text: str,
    *,
    num_envs: int,
    num_steps: int,
) -> dict[str, Any]:
    """Run a zero-action rollout and record viewport and embodiment-camera videos."""
    import gymnasium as gym
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.evaluation.policy_runner import rollout_policy
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

    started_at = time.monotonic()
    _preview_log(started_at, "run_sim_preview started")

    _close_env_and_reset_sim(suppress_exceptions=True, app=app)
    _preview_log(started_at, "cleared stale sim state")

    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping, got {type(raw).__name__}")

    graph_spec = ArenaEnvGraphSpec.model_validate(raw)
    arena_env = graph_spec.to_arena_env(enable_cameras=True)
    preview_name = f"{arena_env.name}_preview_{uuid.uuid4().hex[:8]}"
    arena_env.name = preview_name
    _preview_log(started_at, f"validated spec → arena env ({preview_name})")

    env_spacing = _compute_env_spacing(arena_env)
    builder_cfg = _preview_cfg(num_envs=num_envs, env_spacing=env_spacing)
    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    policy = ZeroActionPolicy(ZeroActionPolicyCfg())

    stamp = int(time.time() * 1000)
    video_dir = sim_preview_cache_dir() / f"{preview_name}_{stamp}"
    video_cfg = VideoRecordingCfg(
        record_viewport_video=True,
        record_camera_video=True,
        video_base_dir=str(video_dir),
        flush_partial_camera_videos=True,
    )

    pool_layouts = builder_cfg.num_envs * _PREVIEW_LAYOUTS_PER_ENV
    env = None
    try:
        _preview_log(
            started_at,
            f"solving spatial relations ({builder_cfg.num_envs} envs, {pool_layouts} layout pool)…",
        )
        t_relations = time.monotonic()
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        _preview_log(started_at, f"relation solver finished ({time.monotonic() - t_relations:.1f}s)")

        _preview_log(started_at, "spawning sim scene (gym.make)…")
        t_spawn = time.monotonic()
        env = builder.make_registered(env_cfg, env_kwargs, render_mode=video_cfg.render_mode)
        env = wrap_env_for_video(env, video_cfg, num_steps=num_steps, num_episodes=None)
        _preview_log(started_at, f"sim scene ready ({time.monotonic() - t_spawn:.1f}s)")

        rollout_policy(env, policy, num_steps=num_steps, num_episodes=None)
        env.close()
        viewport_video, camera_videos = _collect_recorded_videos(video_dir)

        print(
            f"[sim_preview] recorded {num_envs} envs @ {env_spacing}m spacing, {num_steps} zero-action steps "
            f"(total {time.monotonic() - started_at:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return {
            "ok": True,
            "viewport_video": str(viewport_video),
            "camera_videos": camera_videos,
            "env_name": preview_name,
            "num_envs": num_envs,
            "env_spacing": env_spacing,
            "num_steps": num_steps,
        }
    finally:
        _close_env_and_reset_sim(env, app=app, suppress_exceptions=True)
        with suppress(Exception):
            if preview_name in gym.registry:
                del gym.registry[preview_name]
