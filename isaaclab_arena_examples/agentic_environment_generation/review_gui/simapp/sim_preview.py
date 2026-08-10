# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation-solver rollout preview for the review GUI SimApp server."""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from contextlib import nullcontext, suppress
from typing import Any

from isaaclab_arena.agentic_environment_generation.spec_io import safe_filename_stem
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    collect_garbage_and_clear_cuda_cache,
    teardown_simulation_app,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    CAPTURE_DONE_TAIL_UPDATES,
    pump_app,
    sim_preview_cache_dir,
)

# Placement pool size when preview uses resolve_on_reset=False (see ObjectPlacerParams).
_PREVIEW_LAYOUTS_PER_ENV = 2


def parse_sim_preview_params(req: dict[str, Any]) -> tuple[int, int, float]:
    """Read required sim-preview rollout settings from a JSON-RPC request."""
    missing = [key for key in ("num_envs", "num_steps", "env_spacing") if key not in req]
    if missing:
        raise ValueError(f"missing required sim preview params: {', '.join(missing)}")
    num_envs = int(req["num_envs"])
    num_steps = int(req["num_steps"])
    env_spacing = float(req["env_spacing"])
    assert num_envs >= 1, f"num_envs must be >= 1, got {num_envs}"
    assert num_steps >= 1, f"num_steps must be >= 1, got {num_steps}"
    assert env_spacing > 0, f"env_spacing must be > 0, got {env_spacing}"
    return num_envs, num_steps, env_spacing


def _preview_log(started_at: float, message: str) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[sim_preview] +{elapsed:.1f}s {message}", file=sys.stderr, flush=True)


def _preview_cfg(*, num_envs: int, env_spacing: float) -> ArenaEnvBuilderCfg:
    return ArenaEnvBuilderCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        resolve_on_reset=False,
        disable_fabric=True,
        device="cpu",
    )


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
    env_spacing: float,
) -> dict[str, Any]:
    """Run relation-solver preview and record the task-configured viewport."""
    import gymnasium as gym
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
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
    arena_env = graph_spec.to_arena_env()
    preview_name = f"{safe_filename_stem(arena_env.name)}_preview_{uuid.uuid4().hex[:8]}"
    arena_env.name = preview_name
    _preview_log(started_at, f"validated spec → arena env ({preview_name})")

    builder_cfg = _preview_cfg(num_envs=num_envs, env_spacing=env_spacing)
    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    policy = ZeroActionPolicy(ZeroActionPolicyCfg())

    cache_dir = sim_preview_cache_dir()
    stamp = int(time.time() * 1000)
    video_dir = cache_dir / f"{preview_name}_{stamp}"
    video_cfg = VideoRecordingCfg(
        record_viewport_video=True,
        video_base_dir=str(video_dir),
    )

    pool_layouts = builder_cfg.num_envs * _PREVIEW_LAYOUTS_PER_ENV
    env = None
    completed = False
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
        _preview_log(started_at, f"sim scene ready ({time.monotonic() - t_spawn:.1f}s)")

        env = wrap_env_for_video(env, video_cfg, num_steps=num_steps, num_episodes=None)
        obs, _ = env.reset()
        for _ in range(num_steps):
            action = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(action)

        env.close()
        env = None
        video_paths = sorted(video_dir.rglob("*.mp4"))
        if len(video_paths) != 1:
            raise RuntimeError(f"expected one viewport video, found {len(video_paths)}")

        print(
            f"[sim_preview] recorded {num_envs} envs @ {env_spacing}m spacing, {num_steps} zero-action steps "
            f"(total {time.monotonic() - started_at:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        completed = True
        return {
            "ok": True,
            "video_path": str(video_paths[0]),
            "env_name": preview_name,
            "num_envs": num_envs,
            "env_spacing": env_spacing,
            "num_steps": num_steps,
        }
    finally:
        _close_env_and_reset_sim(env, app=app, suppress_exceptions=True)
        if not completed:
            shutil.rmtree(video_dir, ignore_errors=True)
        with suppress(Exception):
            if preview_name in gym.registry:
                del gym.registry[preview_name]
