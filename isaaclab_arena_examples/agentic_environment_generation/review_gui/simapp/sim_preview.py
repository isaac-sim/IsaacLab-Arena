# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation-solver rollout preview for the review GUI SimApp server."""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext, suppress
from typing import Any

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec
from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    collect_garbage_and_clear_cuda_cache,
    teardown_simulation_app,
)
from isaaclab_arena_examples.agentic_environment_generation.review_gui.simapp.kit_viewport import (
    CAPTURE_DONE_TAIL_UPDATES,
    PRE_CAPTURE_UPDATES,
    capture_viewport_png,
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
    assert num_steps >= 0, f"num_steps must be >= 0, got {num_steps}"
    assert env_spacing > 0, f"env_spacing must be > 0, got {env_spacing}"
    return num_envs, num_steps, env_spacing


def _preview_log(started_at: float, message: str) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[sim_preview] +{elapsed:.1f}s {message}", file=sys.stderr, flush=True)


@contextmanager
def _skip_task_viewer_cfg(arena_env):
    """Stub task viewer cfg during compose; preview replaces it with the overview camera."""
    task = arena_env.task
    if task is None:
        yield
        return
    original_get_viewer_cfg = task.get_viewer_cfg
    task.get_viewer_cfg = lambda: ViewerCfg()
    try:
        yield
    finally:
        task.get_viewer_cfg = original_get_viewer_cfg


def _preview_args(*, num_envs: int, env_spacing: float) -> argparse.Namespace:
    return argparse.Namespace(
        num_envs=num_envs,
        env_spacing=env_spacing,
        device="cuda:0",
        disable_fabric=False,
        seed=42,
        solve_relations=True,
        placement_seed=None,
        resolve_on_reset=False,
        random_yaw_init=False,
        mimic=False,
        distributed=False,
        presets=None,
    )


def _overview_camera(
    num_envs: int, env_spacing: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """World-frame overview camera for the env grid."""
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    max_x = max((cols - 1) * env_spacing, 0.0)
    max_y = max((rows - 1) * env_spacing, 0.0)
    span = max(max_x, max_y) + env_spacing
    target = (0.0, 0.0, 0.0)
    height = span * 0.8 + target[2]
    back = span * 1.1
    side = span * 1.1
    eye = (side, back, height)
    return eye, target


def _apply_overview_camera(env, app, num_envs: int, env_spacing: float) -> None:
    """Point the Kit viewport at the full multi-env grid (world frame)."""
    eye, target = _overview_camera(num_envs, env_spacing)
    unwrapped = env.unwrapped
    vcc = getattr(unwrapped, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.update_view_to_world()
        vcc.update_view_location(eye=list(eye), lookat=list(target))
    else:
        unwrapped.sim.set_camera_view(eye, target)
    pump_app(app, count=PRE_CAPTURE_UPDATES)


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
    """Run relation-solver preview and capture viewport frames."""
    import gymnasium as gym
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs

    started_at = time.monotonic()
    _preview_log(started_at, "run_sim_preview started")

    _close_env_and_reset_sim(suppress_exceptions=True, app=app)
    _preview_log(started_at, "cleared stale sim state")

    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping, got {type(raw).__name__}")

    initial_spec = ArenaEnvInitialGraphSpec.model_validate(raw)
    graph_spec = initial_spec.link()
    arena_env = graph_spec.to_arena_env()
    preview_name = f"{arena_env.name}_preview_{uuid.uuid4().hex[:8]}"
    arena_env.name = preview_name
    _preview_log(started_at, f"linked spec → arena env ({preview_name})")

    args = _preview_args(num_envs=num_envs, env_spacing=env_spacing)
    builder = ArenaEnvBuilder(arena_env, args)
    policy = ZeroActionPolicy(ZeroActionPolicyArgs())

    cache_dir = sim_preview_cache_dir()
    stamp = int(time.time() * 1000)
    first_path = cache_dir / f"{preview_name}_{stamp}_first.png"
    last_path = cache_dir / f"{preview_name}_{stamp}_last.png"

    pool_layouts = args.num_envs * _PREVIEW_LAYOUTS_PER_ENV
    env = None
    try:
        eye, target = _overview_camera(args.num_envs, args.env_spacing)
        _preview_log(
            started_at,
            f"solving spatial relations ({args.num_envs} envs, {pool_layouts} layout pool)…",
        )
        t_relations = time.monotonic()
        with _skip_task_viewer_cfg(arena_env):
            env_cfg, env_kwargs = builder.compose_manager_cfg()
        _preview_log(started_at, f"relation solver finished ({time.monotonic() - t_relations:.1f}s)")

        env_cfg.viewer = ViewerCfg(eye=eye, lookat=target, origin_type="world")
        _preview_log(started_at, "spawning sim scene (gym.make)…")
        t_spawn = time.monotonic()
        env = builder.make_registered(env_cfg, env_kwargs)
        _preview_log(started_at, f"sim scene ready ({time.monotonic() - t_spawn:.1f}s)")

        obs, _ = env.reset()
        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if capture_viewport_png(app, first_path) is None:
            raise RuntimeError("failed to capture first-frame viewport screenshot")

        for _ in range(num_steps):
            action = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(action)

        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if capture_viewport_png(app, last_path) is None:
            raise RuntimeError("failed to capture last-frame viewport screenshot")

        print(
            f"[sim_preview] captured {num_envs} envs @ {env_spacing}m spacing, {num_steps} zero-action steps "
            f"(total {time.monotonic() - started_at:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return {
            "ok": True,
            "first_frame": str(first_path),
            "last_frame": str(last_path),
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
