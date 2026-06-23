# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""16-env relation-solver rollout preview for the review GUI SimApp server."""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec

PREVIEW_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm_env_gen_sim_preview"

NUM_ENVS = 16
ENV_SPACING_M = 1.5
NUM_STEPS = 10

# Placement pool size when preview uses resolve_on_reset=False (see ObjectPlacerParams).
_PREVIEW_LAYOUTS_PER_ENV = 5


def _preview_log(started_at: float, message: str) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[sim_preview] +{elapsed:.1f}s {message}", file=sys.stderr, flush=True)


def _preview_args(*, num_envs: int = NUM_ENVS, env_spacing: float = ENV_SPACING_M) -> argparse.Namespace:
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
    """Return (eye, lookat) in world frame for a high oblique view of the full env grid."""
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    max_x = max((cols - 1) * env_spacing, 0.0)
    max_y = max((rows - 1) * env_spacing, 0.0)
    span = max(max_x, max_y, env_spacing)
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
    for _ in range(20):
        app.update()


def _capture_viewport(app, cache_path: Path) -> bytes | None:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: PLC0415

    from isaaclab_arena_examples.agentic_environment_generation.review_gui.thumbnail_render import (  # noqa: PLC0415
        _wait_for_capture,
    )

    viewport = get_active_viewport()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    capture_obj = capture_viewport_to_file(viewport, str(cache_path))
    for _ in range(10):
        app.update()
    _wait_for_capture(app, capture_obj, cache_path, max_updates=300)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    return None


def run_sim_preview(
    app,
    yaml_text: str,
    *,
    num_envs: int = NUM_ENVS,
    num_steps: int = NUM_STEPS,
    env_spacing: float = ENV_SPACING_M,
) -> dict[str, Any]:
    """Link spec → arena env → relation solver → zero-action steps; capture viewport frames."""
    import gymnasium as gym
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import close_env_and_reset_sim

    started_at = time.monotonic()
    _preview_log(started_at, "run_sim_preview started")

    close_env_and_reset_sim(suppress_exceptions=True, app=app)
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

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    first_path = PREVIEW_CACHE_DIR / f"{preview_name}_{stamp}_first.png"
    last_path = PREVIEW_CACHE_DIR / f"{preview_name}_{stamp}_last.png"

    pool_layouts = args.num_envs * _PREVIEW_LAYOUTS_PER_ENV
    env = None
    try:
        eye, target = _overview_camera(args.num_envs, args.env_spacing)
        _preview_log(
            started_at,
            f"solving spatial relations ({args.num_envs} envs, {pool_layouts} layout pool)…",
        )
        t_relations = time.monotonic()
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        _preview_log(started_at, f"relation solver finished ({time.monotonic() - t_relations:.1f}s)")

        env_cfg.viewer = ViewerCfg(eye=eye, lookat=target, origin_type="world")
        _preview_log(started_at, "spawning sim scene (gym.make)…")
        t_spawn = time.monotonic()
        env = builder.make_registered(env_cfg, env_kwargs)
        _preview_log(started_at, f"sim scene ready ({time.monotonic() - t_spawn:.1f}s)")

        obs, _ = env.reset()
        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if _capture_viewport(app, first_path) is None:
            raise RuntimeError("failed to capture first-frame viewport screenshot")

        for _ in range(num_steps):
            action = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(action)

        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if _capture_viewport(app, last_path) is None:
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
        close_env_and_reset_sim(env, app=app)
        with suppress(Exception):
            if preview_name in gym.registry:
                del gym.registry[preview_name]
