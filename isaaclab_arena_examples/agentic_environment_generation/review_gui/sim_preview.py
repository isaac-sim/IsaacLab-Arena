# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""16-env relation-solver rollout preview for the review GUI SimApp sidecar."""

from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec

PREVIEW_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm_env_gen_sim_preview"

NUM_ENVS = 16
ENV_SPACING_M = 1.5
NUM_STEPS = 10


def _preview_args() -> argparse.Namespace:
    return argparse.Namespace(
        num_envs=NUM_ENVS,
        env_spacing=ENV_SPACING_M,
        device="cuda:0",
        disable_fabric=False,
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
    cx, cy = max_x * 0.5, max_y * 0.5
    span = max(max_x, max_y, env_spacing)
    # Oblique overview: close enough to fill the frame, still high enough for all clones.
    height = span * 1.75 + env_spacing * 2.5
    back = span * 1.4 + env_spacing * 2.0
    side = span * 0.25
    eye = (cx + side, cy - back, height)
    target = (cx, cy, 0.75)
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


def run_sim_preview(app, yaml_text: str) -> dict[str, Any]:
    """Link spec → arena env → relation solver → zero-action steps; capture viewport frames."""
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import close_env_and_reset_sim

    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping, got {type(raw).__name__}")

    initial_spec = ArenaEnvInitialGraphSpec.model_validate(raw)
    graph_spec = initial_spec.link()
    arena_env = graph_spec.to_arena_env()
    preview_name = f"{arena_env.name}_preview_{uuid.uuid4().hex[:8]}"
    arena_env.name = preview_name

    args = _preview_args()
    builder = ArenaEnvBuilder(arena_env, args)
    policy = ZeroActionPolicy(ZeroActionPolicyArgs())

    PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    first_path = PREVIEW_CACHE_DIR / f"{preview_name}_{stamp}_first.png"
    last_path = PREVIEW_CACHE_DIR / f"{preview_name}_{stamp}_last.png"

    env = None
    try:
        eye, target = _overview_camera(args.num_envs, args.env_spacing)
        env_cfg = builder.compose_manager_cfg()
        # World-frame overview (not task look-at-object) so all env clones are visible.
        env_cfg.viewer = ViewerCfg(eye=eye, lookat=target, origin_type="world")
        env = builder.make_registered(env_cfg)

        obs, _ = env.reset()
        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if _capture_viewport(app, first_path) is None:
            raise RuntimeError("failed to capture first-frame viewport screenshot")

        for _ in range(NUM_STEPS):
            action = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(action)

        _apply_overview_camera(env, app, args.num_envs, args.env_spacing)

        if _capture_viewport(app, last_path) is None:
            raise RuntimeError("failed to capture last-frame viewport screenshot")

        print(
            f"[sim_preview] captured {NUM_ENVS} envs @ {ENV_SPACING_M}m spacing, {NUM_STEPS} zero-action steps",
            file=sys.stderr,
        )
        return {
            "ok": True,
            "first_frame": str(first_path),
            "last_frame": str(last_path),
            "env_name": preview_name,
            "num_envs": args.num_envs,
            "env_spacing": args.env_spacing,
            "num_steps": NUM_STEPS,
        }
    finally:
        close_env_and_reset_sim(env)
