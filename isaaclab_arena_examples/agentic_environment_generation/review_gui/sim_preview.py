# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""16-env relation-solver rollout preview for the review GUI SimApp sidecar."""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvInitialGraphSpec

PREVIEW_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "llm_env_gen_sim_preview"

NUM_ENVS = 16
ENV_SPACING_M = 1.5
NUM_STEPS = 50


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
    """Return (eye, target) for a diagonal overview of the env grid."""
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    span_x = max((cols - 1) * env_spacing, env_spacing)
    span_y = max((rows - 1) * env_spacing, env_spacing)
    cx, cy = span_x * 0.5, span_y * 0.5
    radius = max(span_x, span_y) + env_spacing
    eye = (cx + radius * 0.2, cy - radius * 1.15, radius * 1.25)
    target = (cx, cy, 0.35)
    return eye, target


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
    """Link spec → arena env → relation solver → 50 zero-action steps; capture overview frames."""
    import yaml

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg, teardown_simulation_app

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
        env = builder.make_registered()
        reapply_viewer_cfg(env)

        eye, target = _overview_camera(args.num_envs, args.env_spacing)
        env.unwrapped.sim.set_camera_view(eye, target)
        for _ in range(15):
            app.update()

        obs, _ = env.reset()
        for _ in range(10):
            app.update()

        if _capture_viewport(app, first_path) is None:
            raise RuntimeError("failed to capture first-frame overview screenshot")

        for _ in range(NUM_STEPS):
            action = policy.get_action(env, obs)
            obs, _, _, _, _ = env.step(action)

        for _ in range(10):
            app.update()

        if _capture_viewport(app, last_path) is None:
            raise RuntimeError("failed to capture last-frame overview screenshot")

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
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        teardown_simulation_app(suppress_exceptions=True)
