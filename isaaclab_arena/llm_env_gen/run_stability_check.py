# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Driver: bring up an Arena env, settle physics, classify each rigid
object as stable / fell_off / tipped / slid / unsettled / spawn_collision.

All primitives live in
:mod:`isaaclab_arena.llm_env_gen.stability_utils`; this module owns only
the SimulationApp lifecycle, env bring-up, settle loop, optional Kit
viewport capture, and the JSON / exit-code surface.

Examples (inside the Arena container)::

    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_check.py \\
        --viz kit --num_envs 1 avocadoPnPbowltable

    /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_check.py \\
        --viz kit --num_envs 1 --object avocado avocadoPnPbowltable

Exit codes:

* 0 — every checked object is ``stable``
* 4 — at least one object is unstable (fell_off / tipped / slid / unsettled)
* 5 — at least one object had a spawn-time collision (AABB overlap or
      first-step pose jump above threshold)

When both 4 and 5 apply, 5 wins — collisions are the upstream cause.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import torch

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.llm_env_gen.stability_utils import (
    STABILITY_STATUS_SPAWN_COLLISION,
    STABILITY_STATUS_STABLE,
    add_stability_cli_args,
    classify_object,
    collect_checkable_objects,
    compute_aabb_overlap_pairs,
    format_metrics_line,
    get_rigid_pose,
    get_rigid_velocity,
    thresholds_from_args,
    tilt_angle_rad,
)
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def main() -> int:
    parser = get_isaaclab_arena_cli_parser()
    add_stability_cli_args(parser)
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli):
        # Env subparsers register after SimApp boots so they can introspect
        # the registry for env-specific flags (e.g. --object on the env).
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli = parser.parse_args()

        arena_builder = get_arena_builder_from_cli(args_cli)
        if os.environ.get("ARENA_FORCE_CONVEX_HULL") == "1":
            arena_builder.arena_env.force_convex_hull = True
        return check_stability_for_arena_builder(arena_builder, args_cli)


def check_stability_for_arena_builder(arena_builder, args_cli) -> int:
    """Run the stability check against an already-built ``ArenaEnvBuilder``.

    Caller owns the SimulationApp lifecycle. The env is built, exercised,
    and closed inside this call so the caller can rebuild a fresh
    ``arena_builder`` (with a different placement seed) and call again
    without re-booting Isaac Sim.

    Returns:
      * 0 — every object is stable
      * 4 — at least one object unstable (no spawn collision)
      * 5 — at least one object had a spawn collision (precedence over 4)
    """
    env_name = getattr(arena_builder.arena_env, "name", type(arena_builder.arena_env).__name__)
    save_render_dir = getattr(args_cli, "save_render_dir", None)

    only_name = getattr(args_cli, "stability_object", None)
    names = collect_checkable_objects(arena_builder.arena_env, only_name=only_name)
    print(
        f"[stability] Env: '{env_name}' | checking {len(names)} object(s): {names}",
        flush=True,
    )
    assert names, f"[stability] No checkable rigid objects found in env '{env_name}'."

    env, _ = arena_builder.make_registered_and_return_cfg()
    if args_cli.seed is not None:
        set_seed(args_cli.seed, env)

    env_id = int(args_cli.env_id)
    num_envs = int(args_cli.num_envs)
    assert 0 <= env_id < num_envs, f"--env_id {env_id} out of range for --num_envs {num_envs}"

    try:
        env.reset()

        # ---- 1) Initial-state pairwise AABB overlap -----------------------
        # Compute *before* stepping. After PhysX kicks in it shoves
        # interpenetrating bodies apart and the overlap signal is gone.
        overlap_pairs = compute_aabb_overlap_pairs(env, arena_builder.arena_env, names, env_id)
        overlap_partners: dict[str, list[str]] = {n: [] for n in names}
        for pair in overlap_pairs:
            overlap_partners[pair["a"]].append(pair["b"])
            overlap_partners[pair["b"]].append(pair["a"])
        if overlap_pairs:
            print("[stability] Initial AABB overlaps detected:", flush=True)
            for pair in overlap_pairs:
                ox, oy, oz = pair["overlap_xyz"]
                print(
                    f"[stability]   {pair['a']} <-> {pair['b']}: overlap=({ox:.4f}, {oy:.4f}, {oz:.4f}) m",
                    flush=True,
                )
        else:
            print("[stability] No initial AABB overlaps.", flush=True)

        # ---- 2) Snapshot spawn pose ---------------------------------------
        zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        spawn_pose: dict[str, tuple[torch.Tensor, torch.Tensor]] = {n: get_rigid_pose(env, n, env_id) for n in names}

        # ---- 3) One physics step → first-step jump ------------------------
        env.step(zero)
        first_step_pose: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            n: get_rigid_pose(env, n, env_id) for n in names
        }

        # ---- 4) Settle loop -----------------------------------------------
        for _ in range(int(args_cli.settle_steps)):
            env.step(zero)

        # ---- 5) Final readout & per-object classification -----------------
        thresholds = thresholds_from_args(args_cli)
        per_object: dict[str, dict] = {}
        for name in names:
            spawn_pos, spawn_quat = spawn_pose[name]
            t1_pos, _ = first_step_pose[name]
            now_pos, now_quat = get_rigid_pose(env, name, env_id)
            lin_vel, ang_vel = get_rigid_velocity(env, name, env_id)

            xy_drift = float(torch.linalg.norm((now_pos - spawn_pos)[:2]).item())
            z_drop = float(max(0.0, (spawn_pos[2] - now_pos[2]).item()))
            first_step_jump = float(torch.linalg.norm(t1_pos - spawn_pos).item())
            tilt = tilt_angle_rad(spawn_quat, now_quat)
            lin_norm = float(torch.linalg.norm(lin_vel).item())
            ang_norm = float(torch.linalg.norm(ang_vel).item())

            metrics = {
                "first_step_jump_m": first_step_jump,
                "xy_drift_m": xy_drift,
                "z_drop_m": z_drop,
                "tilt_rad": tilt,
                "lin_vel_norm": lin_norm,
                "ang_vel_norm": ang_norm,
                "aabb_overlap_with": overlap_partners[name],
                "spawn_xyz": [float(v.item()) for v in spawn_pos],
                "settled_xyz": [float(v.item()) for v in now_pos],
            }
            status = classify_object(metrics, thresholds)
            metrics["status"] = status
            per_object[name] = metrics
            print(format_metrics_line(name, metrics, status), flush=True)

        # ---- 6) Overall verdict + exit code -------------------------------
        statuses = [m["status"] for m in per_object.values()]
        if STABILITY_STATUS_SPAWN_COLLISION in statuses:
            overall = STABILITY_STATUS_SPAWN_COLLISION
            exit_code = 5
        elif all(s == STABILITY_STATUS_STABLE for s in statuses):
            overall = STABILITY_STATUS_STABLE
            exit_code = 0
        else:
            # Pick the worst non-collision status as the headline label.
            overall = next(s for s in statuses if s != STABILITY_STATUS_STABLE)
            exit_code = 4
        print(f"[stability] Overall: {overall}", flush=True)

        # ---- 7) Optional viewport dwell + capture -------------------------
        if not getattr(args_cli, "headless", False):
            for _ in range(int(args_cli.dwell_steps)):
                env.step(zero)

        if save_render_dir:
            _save_scene_render(env, save_render_dir, env_name, overall)

        if args_cli.json:
            payload = {
                "env_name": env_name,
                "overall_status": overall,
                "objects": per_object,
                "aabb_overlap_pairs": overlap_pairs,
            }
            print(json.dumps(payload), flush=True)
            sys.stdout.flush()

        return exit_code

    finally:
        # Mirror run_reachability_check.py: tear down the SimulationContext
        # and open a fresh USD stage *before* env.close() so a subsequent
        # gym.make() in the same process gets a clean stage.
        teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
        env.close()


def _save_scene_render(env, save_render_dir: str, env_name: str, overall_status: str) -> None:
    """Capture the active Kit viewport and save it under ``save_render_dir``.

    Naming: ``<env_name>_stability_<overall_status>.png``. Uses
    ``omni.kit.viewport.utility.capture_viewport_to_file`` so it works
    without ``--enable_cameras``. Headless runs (no viewport) skip with
    a log line.
    """
    from pathlib import Path

    out_dir = Path(save_render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_name}_stability_{overall_status}.png"

    try:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    except ImportError as exc:
        print(f"[stability] viewport capture unavailable ({exc!r}); skipping render save.", flush=True)
        return

    viewport = get_active_viewport()
    if viewport is None:
        print(
            "[stability] no active Kit viewport (running headless?); skipping render save.",
            flush=True,
        )
        return

    capture_viewport_to_file(viewport, file_path=str(out_path))
    # Capture is async; pump the app's frame loop until the PNG lands so
    # teardown right after this call doesn't destroy the viewport mid-write.
    with contextlib.suppress(ImportError):
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(60):
            app.update()
            if out_path.exists():
                break

    if out_path.exists():
        print(f"[stability] Saved viewport render: {out_path}", flush=True)
    else:
        print(
            f"[stability] viewport capture queued for {out_path} but did not land within 60 frames; "
            "subsequent app.update() ticks should still flush it.",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
