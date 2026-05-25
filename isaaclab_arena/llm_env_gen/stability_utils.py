# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-stability primitives shared by the stability-check driver.

Pure functions and constants only — no CLI entrypoint and no
:class:`SimulationAppContext` lifecycle. The companion driver
:mod:`isaaclab_arena.llm_env_gen.run_stability_check` composes these
primitives into the full env-bring-up → settle → classify flow.

Scope: any Arena env. Iterates :pyattr:`scene.rigid_objects` and skips
anchor-tagged references and the robot. Measures, per object:

* spawn-time pairwise AABB overlap (geometric, no physics needed)
* first-step pose jump (1 physics step — catches PhysX-resolved
  interpenetration)
* settle-time XY drift, Z drop, linear / angular velocity, and tilt
  angle from the initial up-axis

These are the same kinds of failure modes an LLM-generated env can
introduce: an item placed inside another, a tippy object, or a pose that
slides off the table once gravity engages.
"""

from __future__ import annotations

import argparse
import math
import torch

# ---------------------------------------------------------------------------
# Status constants (stable strings — safe to emit in JSON payloads)
# ---------------------------------------------------------------------------

STABILITY_STATUS_STABLE = "stable"
STABILITY_STATUS_SPAWN_COLLISION = "spawn_collision"
STABILITY_STATUS_FELL_OFF = "fell_off"
STABILITY_STATUS_TIPPED = "tipped"
STABILITY_STATUS_SLID = "slid"
STABILITY_STATUS_UNSETTLED = "unsettled"


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_stability_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register stability-check CLI arguments on the given parser.

    Call from the driver before :func:`SimulationAppContext` is entered
    so ``--help`` works without Isaac Sim booted.
    """
    group = parser.add_argument_group(
        "Physics Stability Check",
        "Arguments for the physics-stability checker.",
    )
    group.add_argument(
        "--env_id",
        type=int,
        default=0,
        help="Environment index to check (0 to --num_envs-1).",
    )
    group.add_argument(
        "--object",
        dest="stability_object",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Optional: only check the named rigid object instead of every "
            "non-anchor rigid in the scene. Use the asset name as it appears "
            "in env.unwrapped.scene.rigid_objects."
        ),
    )
    group.add_argument(
        "--settle_steps",
        type=int,
        default=60,
        help=(
            "Number of zero-action env.step() calls used to let physics settle "
            "after the initial spawn jump. Default 60 (~2s at 30 Hz)."
        ),
    )
    group.add_argument(
        "--vel_thresh_lin",
        type=float,
        default=0.05,
        help="Linear-velocity norm threshold (m/s) at end-of-settle. Default 0.05.",
    )
    group.add_argument(
        "--vel_thresh_ang",
        type=float,
        default=0.20,
        help="Angular-velocity norm threshold (rad/s) at end-of-settle. Default 0.20.",
    )
    group.add_argument(
        "--xy_drift_thresh",
        type=float,
        default=0.05,
        help=(
            "XY position drift threshold (m) between spawn and settled state. "
            "Above this an object is flagged as 'slid'. Default 0.05."
        ),
    )
    group.add_argument(
        "--z_drop_thresh",
        type=float,
        default=0.30,
        help=(
            "Z drop threshold (m) — settled Z below spawn Z by more than this is "
            "flagged as 'fell_off'. Default 0.30 — comfortably larger than any "
            "settling drop (an asset-stability probe spawns 0.10 m above the "
            "tabletop and settles within ~0.10–0.15 m), but well below a real "
            "table-edge fall (~0.7 m). Tighten with a smaller value if your env "
            "expects assets to land already in contact with the surface."
        ),
    )
    group.add_argument(
        "--tilt_thresh_deg",
        type=float,
        default=20.0,
        help=(
            "Tilt threshold (degrees) — angle between initial and settled body "
            "Z-axes. Above this the object is flagged as 'tipped'. Default 20."
        ),
    )
    group.add_argument(
        "--first_step_jump_thresh",
        type=float,
        default=0.02,
        help=(
            "First-step pose jump threshold (m). After one physics step, an "
            "object whose center moved more than this is flagged as a "
            "spawn collision. Default 0.02 (a settling drop on contact is "
            "smaller; PhysX shoving an interpenetration apart is larger)."
        ),
    )
    group.add_argument(
        "--dwell_steps",
        type=int,
        default=90,
        help=(
            "Zero-action sim steps to take after classification so the Kit "
            "viewer stays up for inspection. Only runs when a viewer is open. "
            "Default 90."
        ),
    )
    group.add_argument(
        "--save_render_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If set, capture the active Kit viewport after the stability check "
            "and save it as '<env_name>_stability_<status>.png'. Requires "
            "'--viz kit'; headless runs are skipped."
        ),
    )
    group.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit a final single-line JSON payload to stdout with per-object metrics.",
    )


# ---------------------------------------------------------------------------
# Object discovery — pure walk over arena_env.scene
# ---------------------------------------------------------------------------


def collect_checkable_objects(arena_env, only_name: str | None = None) -> list[str]:
    """Return the names of rigid objects worth checking for stability.

    Iterates the arena scene's assets and returns those that:

    * are tagged ``ObjectType.RIGID``
    * are not the robot embodiment
    * carry no ``IsAnchor`` relation (anchors reference background sub-prims
      and are static — checking them is meaningless)

    If ``only_name`` is given, the result is filtered to that single name
    (asserts the name is in the candidate set).
    """
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor

    embodiment_name = getattr(arena_env.embodiment, "name", None)

    names: list[str] = []
    for asset in arena_env.scene.assets.values():
        # Filter to assets that expose object_type — backgrounds, lights,
        # ground planes, and the embodiment do not.
        object_type = getattr(asset, "object_type", None)
        if object_type != ObjectType.RIGID:
            continue
        # Skip the embodiment if it ever gets RIGID-tagged.
        if asset.name == embodiment_name:
            continue
        # Skip anchor references (tabletop_anchor and friends — static).
        relations = getattr(asset, "get_relations", lambda: [])()
        if any(isinstance(r, IsAnchor) for r in relations):
            continue
        names.append(asset.name)

    if only_name is not None:
        assert only_name in names, f"--object {only_name!r} not in checkable rigid objects: {names}"
        return [only_name]
    return names


# ---------------------------------------------------------------------------
# Per-object world-frame readouts
# ---------------------------------------------------------------------------


def get_rigid_pose(env, name: str, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos_xyz, quat_wxyz)`` of a rigid object in world coordinates.

    Both tensors are 1-D (``(3,)`` and ``(4,)``) on the env's compute device.
    """
    import warp as wp

    rigid_objects = env.unwrapped.scene.rigid_objects
    assert (
        name in rigid_objects
    ), f"Object '{name}' not found in scene.rigid_objects. Available: {list(rigid_objects.keys())}"
    obj = rigid_objects[name]
    pos = wp.to_torch(obj.data.root_pos_w)[env_id].clone()
    quat = wp.to_torch(obj.data.root_quat_w)[env_id].clone()
    return pos, quat


def get_rigid_velocity(env, name: str, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(lin_vel_w, ang_vel_w)`` of a rigid object in world coordinates."""
    import warp as wp

    rigid_objects = env.unwrapped.scene.rigid_objects
    assert (
        name in rigid_objects
    ), f"Object '{name}' not found in scene.rigid_objects. Available: {list(rigid_objects.keys())}"
    obj = rigid_objects[name]
    lin = wp.to_torch(obj.data.root_lin_vel_w)[env_id].clone()
    ang = wp.to_torch(obj.data.root_ang_vel_w)[env_id].clone()
    return lin, ang


# ---------------------------------------------------------------------------
# Initial-state pairwise AABB overlap (geometric, no physics step needed)
# ---------------------------------------------------------------------------


def _world_aabb_from_local(
    local_min_xyz: torch.Tensor,
    local_max_xyz: torch.Tensor,
    pos_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World axis-aligned envelope of an oriented body-frame box.

    Builds the 8 local corners, rotates them by ``quat_wxyz``, translates by
    ``pos_w``, and takes the envelope min/max. For yaw-only orientations
    (the common case after On(table) resolution) this is exact; for
    arbitrary tilts it's a conservative over-approximation, which can
    produce false-positive overlap reports — but never false negatives.
    """
    import isaaclab.utils.math as PoseUtils

    device = pos_w.device
    dtype = pos_w.dtype
    lo = local_min_xyz.to(device=device, dtype=dtype)
    hi = local_max_xyz.to(device=device, dtype=dtype)
    corners = torch.tensor(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        device=device,
        dtype=dtype,
    )
    R = PoseUtils.matrix_from_quat(quat_wxyz.to(device=device, dtype=dtype))
    world_corners = corners @ R.T + pos_w
    return world_corners.amin(dim=0), world_corners.amax(dim=0)


def compute_aabb_overlap_pairs(env, arena_env, names: list[str], env_id: int) -> list[dict]:
    """Return a list of pairwise world-AABB overlaps among ``names`` at current state.

    Each entry: ``{"a": name_a, "b": name_b, "overlap_xyz": (dx, dy, dz)}``
    where each component is the linear overlap on that axis (always > 0
    for entries returned). Pairs with non-positive overlap on any axis
    are omitted.

    Uses each object's *local* bbox (``asset.get_bounding_box()``)
    combined with the live ``root_pos_w`` / ``root_quat_w`` from the
    spawned rigid body — :pymeth:`Asset.get_world_bounding_box()` returns
    the local box for objects whose ``initial_pose`` is unset (e.g.
    placed by an ``On(table)`` resolver), so it can't be used directly.
    """
    asset_by_name = {a.name: a for a in arena_env.scene.assets.values() if a.name in names}
    bboxes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        asset = asset_by_name[name]
        local_bbox = asset.get_bounding_box()
        local_min = local_bbox.min_point[0].clone()
        local_max = local_bbox.max_point[0].clone()
        pos_w, quat_w = get_rigid_pose(env, name, env_id)
        bboxes[name] = _world_aabb_from_local(local_min, local_max, pos_w, quat_w)

    out: list[dict] = []
    sorted_names = sorted(bboxes.keys())
    for i, a in enumerate(sorted_names):
        a_min, a_max = bboxes[a]
        for b in sorted_names[i + 1 :]:
            b_min, b_max = bboxes[b]
            ox = float((torch.min(a_max[0], b_max[0]) - torch.max(a_min[0], b_min[0])).item())
            oy = float((torch.min(a_max[1], b_max[1]) - torch.max(a_min[1], b_min[1])).item())
            oz = float((torch.min(a_max[2], b_max[2]) - torch.max(a_min[2], b_min[2])).item())
            if ox > 0.0 and oy > 0.0 and oz > 0.0:
                out.append({"a": a, "b": b, "overlap_xyz": (ox, oy, oz)})
    return out


# ---------------------------------------------------------------------------
# Tilt computation
# ---------------------------------------------------------------------------


def tilt_angle_rad(quat_init_wxyz: torch.Tensor, quat_now_wxyz: torch.Tensor) -> float:
    """Angle (rad) between the initial and current body Z-axes in world frame.

    A perfectly upright object that hasn't rotated about world Z returns 0.
    Pure yaw rotation also returns 0 (the body Z-axis is unchanged), which
    is what we want — yaw doesn't affect physical stability.
    """
    import isaaclab.utils.math as PoseUtils

    R_init = PoseUtils.matrix_from_quat(quat_init_wxyz)
    R_now = PoseUtils.matrix_from_quat(quat_now_wxyz)
    z_init = R_init[:, 2]
    z_now = R_now[:, 2]
    cos_theta = torch.clamp(torch.dot(z_init, z_now), -1.0, 1.0)
    return float(torch.acos(cos_theta).item())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_object(metrics: dict, thresholds: dict) -> str:
    """Return the stability status string for one object.

    Precedence (worst-cause-first):
      1. ``spawn_collision`` if ``first_step_jump_m`` > thresh OR the object
         appears in any AABB overlap pair (caller folds that into metrics).
      2. ``fell_off``  if ``z_drop_m`` > thresh
      3. ``tipped``    if ``tilt_rad`` > tilt_thresh_rad
      4. ``slid``      if ``xy_drift_m`` > thresh
      5. ``unsettled`` if either velocity norm exceeds thresh
      6. ``stable``    otherwise
    """
    if metrics.get("aabb_overlap_with"):
        return STABILITY_STATUS_SPAWN_COLLISION
    if metrics["first_step_jump_m"] > thresholds["first_step_jump_thresh"]:
        return STABILITY_STATUS_SPAWN_COLLISION
    if metrics["z_drop_m"] > thresholds["z_drop_thresh"]:
        return STABILITY_STATUS_FELL_OFF
    if metrics["tilt_rad"] > thresholds["tilt_thresh_rad"]:
        return STABILITY_STATUS_TIPPED
    if metrics["xy_drift_m"] > thresholds["xy_drift_thresh"]:
        return STABILITY_STATUS_SLID
    if metrics["lin_vel_norm"] > thresholds["vel_thresh_lin"] or metrics["ang_vel_norm"] > thresholds["vel_thresh_ang"]:
        return STABILITY_STATUS_UNSETTLED
    return STABILITY_STATUS_STABLE


def thresholds_from_args(args_cli: argparse.Namespace) -> dict:
    """Pack the CLI threshold values into the dict ``classify_object`` expects."""
    return {
        "first_step_jump_thresh": float(args_cli.first_step_jump_thresh),
        "z_drop_thresh": float(args_cli.z_drop_thresh),
        "tilt_thresh_rad": math.radians(float(args_cli.tilt_thresh_deg)),
        "xy_drift_thresh": float(args_cli.xy_drift_thresh),
        "vel_thresh_lin": float(args_cli.vel_thresh_lin),
        "vel_thresh_ang": float(args_cli.vel_thresh_ang),
    }


def format_metrics_line(name: str, metrics: dict, status: str) -> str:
    """One-line human-readable summary suitable for ``print(..., flush=True)``."""
    return (
        f"[stability] {name}: {status} | "
        f"jump1={metrics['first_step_jump_m']:.4f}m "
        f"xy_drift={metrics['xy_drift_m']:.4f}m "
        f"z_drop={metrics['z_drop_m']:.4f}m "
        f"tilt={math.degrees(metrics['tilt_rad']):.2f}deg "
        f"|v|={metrics['lin_vel_norm']:.4f}m/s "
        f"|w|={metrics['ang_vel_norm']:.4f}rad/s"
        + (f" | overlaps={metrics['aabb_overlap_with']}" if metrics.get("aabb_overlap_with") else "")
    )
