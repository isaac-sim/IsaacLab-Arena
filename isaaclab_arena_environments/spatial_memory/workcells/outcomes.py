# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical end conditions in the floor-aligned workcell frame, in metres."""

import torch
from dataclasses import dataclass

from isaaclab.utils.math import quat_apply

from isaaclab_arena_environments.spatial_memory.workcells.layouts import REQUIRED_SUPPORT


@dataclass(frozen=True)
class Region:
    """Require a body origin to remain within a box in workcell coordinates."""

    body: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]


# Broad workspace dimensions along free axes allow different approaches to the same end condition.
GOALS = {
    "reverse_gap": (
        Region("load", (-2.5, -0.03, 0.78), (-1.36, 0.88, 1.25)),
        Region("robot", (-2.8, -0.03, 0.48), (-1.32, 0.88, 1.1)),
    ),
    "partition_delivery": (
        Region("load", (-2.5, 0.17, 0.76), (-1.94, 0.73, 1.3)),
        Region("robot", (-2.9, 0.04, 0.45), (-1.80, 0.85, 1.15)),
    ),
    "tote_extraction": (Region("tote", (-0.28, -0.42, 0.82), (0.28, -0.24, 0.93)),),
    "hanging_pan": (Region("pan", (-0.50, -0.40, 1.04), (0.02, 0.01, 1.45)),),
    "dishwasher": (
        Region("plate", (-0.07, -0.066, 1.06), (0.07, 0.045, 1.105)),
        Region("neighbor", (-0.03, 0.10, 1.05), (0.03, 0.16, 1.12)),
    ),
    "billet_insertion": (Region("billet", (-0.12, 0.015, 1.04), (0.12, 0.072, 1.16)),),
    "rail_support": (Region("crate", (-0.045, -0.09, 1.08), (0.045, 0.08, 1.11)),),
    "stack_bracing": (
        Region("spacer", (-0.03, -0.65, 0.84), (0.35, -0.31, 1.1)),
        Region("crate", (-0.045, -0.045, 1.12), (0.045, 0.045, 1.21)),
    ),
    "loaded_drawer": (Region("drawer", (-0.01, -0.023, 0.83), (0.01, 0.02, 0.85)),),
    "pipe_bracket": (Region("bracket", (-0.23, 0.012, 1.035), (0.23, 0.075, 1.105)),),
    "panel_removal": (Region("panel", (-0.45, -0.70, 1), (0.45, -0.39, 1.48)),),
    "hatch_transfer": (Region("tray", (0.49, -0.12, 0.82), (0.60, 0.12, 0.92)),),
    "probe_withdrawal": (Region("probe", (-0.38, -0.55, 0.93), (0.35, -0.115, 1.35)),),
    "display_cover": (Region("cover", (-0.4, -0.6, 1.0), (0.4, -0.20, 1.40)),),
    "panel_corner": (Region("panel", (-0.37, 0.31, 0.70), (0.44, 0.72, 1.28)),),
    "aisle_retreat": (Region("chassis", (-0.35, -1.55, 0.72), (0.35, -1.02, 1.22)),),
    "platform_step": (
        Region("robot", (-0.37, 0.105, 0.82), (0.35, 0.60, 1.17)),
        Region("crate", (-0.55, 0.05, 0.88), (0.50, 0.85, 1.45)),
    ),
    "beam_clearance": (
        Region("load", (-0.52, 0.40, 0.62), (0.49, 1.10, 1.35)),
        Region("robot", (-0.52, 0.34, 0.47), (0.52, 1.3, 1.15)),
    ),
    "trolley_dock": (Region("trolley", (-0.06, 0.13, -0.025), (0.06, 0.28, 0.04)),),
    "pillar_pass": (
        Region("case", (0.29, 0.50, 0.72), (1.15, 1.03, 1.35)),
        Region("robot", (0.29, 0.40, 0.46), (1.2, 1.3, 1.12)),
    ),
    "closing_door": (
        Region("parcel", (-0.16, 0.36, 0.72), (0.16, 1.10, 1.25)),
        Region("robot", (-0.39, 0.24, 0.45), (0.39, 1.30, 1.12)),
    ),
    "moving_carriage": (Region("part", (-0.72, -0.49, 1.24), (0.72, 0.16, 1.63)),),
    "descending_hood": (Region("tray", (-0.24, -0.85, 0.81), (0.24, -0.52, 1.14)),),
}

LEVEL_BODIES = {
    "rail_support": "crate",
    "stack_bracing": "crate",
    "hatch_transfer": "tray",
    "platform_step": "crate",
    "trolley_dock": "trolley",
    "shelf_bay": "load",
    "billet_insertion": "billet",
}


def pose_e(env, body):
    """Read a live body pose relative to the environment origin."""
    pose = env.scene[body].data.root_pose_w.torch.clone()
    pose[:, :3] -= env.scene.env_origins
    return pose


def inside(point, lower, upper):
    return ((point >= point.new_tensor(lower)) & (point <= point.new_tensor(upper))).all(dim=-1)


def at_seam(env, target):
    """Check that the tool tip follows each seam waypoint in order."""
    tip = pose_e(env, "probe")
    offset = tip.new_tensor((0.03, 0.035, -0.02)).expand(env.num_envs, 3)
    point = tip[:, :3] + quat_apply(tip[:, 3:], offset)
    return (point - point.new_tensor(target)).norm(dim=1) < 0.035


# Traversal tasks pass the workpiece through the opening before the final pose can count.
CROSSINGS = {
    "reverse_gap": Region("load", (-1.25, -0.03, 0.78), (-1.05, 0.88, 1.15)),
    "partition_delivery": Region("load", (-1.60, 0.16, 0.75), (-1.39, 0.76, 1.25)),
    "aisle_retreat": Region("chassis", (-0.24, -0.64, 0.74), (0.24, -0.48, 1.19)),
    "beam_clearance": Region("load", (-0.45, -0.10, 0.54), (0.45, 0.10, 1.10)),
    "closing_door": Region("parcel", (-0.18, -0.08, 0.68), (0.18, 0.08, 1.30)),
}


def crossing_opening(env, scene):
    region = CROSSINGS[scene]
    passed = inside(pose_e(env, region.body)[:, :3], region.lower, region.upper)
    if scene in {"reverse_gap", "aisle_retreat"}:
        q = pose_e(env, "robot")[:, 3:]
        forward = quat_apply(q, q.new_tensor((1.0, 0.0, 0.0)).expand(env.num_envs, 3))
        passed &= forward[:, 0 if scene == "reverse_gap" else 1] > 0.60
    return passed


def scoop_tipped(env):
    """Require a real pour over the chute before accepting delivery of the pellets."""
    scoop = pose_e(env, "scoop")
    vertical = scoop.new_tensor((0.0, 0.0, 1.0)).expand(env.num_envs, 3)
    return inside(scoop[:, :3], (-0.23, -0.30, 1.06), (0.23, 0.16, 1.52)) & (
        quat_apply(scoop[:, 3:], vertical)[:, 2] < 0.80
    )


def at_rest_position(env, body, tolerance_m):
    asset = env.scene[body]
    delta = pose_e(env, body)[:, :3] - asset.data.default_root_pose.torch[:, :3]
    return delta.norm(dim=1) < tolerance_m


def reached_goal(env, scene, layout):
    """Check the workpiece and support poses for the requested motion."""
    done = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for region in GOALS.get(scene, ()):
        done &= inside(pose_e(env, region.body)[:, :3], region.lower, region.upper)
    if scene in LEVEL_BODIES:
        q = pose_e(env, LEVEL_BODIES[scene])[:, 3:]
        up = q.new_tensor((0.0, 0.0, 1.0)).expand(env.num_envs, 3)
        done &= quat_apply(q, up)[:, 2] > 0.96
    if scene == "shelf_bay":
        ymin, ymax = ((-0.366, -0.239), (-0.756, -0.629))[layout]
        done &= inside(pose_e(env, "load")[:, :3], (-2.005, ymin, 0.949), (-1.90, ymax, 0.97))
        for neighbor in ("stock_a", "stock_b", "stock_c"):
            done &= at_rest_position(env, neighbor, 0.03)
    elif scene == "hanging_pan":
        done &= at_rest_position(env, "neighbor", 0.045)
    elif scene == "loaded_drawer":
        done &= inside(
            pose_e(env, "tray")[:, :3] - pose_e(env, "drawer")[:, :3], (-0.13, -0.08, 0.015), (0.13, 0.08, 0.075)
        )
    elif scene == "cabinet_reach":
        done &= pose_e(env, "button")[:, 1] > 0.254
    elif scene == "panel_catches":
        done &= (pose_e(env, "left")[:, 1] > 0.043) & (pose_e(env, "right")[:, 1] > 0.042)
    elif scene == "valve_turn":
        q = pose_e(env, "lever")[:, 3:]
        done &= quat_apply(q, q.new_tensor((1.0, 0.0, 0.0)).expand(env.num_envs, 3))[:, 2] > 0.73
    elif scene == "worktop_clamp":
        jaw = pose_e(env, "jaw")[:, :3]
        x = (-0.12, 0.11)[layout]
        done &= inside(jaw, (x - 0.033, -0.513, 0.774), (x + 0.033, -0.447, 0.797))
        done &= (jaw[:, 2] - pose_e(env, "clamp")[:, 2]) > -0.052
    elif scene == "scoop_pour":
        for i in range(6):
            done &= inside(pose_e(env, f"pellet{i}")[:, :3], (-0.18, -0.14, 0.82), (0.18, 0.14, 0.99))
    elif scene == "hatch_transfer":
        tray = pose_e(env, "tray")[:, :3]
        for cup in ("cup0", "cup1"):
            done &= inside(pose_e(env, cup)[:, :3] - tray, (-0.24, -0.11, 0.06), (0.24, 0.11, 0.12))
    elif scene == "display_cover":
        done &= ~env.cover_attached.any(dim=1)
    elif scene == "dishwasher":
        # The initially upright plate may lean slightly; its face must still be nearly vertical.
        q = pose_e(env, "plate")[:, 3:]
        normal = q.new_tensor((-0.087, -0.995, 0.052)).expand(env.num_envs, 3)
        done &= quat_apply(q, normal)[:, 1].abs() > 0.92
    elif scene == "panel_corner":
        q = pose_e(env, "panel")[:, 3:]
        done &= quat_apply(q, q.new_tensor((1.0, 0.0, 0.0)).expand(env.num_envs, 3))[:, 1].abs() > 0.75
    for surface, _ in REQUIRED_SUPPORT.get(scene, ()):
        force = env.scene["support_" + surface].data.force_matrix_w.torch[:, 0, 0, :]
        done &= force.norm(dim=-1) > 0.15
    return done
