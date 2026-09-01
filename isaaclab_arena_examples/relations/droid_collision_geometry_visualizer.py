# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize DROID placement and the object it is next to.

Run from the repository root inside the development container:

    DISPLAY=:1 /isaac-sim/python.sh \
        isaaclab_arena_examples/relations/droid_collision_geometry_visualizer.py \
        --viz kit --num_envs 1 \
        --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml \
        --view_steps 0

The overlay shows DROID's posed link geometry, its stand-only placement bounds,
the NextTo reference bounds, and the measured gap along the requested side.
"""

from __future__ import annotations

import math
import numpy as np
import time

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

GREEN = (0.46, 0.73, 0.0, 1.0)
LINK_CYAN = (0.15, 0.82, 0.88, 1.0)
REFERENCE_BLUE = (0.18, 0.42, 0.86, 1.0)
YELLOW = (1.0, 0.67, 0.0, 1.0)


def _rotated_world_bbox(local_bbox, position: np.ndarray, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Transform local axis-aligned bounds into enclosing world bounds."""
    local_min = local_bbox.min_point[0].detach().cpu().numpy()
    local_max = local_bbox.max_point[0].detach().cpu().numpy()
    corners_xy = np.array([[x, y] for x in (local_min[0], local_max[0]) for y in (local_min[1], local_max[1])])
    rotation = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad), math.cos(yaw_rad)],
    ])
    rotated_xy = corners_xy @ rotation.T
    world_min = np.array([
        rotated_xy[:, 0].min() + position[0],
        rotated_xy[:, 1].min() + position[1],
        local_min[2] + position[2],
    ])
    world_max = np.array([
        rotated_xy[:, 0].max() + position[0],
        rotated_xy[:, 1].max() + position[1],
        local_max[2] + position[2],
    ])
    return world_min, world_max


def _bbox_edges(minimum: np.ndarray, maximum: np.ndarray) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Return the twelve edges of an axis-aligned box."""
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edge_indices = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return [(corners[start], corners[end]) for start, end in edge_indices]


def _draw_lines(debug_draw, edges, color, width: float) -> None:
    """Draw line segments with a shared color and width."""
    starts = [edge[0] for edge in edges]
    ends = [edge[1] for edge in edges]
    debug_draw.draw_lines(starts, ends, [color] * len(edges), [width] * len(edges))


def _posed_link_edges(embodiment, position: np.ndarray, yaw_rad: float):
    """Return the collision-mesh box edges posed into world, without re-fitting an OBB.

    Each link solid is already a rectangular box in the link frame. ``bounding_box_oriented``
    uses PCA and can rotate that box off the long axis (panda_link5 is the clear case).
    """
    mesh = embodiment.get_collision_mesh()
    assert mesh is not None
    components = list(mesh.split(only_watertight=False))
    rotation = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad), math.cos(yaw_rad)],
    ])

    edges = []
    for component in components:
        vertices = np.asarray(component.vertices, dtype=np.float64)
        vertices[:, :2] = vertices[:, :2] @ rotation.T
        vertices += position
        seen: set[tuple[int, int]] = set()
        for face in component.faces:
            for i in range(3):
                start_index = int(face[i])
                end_index = int(face[(i + 1) % 3])
                key = (min(start_index, end_index), max(start_index, end_index))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((tuple(vertices[start_index]), tuple(vertices[end_index])))
    return edges


def _embodiment_next_to(embodiment):
    """Return the robot's NextTo relation (the object it is placed beside)."""
    from isaaclab_arena.relations.relations import NextTo

    next_to_relations = [relation for relation in embodiment.get_relations() if isinstance(relation, NextTo)]
    assert next_to_relations, "DROID has no next_to relation to visualize"
    assert len(next_to_relations) == 1, "DROID has more than one next_to relation"
    return next_to_relations[0]


def _next_to_gap_segment(
    stand_min: np.ndarray,
    stand_max: np.ndarray,
    reference_min: np.ndarray,
    reference_max: np.ndarray,
    robot_position: np.ndarray,
    side,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return gap endpoints and measured face distance for the NextTo side."""
    from isaaclab_arena.relations.relation_loss_strategies import SIDE_CONFIGS

    cfg = SIDE_CONFIGS[side]
    axis = int(cfg.primary_axis)
    band = 1 - axis
    band_value = float(np.clip(robot_position[band], reference_min[band], reference_max[band]))
    height = float(min(stand_max[2], reference_max[2]) - 0.08)
    if int(cfg.direction) > 0:
        stand_face = float(stand_min[axis])
        reference_face = float(reference_max[axis])
        distance = stand_face - reference_face
    else:
        stand_face = float(stand_max[axis])
        reference_face = float(reference_min[axis])
        distance = reference_face - stand_face
    start = [0.0, 0.0, height]
    end = [0.0, 0.0, height]
    start[axis] = stand_face
    end[axis] = reference_face
    start[band] = band_value
    end[band] = band_value
    return tuple(start), tuple(end), distance


def main() -> None:
    """Build the kitchen environment and draw placement constraints."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--view_steps",
        type=int,
        default=1,
        help="Additional viewer frames after capture; 0 keeps the viewer open until closed.",
    )
    args_cli, _ = parser.parse_known_args()

    with SimulationAppContext(args_cli) as simulation_context:
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args_cli, hydra_overrides = parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, parser)
        builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        env = builder.make_registered()

        try:
            from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

            env.reset()
            embodiment = builder.arena_env.embodiment
            assert embodiment is not None
            # Read the spawned robot rather than the configured pose: with resolve_on_reset the reset event
            # draws its own layout from the placement pool, so the solved spawn pose can be a different draw.
            root_pose = env.unwrapped.scene[embodiment.get_scene_key()].data.root_pose_w.torch[0]
            root_pose = root_pose.detach().cpu().numpy()
            robot_position = root_pose[:3]
            robot_yaw = yaw_from_quat_xyzw(tuple(float(value) for value in root_pose[3:]))

            next_to = _embodiment_next_to(embodiment)
            reference = next_to.parent
            reference_bbox = reference.get_world_bounding_box()
            # Reference bounds are env-local; the robot pose and debug draw are world, so shift by env 0's origin.
            env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
            reference_min = reference_bbox.min_point[0].detach().cpu().numpy() + env_origin
            reference_max = reference_bbox.max_point[0].detach().cpu().numpy() + env_origin
            stand_min, stand_max = _rotated_world_bbox(
                embodiment.get_bounding_box(),
                robot_position,
                robot_yaw,
            )
            distance_start, distance_end, next_to_distance = _next_to_gap_segment(
                stand_min,
                stand_max,
                reference_min,
                reference_max,
                robot_position,
                next_to.side,
            )
            print(
                f"NextTo reference: {reference.name} side={next_to.side.value} "
                f"target={next_to.distance_m:.4f} m measured={next_to_distance:.4f} m"
            )

            import omni.kit.app
            from isaacsim.util.debug_draw import _debug_draw

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            debug_draw = _debug_draw.acquire_debug_draw_interface()
            debug_draw.clear_lines()
            debug_draw.clear_points()

            _draw_lines(debug_draw, _bbox_edges(stand_min, stand_max), GREEN, 6.0)
            _draw_lines(debug_draw, _bbox_edges(reference_min, reference_max), REFERENCE_BLUE, 5.0)
            _draw_lines(
                debug_draw,
                _posed_link_edges(embodiment, robot_position, robot_yaw),
                LINK_CYAN,
                2.0,
            )
            _draw_lines(debug_draw, [(distance_start, distance_end)], YELLOW, 9.0)
            debug_draw.draw_points([distance_start, distance_end], [YELLOW, YELLOW], [12.0, 12.0])

            env.unwrapped.sim.set_camera_view(
                eye=tuple((robot_position + np.array([2.8, -2.7, 2.25])).tolist()),
                target=tuple((robot_position + np.array([0.0, 0.0, 0.65])).tolist()),
            )

            print("Viewer ready. Close the Kit window to exit.", flush=True)
            step = 0
            while simulation_context.is_running():
                if args_cli.view_steps and step >= args_cli.view_steps:
                    break
                env.unwrapped.sim.step(render=True)
                time.sleep(1.0 / 60.0)
                step += 1
        finally:
            env.close()


if __name__ == "__main__":
    main()
