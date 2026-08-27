# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualize DROID placement and collision geometry in the RoboCasa kitchen.

Run from the repository root inside the development container:

    DISPLAY=:1 /isaac-sim/python.sh \
        isaaclab_arena_examples/relations/droid_collision_geometry_visualizer.py \
        --viz kit --num_envs 1 \
        --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml \
        --view_steps 0

The overlay shows DROID's posed link geometry, its stand-only placement bounds,
the referenced counter bounds, the On support area, and the measured NextTo gap.
"""

from __future__ import annotations

import math
import time

import numpy as np

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

GREEN = (0.46, 0.73, 0.0, 1.0)
LINK_CYAN = (0.15, 0.82, 0.88, 1.0)
COUNTER_BLUE = (0.18, 0.42, 0.86, 1.0)
YELLOW = (1.0, 0.67, 0.0, 1.0)


def _rotated_world_bbox(local_bbox, position: np.ndarray, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Transform local axis-aligned bounds into enclosing world bounds."""
    local_min = local_bbox.min_point[0].detach().cpu().numpy()
    local_max = local_bbox.max_point[0].detach().cpu().numpy()
    corners_xy = np.array(
        [[x, y] for x in (local_min[0], local_max[0]) for y in (local_min[1], local_max[1])]
    )
    rotation = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad)],
            [math.sin(yaw_rad), math.cos(yaw_rad)],
        ]
    )
    rotated_xy = corners_xy @ rotation.T
    world_min = np.array(
        [
            rotated_xy[:, 0].min() + position[0],
            rotated_xy[:, 1].min() + position[1],
            local_min[2] + position[2],
        ]
    )
    world_max = np.array(
        [
            rotated_xy[:, 0].max() + position[0],
            rotated_xy[:, 1].max() + position[1],
            local_max[2] + position[2],
        ]
    )
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
    """Return oriented-box edges for each posed DROID collision component."""
    import trimesh

    mesh = embodiment.get_collision_mesh()
    assert mesh is not None
    components = list(mesh.split(only_watertight=False))
    rotation = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad)],
            [math.sin(yaw_rad), math.cos(yaw_rad)],
        ]
    )

    edges = []
    for component in components:
        oriented_box = component.bounding_box_oriented
        half_extents = np.asarray(oriented_box.extents, dtype=np.float64) / 2.0
        vertices = np.array(
            [
                [-half_extents[0], -half_extents[1], -half_extents[2]],
                [half_extents[0], -half_extents[1], -half_extents[2]],
                [half_extents[0], half_extents[1], -half_extents[2]],
                [-half_extents[0], half_extents[1], -half_extents[2]],
                [-half_extents[0], -half_extents[1], half_extents[2]],
                [half_extents[0], -half_extents[1], half_extents[2]],
                [half_extents[0], half_extents[1], half_extents[2]],
                [-half_extents[0], half_extents[1], half_extents[2]],
            ]
        )
        vertices = trimesh.transform_points(vertices, oriented_box.primitive.transform)
        vertices[:, :2] = vertices[:, :2] @ rotation.T
        vertices += position
        for start_index, end_index in (
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
        ):
            edges.append((tuple(vertices[start_index]), tuple(vertices[end_index])))
    return edges


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
            from isaaclab_arena.utils.pose import Pose
            from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

            env.reset()
            embodiment = builder.arena_env.embodiment
            assert embodiment is not None
            placement_pose = embodiment.get_initial_pose()
            assert isinstance(placement_pose, Pose), "The single-environment visualizer requires one fixed DROID pose"
            env_origin = env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
            robot_position = np.asarray(placement_pose.position_xyz) + env_origin
            robot_yaw = yaw_from_quat_xyzw(placement_pose.rotation_xyzw)

            assets = builder.arena_env.scene.assets
            counter_bbox = assets["right_counter_top"].get_world_bounding_box()
            floor_bbox = assets["floor"].get_world_bounding_box()
            counter_min = counter_bbox.min_point[0].detach().cpu().numpy()
            counter_max = counter_bbox.max_point[0].detach().cpu().numpy()
            floor_max = floor_bbox.max_point[0].detach().cpu().numpy()
            stand_min, stand_max = _rotated_world_bbox(
                embodiment.get_bounding_box(),
                robot_position,
                robot_yaw,
            )

            next_to_distance = float(counter_min[1] - stand_max[1])
            on_clearance = float(stand_min[2] - floor_max[2])
            print(f"Measured NextTo face distance: {next_to_distance:.4f} m")
            print(f"Measured On floor clearance: {on_clearance:.4f} m")

            import omni.kit.app
            from isaacsim.util.debug_draw import _debug_draw

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            debug_draw = _debug_draw.acquire_debug_draw_interface()
            debug_draw.clear_lines()
            debug_draw.clear_points()

            _draw_lines(debug_draw, _bbox_edges(stand_min, stand_max), GREEN, 6.0)
            _draw_lines(debug_draw, _bbox_edges(counter_min, counter_max), COUNTER_BLUE, 5.0)
            _draw_lines(
                debug_draw,
                _posed_link_edges(embodiment, robot_position, robot_yaw),
                LINK_CYAN,
                2.0,
            )

            line_x = float(np.clip(robot_position[0], counter_min[0], counter_max[0]))
            line_z = float(min(stand_max[2], counter_max[2]) - 0.08)
            distance_start = (line_x, float(stand_max[1]), line_z)
            distance_end = (line_x, float(counter_min[1]), line_z)
            _draw_lines(debug_draw, [(distance_start, distance_end)], YELLOW, 9.0)
            debug_draw.draw_points([distance_start, distance_end], [YELLOW, YELLOW], [12.0, 12.0])

            support_margin = 0.08
            support_z = float(floor_max[2] + 0.004)
            support_patch = [
                (
                    (float(stand_min[0] - support_margin), float(stand_min[1] - support_margin), support_z),
                    (float(stand_max[0] + support_margin), float(stand_min[1] - support_margin), support_z),
                ),
                (
                    (float(stand_max[0] + support_margin), float(stand_min[1] - support_margin), support_z),
                    (float(stand_max[0] + support_margin), float(stand_max[1] + support_margin), support_z),
                ),
                (
                    (float(stand_max[0] + support_margin), float(stand_max[1] + support_margin), support_z),
                    (float(stand_min[0] - support_margin), float(stand_max[1] + support_margin), support_z),
                ),
                (
                    (float(stand_min[0] - support_margin), float(stand_max[1] + support_margin), support_z),
                    (float(stand_min[0] - support_margin), float(stand_min[1] - support_margin), support_z),
                ),
            ]
            _draw_lines(debug_draw, support_patch, GREEN, 7.0)

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
