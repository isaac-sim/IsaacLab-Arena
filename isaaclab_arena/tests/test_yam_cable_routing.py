# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks for the cable-routing success geometry and the Newton YAM cable-routing environment."""

import math

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ROUTE_DIRECTIONS = (-1.0, 1.0)
WINDING_RADIUS_M = 0.012
APPROACH_DISTANCE_M = 0.065
"""Distance from a peg at which the routed path turns toward it; larger than the default radial cutoff."""
LEAD_IN_LENGTH_M = 0.05
"""Cable length before the path turns toward the first peg."""


def _routed_cable_points(peg_xy_0, peg_xy_1, z, num_points, spacing_m):
    """Return evenly spaced points of a cable wound half a turn around each peg in ROUTE_DIRECTIONS order.

    The path reaches each peg radially from the side facing away from the next peg, so only the arcs change
    the winding angle. The result is a list of ``num_points`` (x, y, z) tuples spaced ``spacing_m`` apart;
    any length beyond the second winding continues straight away from the pegs.
    """
    peg_0_to_peg_1_angle = math.atan2(peg_xy_1[1] - peg_xy_0[1], peg_xy_1[0] - peg_xy_0[0])
    behind_angle = peg_0_to_peg_1_angle + math.pi
    cable_length = (num_points - 1) * spacing_m

    def point_at(peg_xy, angle, distance):
        return (peg_xy[0] + distance * math.cos(angle), peg_xy[1] + distance * math.sin(angle))

    def arc(peg_xy, start_angle, direction):
        # Direction -1 winds counterclockwise, which increases the polar angle.
        sweep = -direction * math.pi
        return [point_at(peg_xy, start_angle + sweep * step / 32, WINDING_RADIUS_M) for step in range(33)]

    polyline = [point_at(peg_xy_0, behind_angle, APPROACH_DISTANCE_M + LEAD_IN_LENGTH_M)]
    polyline += arc(peg_xy_0, behind_angle, ROUTE_DIRECTIONS[0])
    polyline.append(point_at(peg_xy_0, peg_0_to_peg_1_angle, APPROACH_DISTANCE_M))
    polyline.append(point_at(peg_xy_1, behind_angle, APPROACH_DISTANCE_M))
    polyline += arc(peg_xy_1, behind_angle, ROUTE_DIRECTIONS[1])
    polyline.append(point_at(peg_xy_1, peg_0_to_peg_1_angle, APPROACH_DISTANCE_M))
    routed_length = sum(math.dist(start, end) for start, end in zip(polyline[:-1], polyline[1:]))
    assert routed_length <= cable_length, f"A {cable_length} m cable cannot reach past the second winding."
    polyline.append(point_at(peg_xy_1, peg_0_to_peg_1_angle, APPROACH_DISTANCE_M + cable_length))

    points = [polyline[0]]
    distance_to_next_point = spacing_m
    for start, end in zip(polyline[:-1], polyline[1:]):
        edge_length = math.dist(start, end)
        travelled = 0.0
        while len(points) < num_points and edge_length - travelled >= distance_to_next_point:
            travelled += distance_to_next_point
            fraction = travelled / edge_length
            points.append((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
            distance_to_next_point = spacing_m
        distance_to_next_point -= edge_length - travelled
    assert len(points) == num_points, f"Routed path produced {len(points)} of {num_points} points."
    return [(x, y, z) for x, y in points]


def _test_cable_route_success_geometry(simulation_app):
    import torch

    from isaaclab_arena.tasks.cable_routing_task import cable_route_success_from_geometry

    peg_positions_w = torch.tensor([[[0.0, 0.0, 0.8], [0.2, 0.0, 0.8]]])
    cable_points_w = torch.tensor([_routed_cable_points((0.0, 0.0), (0.2, 0.0), 0.8, 60, 0.01)])

    def is_routed(points, pegs=peg_positions_w, directions=ROUTE_DIRECTIONS, **thresholds):
        return bool(cable_route_success_from_geometry(points, pegs, directions, **thresholds)[0])

    assert is_routed(cable_points_w)
    assert not is_routed(cable_points_w, directions=(1.0, -1.0)), "Winding in the opposite directions succeeded."
    assert not is_routed(
        cable_points_w, pegs=peg_positions_w.flip(1), directions=ROUTE_DIRECTIONS[::-1]
    ), "Visiting the pegs in the wrong order succeeded."
    assert not is_routed(cable_points_w, completion_winding=3.5), "A half turn met a 3.5 rad winding requirement."
    lifted_points_w = cable_points_w.clone()
    lifted_points_w[..., 2] += 0.03
    assert not is_routed(lifted_points_w), "A cable above the peg's axial cutoff succeeded."
    nan_points_w = cable_points_w.clone()
    nan_points_w[0, 10, 0] = float("nan")
    assert not is_routed(nan_points_w), "A cable with non-finite points succeeded."
    assert not is_routed(cable_points_w, maximum_local_cable_length=0.05), "An over-long local winding succeeded."
    return True


def test_cable_route_success_geometry():
    assert run_function_with_persistent_simulation_app(_test_cable_route_success_geometry)


def _test_yam_cable_routing_environment(simulation_app):
    import gymnasium as gym
    import torch

    from isaaclab_arena.embodiments.yam.yam import YAM_DEFAULT_JOINT_POSITIONS
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.yam_cable_routing.scene import BOARD_POSITION, BOARD_SIZE, TABLE_TOP_Z
    from isaaclab_arena_environments.yam_cable_routing_environment import (
        YamCableRoutingEnvironment,
        YamCableRoutingEnvironmentCfg,
    )

    num_envs = 2
    arena = YamCableRoutingEnvironment().build(YamCableRoutingEnvironmentCfg())
    builder = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=num_envs, seed=42))
    name, cfg, _ = builder.build_registered()
    cfg.recorders = {}
    env = gym.make(name, cfg=cfg)
    try:
        base = env.unwrapped
        cable = base.scene[arena.task.cable.name]
        route_success = arena.task.get_termination_cfg().success[0].predicate_sequence[0]
        env_origins = base.scene.env_origins

        arm_joint_targets = [YAM_DEFAULT_JOINT_POSITIONS[f"joint{index}"] for index in range(1, 7)]
        open_gripper = [0.0]
        hold_default_pose = torch.tensor(
            [arm_joint_targets + open_gripper + arm_joint_targets + open_gripper], device=base.device
        ).repeat(num_envs, 1)
        assert hold_default_pose.shape[1] == base.action_manager.total_action_dim

        with torch.inference_mode():
            env.reset()
            default_segment_pose_w = cable.data.default_segment_pose_w.torch.clone()
            assert not route_success(base).any(), "The initial cable layout already counts as routed."

            for _ in range(30):
                env.step(hold_default_pose)

            t_W_segments = cable.data.segment_pose_w.torch[..., :3]
            assert torch.isfinite(t_W_segments).all(), "Cable segment positions became non-finite."
            t_E_segments = t_W_segments - env_origins[:, None, :]
            board_margin_m = 0.02
            board_half_extents_xy = torch.tensor(BOARD_SIZE, device=base.device) / 2 + board_margin_m
            board_center_xy = torch.tensor(BOARD_POSITION[:2], device=base.device)
            assert (
                (t_E_segments[..., :2] - board_center_xy).abs() <= board_half_extents_xy
            ).all(), "The cable left the board while both arms held their default pose."
            assert (
                (t_E_segments[..., 2] > TABLE_TOP_Z) & (t_E_segments[..., 2] < TABLE_TOP_Z + 0.05)
            ).all(), "The cable fell through or jumped above the board."

            # Write a routed cable into env 0 and keep the default layout in env 1.
            t_W_pegs = torch.stack([base.scene[peg.name].data.root_pos_w.torch for peg in arena.task.pegs], dim=1)
            num_segments = default_segment_pose_w.shape[1]
            routed_t_W = torch.tensor(
                _routed_cable_points(
                    t_W_pegs[0, 0, :2].tolist(),
                    t_W_pegs[0, 1, :2].tolist(),
                    t_W_pegs[0, 0, 2].item(),
                    num_segments,
                    0.01,
                ),
                device=base.device,
            )
            segment_pose_w = default_segment_pose_w.clone()
            segment_pose_w[0, :, :3] = routed_t_W
            cable.write_segment_pose_to_sim_index(segment_pose=segment_pose_w)
            cable.update(0.0)
            assert route_success(base).tolist() == [True, False]

            env.reset()
            torch.testing.assert_close(cable.data.segment_pose_w.torch, default_segment_pose_w)
            assert not route_success(base).any(), "Reset did not restore the unrouted cable."
    finally:
        env.close()
    return True


@pytest.mark.with_newton
def test_yam_cable_routing_environment():
    assert run_function_with_persistent_simulation_app(_test_yam_cable_routing_environment)
