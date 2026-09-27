# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic layout sampler for Berkeley's terminated cable weave tiers."""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class CableRoutingLayout:
    """The model-changing values drawn for one cable-routing layout."""

    bow_fraction: float
    shift_fraction: float
    yaw_fraction: float
    wave_fraction: float
    peg_positions_xy: tuple[tuple[float, float], ...]
    port_pose_xyyaw: tuple[float, float, float]


def _route(
    stations: Sequence[tuple[float, float]],
    sides: Sequence[float],
    *,
    seat_offset: float,
    tail: float,
) -> list[tuple[float, float]]:
    seated = [(x + seat_offset * sides[index], y) for index, (x, y) in enumerate(stations)]
    return [
        (seated[0][0], seated[0][1] - tail),
        *seated,
        (seated[-1][0], seated[-1][1] + tail),
    ]


def _turns(polyline: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    turns = []
    for index in range(1, len(polyline) - 1):
        ax = polyline[index][0] - polyline[index - 1][0]
        ay = polyline[index][1] - polyline[index - 1][1]
        bx = polyline[index + 1][0] - polyline[index][0]
        by = polyline[index + 1][1] - polyline[index][1]
        turns.append(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))
    return tuple(turns)


def _profile(
    stations: Sequence[tuple[float, float]],
    sides: Sequence[float],
    *,
    seat_offset: float,
    tail: float,
) -> tuple[float, float, tuple[float, ...]]:
    turns = _turns(_route(stations, sides, seat_offset=seat_offset, tail=tail))
    return sum(abs(turn) for turn in turns), min(abs(turn) for turn in turns), turns


def _terminated_route_length(
    stations: Sequence[tuple[float, float]],
    sides: Sequence[float],
    *,
    anchor_xy: tuple[float, float],
    port_xy: tuple[float, float],
    seat_offset: float,
) -> float:
    seated = [(x + seat_offset * sides[index], y) for index, (x, y) in enumerate(stations)]
    polyline = [anchor_xy, *seated, port_xy]
    return sum(math.dist(a, b) for a, b in pairwise(polyline))


def sample_cable_routing_layout(distribution: Mapping[str, Any], seed: int) -> CableRoutingLayout:
    """Draw one valid layout with Berkeley's terminated-board sampling rules."""
    rng = np.random.default_rng(seed)
    fraction_low, fraction_high = (float(value) for value in distribution["cable_fraction_range"])
    bow, shift, yaw, wave = (float(value) for value in rng.uniform(fraction_low, fraction_high, size=4))

    reference = distribution["reference"]
    board = distribution["board"]
    stations = tuple(tuple(float(value) for value in position) for position in reference["peg_positions_xy"])
    sides = tuple(float(value) for value in reference["route_sides"])
    assert len(stations) == len(sides) and len(stations) >= 2
    anchor_xy = tuple(float(value) for value in reference["anchor_position_xy"])
    port_xy = tuple(float(value) for value in reference["port_position_xy"])
    port_xy = (
        port_xy[0] - float(distribution.get("port_reach_inset_m", 0.0)),
        port_xy[1],
    )
    port_slot_xy = float(distribution["port_slot_xy_m"])
    port_xy = tuple(value + float(rng.uniform(-port_slot_xy, port_slot_xy)) for value in port_xy)
    port_slot_yaw = float(distribution["port_slot_yaw_rad"])
    port_yaw = float(reference["port_yaw_rad"]) + float(rng.uniform(-port_slot_yaw, port_slot_yaw))
    seat_offset = float(reference["seat_offset_m"])
    tail = float(reference["route_tail_m"])

    reference_turn, reference_weakest, _ = _profile(stations, sides, seat_offset=seat_offset, tail=tail)
    reference_length = _terminated_route_length(
        stations,
        sides,
        anchor_xy=anchor_xy,
        port_xy=port_xy,
        seat_offset=seat_offset,
    )
    pitch = float(np.mean(np.diff([y for _, y in stations])))
    center_x = float(np.mean([x for x, _ in stations]))
    depth = float(np.mean([abs(x - center_x) for x, _ in stations]))
    depth *= float(board.get("stagger_scale", 1.0))

    workspace_x = tuple(float(value) for value in board["workspace_x_range_m"])
    workspace_y = tuple(float(value) for value in board["workspace_y_range_m"])
    terminal_clearance = float(board["terminal_clearance_m"])
    corridor = (
        max(workspace_y[0], anchor_xy[1] + terminal_clearance),
        min(workspace_y[1], port_xy[1] - terminal_clearance),
    )
    pitch_scale = tuple(float(value) for value in board["pitch_scale_range"])
    gap_low = max(float(board["min_seat_y_separation_m"]), pitch_scale[0] * pitch)
    gap_high = min(pitch_scale[1] * pitch, (corridor[1] - corridor[0]) / (len(stations) - 1))
    assert gap_low <= gap_high
    x_window = (max(workspace_x[0], min(x for x, _ in stations)), workspace_x[1])

    turn_ratio = float(board["turn_ratio"])
    turn_low = max(reference_turn / turn_ratio, float(board.get("level_lower_turn_deg", 0.0)))
    turn_high = reference_turn * turn_ratio
    depth_spread = float(board["depth_spread_fraction"])
    wobble = float(board["peg_wobble_m"])
    yaw_low, yaw_high = (float(value) for value in board["yaw_range_rad"])
    min_station_distance = float(board["min_station_distance_m"])
    min_wrap = float(board["min_wrap_fraction"]) * reference_weakest
    support_xy = tuple(float(value) for value in board["support_position_xy"])
    min_hand_distance = float(board["min_hand_distance_m"])
    max_route_length = reference_length + float(board["route_slack_allowance_m"])
    level_upper_turn = float(board["level_upper_turn_deg"])

    for _ in range(int(board["max_attempts"])):
        gaps = rng.uniform(gap_low, gap_high, size=len(stations) - 1)
        along = np.concatenate(([0.0], np.cumsum(gaps)))
        along -= along.mean()
        across = np.asarray([
            sides[index]
            * rng.uniform(
                depth * (1.0 - depth_spread),
                depth * (1.0 + depth_spread),
            )
            + rng.uniform(-wobble, wobble)
            for index in range(len(stations))
        ])
        board_yaw = float(rng.uniform(yaw_low, yaw_high))
        cosine, sine = math.cos(board_yaw), math.sin(board_yaw)
        shape_x = cosine * across - sine * along
        shape_y = sine * across + cosine * along
        room_x = (
            x_window[0] - float(shape_x.min()),
            x_window[1] - float(shape_x.max()),
        )
        room_y = (
            corridor[0] - float(shape_y.min()),
            corridor[1] - float(shape_y.max()),
        )
        if room_x[1] < room_x[0] or room_y[1] < room_y[0]:
            continue

        xs = float(rng.uniform(*room_x)) + shape_x
        ys = float(rng.uniform(*room_y)) + shape_y
        drawn = tuple(zip(xs.tolist(), ys.tolist(), strict=True))
        if any(math.dist(a, b) < min_station_distance for index, a in enumerate(drawn) for b in drawn[index + 1 :]):
            continue
        if float(np.diff(ys).min()) < float(board["min_seat_y_separation_m"]):
            continue

        turning, weakest, signed_turns = _profile(drawn, sides, seat_offset=seat_offset, tail=tail)
        inner_turns = [turn for turn in signed_turns[1:-1] if abs(turn) > 1.0e-6]
        if any(a * b >= 0.0 for a, b in pairwise(inner_turns)):
            continue
        if not (turn_low <= turning <= turn_high and turning < level_upper_turn):
            continue
        if weakest < min_wrap:
            continue

        far_station = drawn[int(np.argmax(ys))]
        if math.dist(far_station, support_xy) < min_hand_distance:
            continue
        if (
            _terminated_route_length(
                drawn,
                sides,
                anchor_xy=anchor_xy,
                port_xy=port_xy,
                seat_offset=seat_offset,
            )
            > max_route_length
        ):
            continue
        return CableRoutingLayout(
            bow_fraction=bow,
            shift_fraction=shift,
            yaw_fraction=yaw,
            wave_fraction=wave,
            peg_positions_xy=drawn,
            port_pose_xyyaw=(*port_xy, port_yaw),
        )

    raise RuntimeError(f"Cable seed {seed} produced no valid board in {board['max_attempts']} attempts")


def anchor_pose_from_cable(
    cable_positions: Sequence[tuple[float, float, float]], mouth_inset: float
) -> tuple[float, float, float]:
    """Place the anchor mouth over the cable's first point and tangent."""
    start, following = cable_positions[:2]
    heading = math.atan2(following[1] - start[1], following[0] - start[0])
    return (
        start[0] + mouth_inset * math.cos(heading),
        start[1] + mouth_inset * math.sin(heading),
        heading,
    )
