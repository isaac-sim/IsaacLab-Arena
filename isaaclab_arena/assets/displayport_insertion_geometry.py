# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometry constants for the DisplayPort insertion connector assets."""

from __future__ import annotations

SOCKET_INSERTION_OFFSET = (0.0375, 0.0, 0.0)
"""Socket root-to-mate point offset in the socket local frame."""

PLUG_INSERTION_OFFSET = (0.0, 0.0, 0.0221)
"""Plug root-to-mate point offset in the plug local frame."""

PLUG_GOAL_ROT = (0.0, -0.70711, 0.0, 0.70711)
"""Plug orientation relative to the socket at the mated pose, in ``(x, y, z, w)`` order."""

PLUG_GOAL_ROT_INV = (0.0, 0.70711, 0.0, 0.70711)
"""Inverse of :data:`PLUG_GOAL_ROT`, in ``(x, y, z, w)`` order."""

DEFAULT_INSERTION_POINT = (0.0, 0.0, 0.1875)
"""Default mate point for the connector-only insertion scene."""

DEFAULT_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
"""Default socket orientation with the opening facing upward."""

DEFAULT_PLUG_CLEARANCE_Z = 0.033
"""Default vertical plug clearance above the socket mate point."""

PASSIVE_DROP_SOCKET_POS = (0.0, 0.0, 0.15)
"""Socket root position used by the passive DisplayPort drop-test profile."""

PASSIVE_DROP_SOCKET_ROT = DEFAULT_SOCKET_ROT
"""Socket root orientation used by the passive DisplayPort drop-test profile."""

PASSIVE_DROP_PLUG_POS = (0.0, 0.0, 0.2096)
"""Plug root position used by the passive DisplayPort drop-test profile."""

PASSIVE_DROP_PLUG_ROT = (0.70711, 0.70711, 0.0, 0.0)
"""Plug root orientation used by the passive DisplayPort drop-test profile."""


def quat_rotate_vec(
    q_xyzw: tuple[float, float, float, float], v_xyz: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply an ``(x, y, z, w)`` quaternion rotation to a 3D vector."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v_xyz
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def quat_mul(
    lhs_xyzw: tuple[float, float, float, float],
    rhs_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two quaternions in ``(x, y, z, w)`` order."""
    x1, y1, z1, w1 = lhs_xyzw
    x2, y2, z2, w2 = rhs_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def compute_socket_root(
    geometry_pos: tuple[float, float, float],
    socket_rot: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Compute the socket USD root position from a desired mate-point position."""
    rotated = quat_rotate_vec(socket_rot, SOCKET_INSERTION_OFFSET)
    return (
        geometry_pos[0] - rotated[0],
        geometry_pos[1] - rotated[1],
        geometry_pos[2] - rotated[2],
    )


def compute_plug_pose(
    geometry_pos: tuple[float, float, float],
    socket_rot: tuple[float, float, float, float],
    z_clearance: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compute plug USD root pose from a desired mate-point position and socket rotation."""
    plug_rot = quat_mul(socket_rot, PLUG_GOAL_ROT)
    plug_offset_world = quat_rotate_vec(plug_rot, PLUG_INSERTION_OFFSET)
    plug_root = (
        geometry_pos[0] - plug_offset_world[0],
        geometry_pos[1] - plug_offset_world[1],
        geometry_pos[2] - plug_offset_world[2] + z_clearance,
    )
    return plug_root, plug_rot


DEFAULT_SOCKET_ROOT_POS = compute_socket_root(DEFAULT_INSERTION_POINT, DEFAULT_SOCKET_ROT)
"""Default socket USD root position for the connector-only insertion scene."""

DEFAULT_PLUG_ROOT_POS, DEFAULT_PLUG_ROT = compute_plug_pose(
    DEFAULT_INSERTION_POINT,
    DEFAULT_SOCKET_ROT,
    z_clearance=DEFAULT_PLUG_CLEARANCE_Z,
)
"""Default plug USD root pose for the connector-only insertion scene."""
