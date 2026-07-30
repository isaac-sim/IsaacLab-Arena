# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for adding up the mass properties of several parts."""

from __future__ import annotations

import numpy as np

from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage

_IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)
"""A rotation that leaves the principal axes alone, as ``(x, y, z, w)``."""

_QUARTER_TURN_ABOUT_Z = (0.0, 0.0, 2**-0.5, 2**-0.5)
"""A rotation that is not the identity, so the ``(x, y, z, w)`` order is worth getting right."""


def _identity():
    """A transform that leaves a part where it is."""
    from pxr import Gf

    return Gf.Matrix4d(1.0)


def _inertia_tensor(properties) -> np.ndarray:
    """Rebuild the full inertia tensor from the principal moments and their rotation."""
    from isaaclab_arena.utils.usd.mass_properties import _quaternion_to_matrix

    rotation = _quaternion_to_matrix(properties.principal_axes)
    return rotation @ np.diag(properties.diagonal_inertia) @ rotation.T


def test_one_part_keeps_its_properties():
    from isaaclab_arena.utils.usd.mass_properties import MassProperties, combine_mass_properties

    part = MassProperties(
        mass=2.0,
        center_of_mass=(0.1, 0.2, 0.3),
        diagonal_inertia=(3.0, 5.0, 7.0),
        principal_axes=_IDENTITY_ROTATION,
    )

    combined = combine_mass_properties([(part, _identity())])

    assert combined.mass == 2.0
    assert np.allclose(combined.center_of_mass, (0.1, 0.2, 0.3))
    assert np.allclose(sorted(combined.diagonal_inertia), [3.0, 5.0, 7.0])
    assert np.allclose(_inertia_tensor(combined), np.diag([3.0, 5.0, 7.0]))


def test_masses_add_up_and_centres_average():
    from isaaclab_arena.utils.usd.mass_properties import MassProperties, combine_mass_properties

    left = MassProperties(1.0, (-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), _IDENTITY_ROTATION)
    right = MassProperties(3.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), _IDENTITY_ROTATION)

    combined = combine_mass_properties([(left, _identity()), (right, _identity())])

    assert combined.mass == 4.0
    # The heavier part pulls the centre of mass towards itself: (1 * -2 + 3 * 2) / 4.
    assert np.allclose(combined.center_of_mass, (1.0, 0.0, 0.0))


def test_parts_apart_resist_spinning_more():
    """Two weightless points 2 apart spin like a dumbbell: I = sum of mass times distance squared."""
    from isaaclab_arena.utils.usd.mass_properties import MassProperties, combine_mass_properties

    left = MassProperties(1.0, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), _IDENTITY_ROTATION)
    right = MassProperties(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), _IDENTITY_ROTATION)

    combined = combine_mass_properties([(left, _identity()), (right, _identity())])

    # Nothing resists spinning about the line through both points, and both resist the other two.
    assert np.allclose(_inertia_tensor(combined), np.diag([0.0, 2.0, 2.0]))


def test_part_transform_moves_its_centre_of_mass():
    from pxr import Gf

    from isaaclab_arena.utils.usd.mass_properties import MassProperties, combine_mass_properties

    part = MassProperties(1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), _IDENTITY_ROTATION)
    moved = Gf.Matrix4d(1.0)
    moved.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 5.0))

    combined = combine_mass_properties([(part, moved)])

    assert np.allclose(combined.center_of_mass, (0.0, 0.0, 5.0))
    # Measured about the new centre of mass, so moving the part does not change the inertia.
    assert np.allclose(_inertia_tensor(combined), np.eye(3))


def test_rotating_a_part_rotates_its_inertia():
    from pxr import Gf

    from isaaclab_arena.utils.usd.mass_properties import MassProperties, combine_mass_properties

    part = MassProperties(1.0, (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), _IDENTITY_ROTATION)
    turned = Gf.Matrix4d(1.0)
    turned.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 90.0))

    combined = combine_mass_properties([(part, turned)])

    # A quarter turn about z swaps how much the part resists spinning about x and y.
    assert np.allclose(_inertia_tensor(combined), np.diag([2.0, 1.0, 3.0]), atol=1e-9)


def test_reading_back_what_was_written():
    from isaaclab_arena.utils.usd.mass_properties import MassProperties, read_mass_properties, write_mass_properties

    stage = new_stage()
    body_path = add_body(stage, "body_01")
    written = MassProperties(1.5, (0.1, 0.2, 0.3), (4.0, 5.0, 6.0), _QUARTER_TURN_ABOUT_Z)

    write_mass_properties(stage.GetPrimAtPath(body_path), written)
    read = read_mass_properties(stage.GetPrimAtPath(body_path))

    assert read.mass == 1.5
    assert np.allclose(read.center_of_mass, written.center_of_mass)
    assert np.allclose(read.diagonal_inertia, written.diagonal_inertia)
    assert np.allclose(read.principal_axes, written.principal_axes), read.principal_axes


def test_a_part_that_does_not_say_what_it_weighs():
    from isaaclab_arena.utils.usd.mass_properties import read_mass_properties

    stage = new_stage()
    body_path = add_body(stage, "body_01")

    assert read_mass_properties(stage.GetPrimAtPath(body_path)) is None
