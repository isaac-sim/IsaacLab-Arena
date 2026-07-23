# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serialization and geometry tests for ReachabilityMap and the checked-in droid map.

Exercises save/load round-tripping and voxel-grid indexing on CPU with a hand-built map, then sanity-
checks the committed droid map. No cuRobo or GPU needed; computing a fresh map is covered separately.
"""

from __future__ import annotations

import numpy as np
import os

from isaaclab_arena_curobo.reachability_map import ReachabilityMap

_DROID_MAP = os.path.join(os.path.dirname(__file__), "data", "reachability_map_droid.npz")


def _tiny_map() -> ReachabilityMap:
    """A 2x2x2 map with a known reachable pattern for exact-value assertions."""
    reachable = np.zeros((2, 2, 2), dtype=bool)
    reachable[0, 0, 0] = True
    reachable[1, 1, 1] = True
    return ReachabilityMap(
        embodiment_name="unit",
        grid_min_xyz=(0.0, -0.1, 0.2),
        resolution_m=0.1,
        reachable=reachable,
        grasp_z_offset=0.0,
        ik_pos_threshold=0.01,
        ik_rot_threshold=0.1,
    )


def test_points_and_mask_align_with_grid():
    """Voxel (i, j, k) center is grid_min + (i, j, k) * resolution, and the mask row matches reachable."""
    m = _tiny_map()
    points = m.points_base_frame()
    mask = m.reachable_mask_flat()
    assert points.shape == (8, 3)
    assert mask.shape == (8,)
    # Row 0 is voxel (0, 0, 0) at the grid min; the last row is voxel (1, 1, 1).
    np.testing.assert_allclose(points[0], [0.0, -0.1, 0.2])
    np.testing.assert_allclose(points[-1], [0.1, 0.0, 0.3])
    assert bool(mask[0]) is True and bool(mask[-1]) is True
    assert int(mask.sum()) == 2


def test_save_load_round_trip(tmp_path):
    """Saving then loading reproduces every field, including the exact reachable pattern."""
    m = _tiny_map()
    path = str(tmp_path / "sub" / "map.npz")  # nested dir exercises save()'s mkdir
    m.save(path)
    loaded = ReachabilityMap.load(path)
    assert loaded.embodiment_name == "unit"
    assert loaded.shape == (2, 2, 2)
    np.testing.assert_allclose(loaded.grid_min_xyz, (0.0, -0.1, 0.2), atol=1e-6)
    assert abs(loaded.resolution_m - 0.1) < 1e-6
    np.testing.assert_array_equal(loaded.reachable, m.reachable)


def test_committed_droid_map_is_well_formed():
    """The checked-in droid map loads, is droid, and has a plausible non-trivial reachable fraction."""
    m = ReachabilityMap.load(_DROID_MAP)
    assert m.embodiment_name == "droid"
    assert m.reachable.dtype == bool
    assert len(m.shape) == 3
    reachable_fraction = m.reachable.mean()
    # A free-space arm reaches a good chunk of the sampled box but not all of it.
    assert 0.05 < reachable_fraction < 0.99
