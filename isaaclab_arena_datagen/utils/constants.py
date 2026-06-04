# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Constants shared across the data-generation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena_datagen.object_registry import ObjectType

# ---------------------------------------------------------------------------
# Colour generation
# ---------------------------------------------------------------------------

# Golden ratio conjugate for quasi-random hue generation (produces
# maximally-spaced hues when iterated as ``(offset + phi * i) % 1``).
GOLDEN_RATIO_CONJUGATE = 0.6180339887498948

# Initial hue offset so the first object is not pure red.
HUE_OFFSET = 0.17

# ---------------------------------------------------------------------------
# Motion detection thresholds
# ---------------------------------------------------------------------------
# A body is dynamic if per-step translation OR rotation exceeds its threshold.
# Defaults reject solver jitter on static bodies while staying well below
# the motion produced by any deliberate velocity.

# Min per-step translation (m) to count as moving; ~0.1 mm.
DEFAULT_TRANSLATION_EPS_M = 2e-4

# Min per-step rotation (rad) to count as moving; ~0.057 deg.
DEFAULT_ROTATION_EPS_RAD = 1e-3

# ---------------------------------------------------------------------------
# Mesh surface sampling limits
# ---------------------------------------------------------------------------

# Cap on points sampled per dynamic-object mesh. Raw count is
# ``mesh.area / spacing_m ** 2``; well-scaled assets stay under the cap.
# Some Objaverse USDs have pathological mesh scale that would otherwise
# hang ``DynamicObjectTracker._sample_on_mesh`` (O(n * k) FPS) indefinitely.
MAX_MESH_SAMPLE_POINTS = 10000

# ---------------------------------------------------------------------------
# Object type display prefixes
# ---------------------------------------------------------------------------


def get_type_prefix_map() -> dict[ObjectType, str]:
    """Return the mapping from ObjectType to display-name prefix.

    Deferred import avoids a circular dependency between constants and
    object_registry.
    """
    from isaaclab_arena_datagen.object_registry import ObjectType

    return {
        ObjectType.RIGID: "rigid_object",
        ObjectType.ARTICULATION: "articulated_object",
        ObjectType.STATIC: "static_object",
        ObjectType.UNSUPPORTED: "unsupported_object",
    }


# ---------------------------------------------------------------------------
# Canonical subfolder names (shared by writer and visualizer)
# ---------------------------------------------------------------------------

SUBFOLDER_COLOR = "color"
SUBFOLDER_DEPTH = "depth"
SUBFOLDER_FLOW2D = "flow2d"
SUBFOLDER_FLOW3D = "flow3d"
SUBFOLDER_FLOW3D_TRACK_TYPE = "flow3d_track_type"
SUBFOLDER_NORMAL = "normal"
SUBFOLDER_EXTRINSIC = "extrinsic"
SUBFOLDER_INTRINSIC = "intrinsic"
SUBFOLDER_SEMANTIC = "semantic"
