# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""HDF5 dataset, group, and attribute name constants for the SyntheticScene format.

Import the whole module as a namespace so every name is reachable through a
single import statement::

    from isaaclab_arena_datagen.io import hdf5_keys as Keys

    cam_group[Keys.COLOR][frame_idx]
    seq_group.attrs[Keys.ATTR_NUM_FRAMES]
"""

# ---------------------------------------------------------------------------
# Per-camera datasets — static modalities
# ---------------------------------------------------------------------------
COLOR = "color"
DEPTH = "depth"
INTRINSIC = "intrinsic"
EXTRINSIC = "extrinsic"
NORMAL = "normal"
SEMANTIC = "semantic"
SEMANTIC_JSON = "semantic_json"

# ---------------------------------------------------------------------------
# Per-camera datasets — adjacent-frame flow  (N-1 rows)
# ---------------------------------------------------------------------------
FLOW2D = "flow2d"
FLOW3D = "flow3d"
FLOW3D_TRACK_TYPE = "flow3d_track_type"

# ---------------------------------------------------------------------------
# Per-camera sub-groups — anchor-frame data  (one dataset per anchor index)
#
# An anchor frame is a designated source frame (identified by its zero-based
# index) from which long-range 3D tracking is computed forward in time.
# For anchor index A and current frame F (F >= A):
#   - FLOW3D_FROM_FRAME[A][F-A]       : 3D scene flow vector field from A to F
#   - TRACKABLE_MASK_FRAME[A][F-A]    : pixels in frame A that can be tracked to F
#   - IN_FRAME_MASK_FRAME[A][F-A]     : pixels in frame A that project inside the image at F
#   - VISIBLE_NOW_MASK_FRAME[A][F-A]  : pixels in frame A that are not occluded at F
# Each sub-group contains one dataset keyed by str(A); the dataset has
# (N - A) rows where N is the total sequence length.
# ---------------------------------------------------------------------------
FLOW3D_FROM_FRAME = "flow3d_from_frame"
TRACKABLE_MASK_FRAME = "trackable_mask_frame"
IN_FRAME_MASK_FRAME = "in_frame_mask_frame"
VISIBLE_NOW_MASK_FRAME = "visible_now_mask_frame"

# ---------------------------------------------------------------------------
# Dynamic-objects groups
# ---------------------------------------------------------------------------
DYNAMIC_OBJECTS = "dynamic_objects"
POSES = "poses"
MESH_SAMPLES = "mesh_samples"

# ---------------------------------------------------------------------------
# Top-level group naming
# ---------------------------------------------------------------------------
SEQUENCE_GROUP_PREFIX = "sequence_"


def sequence_group_name(index: int) -> str:
    """Return the HDF5 group name for a sequence index, e.g. ``0`` -> ``"sequence_000000"``."""
    return f"{SEQUENCE_GROUP_PREFIX}{index:06d}"


# ---------------------------------------------------------------------------
# Sequence-level attributes
# ---------------------------------------------------------------------------
ATTR_SEQUENCE_ID = "sequence_id"
ATTR_NUM_FRAMES = "num_frames"
ATTR_ANCHOR_FRAME_INDICES = "anchor_frame_indices"
ATTR_CAMERA_IDS = "camera_ids"
ATTR_OBJECT_IDS = "object_ids"

# ---------------------------------------------------------------------------
# Camera-group attributes
# ---------------------------------------------------------------------------
ATTR_HEIGHT = "height"
ATTR_WIDTH = "width"

# ---------------------------------------------------------------------------
# Dynamic-objects group attributes
# ---------------------------------------------------------------------------
ATTR_METADATA_JSON = "metadata_json"
