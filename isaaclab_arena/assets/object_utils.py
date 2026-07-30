# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from pxr import Usd

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.usd.physics_structure import get_physics_structure_from_usd
from isaaclab_arena.utils.usd_helpers import (
    apply_usd_variant_selections,
    get_prim_depth,
    is_articulation_root,
    is_rigid_body,
)


def _detect_type_from_joints(stage: Usd.Stage, found_depth: int) -> ObjectType:
    """Decide the type of an asset that has several rigid bodies side by side.

    The number of bodies alone does not say much: SimReady props give each part its own body and
    join the parts back together. Only the joints say whether the parts can still move.
    """
    structure = get_physics_structure_from_usd(stage)
    if structure.is_single_rigid_body:
        # Nothing can move, so the asset behaves like a single rigid body.
        return ObjectType.RIGID
    if structure.moving_joints:
        return ObjectType.ARTICULATION
    raise ValueError(
        f"Found {len(structure.rigid_body_paths)} rigid bodies at depth {found_depth} in"
        f" {len(structure.body_groups)} groups that are not connected to each other:"
        f" {structure.body_groups}. Expected one rigid body, bodies held together by fixed joints,"
        " or an articulation."
    )


def detect_object_type(
    usd_path: str | None = None,
    stage: Usd.Stage | None = None,
    variants: dict[str, str] | None = None,
) -> ObjectType:
    """Detect the object type of the asset

    Goes through the USD tree and detects the object type. The detection is based
    on the presence of a RigidBodyAPI or ArticulationRootAPI at the shallowest depth
    in which one of these APIs is present.

    When that depth holds several rigid bodies, the joints decide: bodies that fixed joints hold
    together count as one rigid body, and anything with a joint that still moves is an
    articulation.

    Args:
        usd_path: The path to the USD file to inspect. Either this or stage must be provided.
        stage: The stage to inspect. Either this or usd_path must be provided.
        variants: USD variants to select before looking at the asset. SimReady props need
            ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        The object type of the asset.
    """
    assert usd_path is not None or stage is not None, "Either usd_path or stage must be provided"
    assert usd_path is None or stage is None, "Either usd_path or stage must be provided"
    if usd_path is not None:
        # Open a stage to inspect the USD.
        stage = Usd.Stage.Open(usd_path)
    apply_usd_variant_selections(stage, variants)
    # We do a Breadth First Search (BFS) through the prims, until we find either
    # a rigid body or an articulation root. At that point, we continue searching
    # the rest of the prims at that depth, to collect everything else at that depth.
    open_prims = [stage.GetPseudoRoot()]
    found_depth = -1
    found_prims = []
    while len(open_prims) > 0:
        # Update the DFS list
        prim = open_prims.pop(0)
        open_prims.extend(prim.GetChildren())
        if found_prims and get_prim_depth(prim) > found_depth:
            break
        # Check if we found an interesting prim on this level
        if is_articulation_root(prim) or is_rigid_body(prim):
            found_depth = get_prim_depth(prim)
            found_prims.append(prim)
    if not found_prims:
        return ObjectType.BASE
    if len(found_prims) == 1:
        return ObjectType.RIGID if is_rigid_body(found_prims[0]) else ObjectType.ARTICULATION
    if any(is_articulation_root(prim) for prim in found_prims):
        raise ValueError(f"Found multiple rigid body or articulation roots at depth {found_depth}")
    return _detect_type_from_joints(stage, found_depth)


# Predefined rigid body property configurations for assembly tasks
# High iteration count for precision tasks (peg/hole insertion)
RIGID_BODY_PROPS_HIGH_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

# Standard iteration count for gear mesh tasks
RIGID_BODY_PROPS_MEDIUM_PRECISION = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=32,
    max_contact_impulse=1e32,
)

# Initial state configuration for articulations without joints (e.g., rigid bodies treated as articulations).
# We explicitly set joint_pos and joint_vel to empty dicts to avoid the default pattern {".*": 0.0} in ArticulationCfg.InitialStateCfg,
# which would fail to match when there are no joints in the articulation.
EMPTY_ARTICULATION_INIT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={},
    joint_vel={},
)
