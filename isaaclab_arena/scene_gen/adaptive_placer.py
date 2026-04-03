"""Adaptive object placer combining RoboLab's spatial solver with Arena objects.

Uses RoboLab's proven circle-based collision resolver for LLM-generated
positions (place-on-base), then uses Arena's differentiable solver for
relative predicates (place-in, place-on, left-of, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab_arena.relations.relations import (
    Inside, IsAnchor, On, RotateAroundSolution, RandomAroundSolution,
)
from isaaclab_arena.scene_gen.predicates import (
    ObjectState, PredicateType, PlaceOnPredicate, PlaceInPredicate,
)
from isaaclab_arena.scene_gen.spatial_solver import SpatialSolver
from isaaclab_arena.scene_gen.arena_asset_manager import DEFAULT_TABLE_BOUNDS
from isaaclab_arena.utils.pose import Pose


DEFAULT_CLEARANCE_M = 0.02


@dataclass
class AdaptivePlacementResult:
    success: bool
    positions: dict  # name -> (x, y, z)
    final_loss: float
    attempts: int


def place_objects_adaptive(
    objects: list,
    table_asset,
    asset_manager,
    table_bounds: tuple[float, float, float, float] | None = None,
    table_top_z: float = 0.35,
    verbose: bool = False,
) -> AdaptivePlacementResult:
    """Place objects using RoboLab's spatial solver for collision resolution.

    Approach:
    1. Build ObjectState for each object from LLM positions (_llm_position)
    2. Run RoboLab's SpatialSolver to resolve collisions (push-apart)
    3. Compute Z from table height + object half-height
    4. Set initial_pose on each Arena Object
    5. Handle relative predicates (place-in, place-on) separately

    Args:
        objects: List of Arena Object instances with _llm_position and relations.
        table_asset: The table Object (IsAnchor).
        asset_manager: ArenaAssetManager for dims lookup.
        table_bounds: (min_x, max_x, min_y, max_y). Uses default if None.
        table_top_z: Z height of the table surface.
        verbose: Print solver progress.

    Returns:
        AdaptivePlacementResult.
    """
    if table_bounds is None:
        table_bounds = DEFAULT_TABLE_BOUNDS

    # Categorize objects by placement type
    table_objects = []     # On table (place-on-base, left-of, etc.) → spatial solver
    stacking_objects = []  # On another object (place-on) → Z computed from support
    container_objects = [] # Inside a container (place-in) → XY from container, Z inside

    for obj in objects:
        if any(isinstance(r, IsAnchor) for r in obj.get_relations()):
            continue  # Skip anchors

        state = getattr(obj, '_object_state', None)
        if state is None:
            table_objects.append(obj)
            continue

        # Check predicates to categorize
        has_place_on = any(isinstance(p, PlaceOnPredicate) for p in state.predicates)
        has_place_in = any(isinstance(p, PlaceInPredicate) for p in state.predicates)

        if has_place_in:
            container_objects.append(obj)
        elif has_place_on:
            stacking_objects.append(obj)
        else:
            table_objects.append(obj)

    if verbose:
        print(f"[AdaptivePlacer] {len(table_objects)} on-table, "
              f"{len(stacking_objects)} stacking, {len(container_objects)} in-container")

    # --- Phase 1: Build ObjectStates from LLM positions ---
    object_states = {}
    object_dims = {}

    import random as rng

    for obj in table_objects:
        llm = getattr(obj, '_llm_position', {})
        x = llm.get('x')
        y = llm.get('y')

        # If no LLM position, assign random position within table bounds
        if x is None:
            x = rng.uniform(table_bounds[0] + 0.05, table_bounds[1] - 0.05)
        if y is None:
            y = rng.uniform(table_bounds[2] + 0.05, table_bounds[3] - 0.05)

        # Get dims from asset manager
        dims = asset_manager.get_object_dims(obj.name)
        if dims is None:
            # Compute from bounding box
            bbox = obj.get_bounding_box()
            dims = (
                bbox.max_point[0] - bbox.min_point[0],
                bbox.max_point[1] - bbox.min_point[1],
                bbox.max_point[2] - bbox.min_point[2],
            )

        object_dims[obj.name] = dims

        state = ObjectState(name=obj.name)
        state.x = x
        state.y = y
        state.yaw = None  # Will be set by rotation relations
        state.predicates = []  # Not used by spatial solver directly

        # Check for rotation
        for rel in obj.get_relations():
            if isinstance(rel, RotateAroundSolution):
                state.yaw = math.degrees(rel._yaw_rad) if hasattr(rel, '_yaw_rad') else 0.0

        if state.yaw is None:
            import random
            state.yaw = random.uniform(0, 360)

        object_states[obj.name] = state

    # --- Phase 2: Run RoboLab spatial solver for collision resolution ---
    if object_states:
        solver = SpatialSolver(table_bounds)
        success, message = solver.solve(object_states, object_dims)

        if verbose:
            print(f"[SpatialSolver] {message}")
            if not success:
                print(f"[SpatialSolver] Applying best positions anyway")

    # --- Phase 3: Compute Z and set poses for table objects ---
    positions = {}

    for obj in table_objects:
        name = obj.name
        if name not in object_states:
            continue

        state = object_states[name]
        dims = object_dims.get(name, (0.05, 0.05, 0.05))
        half_height = dims[2] / 2.0
        z = table_top_z + half_height + DEFAULT_CLEARANCE_M

        x = state.x if state.x is not None else 0.55
        y = state.y if state.y is not None else 0.0
        yaw_rad = math.radians(state.yaw) if state.yaw is not None else 0.0

        positions[name] = (x, y, z)
        rot = _yaw_to_quat(yaw_rad)
        obj.set_initial_pose(Pose(position_xyz=(x, y, z), rotation_wxyz=rot))

    # --- Phase 3b: Handle place-on (stacking on another object) ---
    for obj in stacking_objects:
        state = obj._object_state
        for pred in state.predicates:
            if isinstance(pred, PlaceOnPredicate):
                support_name = pred.support_object
                if support_name in positions:
                    sx, sy, sz = positions[support_name]
                    support_dims = object_dims.get(support_name, (0.1, 0.1, 0.1))
                    support_top_z = sz + support_dims[2] / 2.0

                    obj_bbox = obj.get_bounding_box()
                    obj_half_h = (obj_bbox.max_point[2] - obj_bbox.min_point[2]) / 2.0

                    # Stack: center on support, Z on top
                    stack_z = support_top_z + obj_half_h + 0.001
                    yaw_rad = math.radians(state.yaw) if state.yaw else 0.0

                    positions[obj.name] = (sx, sy, stack_z)
                    obj.set_initial_pose(Pose(
                        position_xyz=(sx, sy, stack_z),
                        rotation_wxyz=_yaw_to_quat(yaw_rad),
                    ))
                    if verbose:
                        print(f"  [Stack] {obj.name} on {support_name} at z={stack_z:.3f}")
                else:
                    # Fallback: place on table
                    obj_bbox = obj.get_bounding_box()
                    obj_half_h = (obj_bbox.max_point[2] - obj_bbox.min_point[2]) / 2.0
                    z = table_top_z + obj_half_h + DEFAULT_CLEARANCE_M
                    positions[obj.name] = (0.55, 0.0, z)
                    obj.set_initial_pose(Pose(position_xyz=(0.55, 0.0, z)))
                break

    # --- Phase 4: Handle place-in (inside container) ---
    for obj in container_objects:
        state = obj._object_state
        for pred in state.predicates:
            if isinstance(pred, PlaceInPredicate):
                container_name = pred.support_object
                if container_name in positions:
                    cx, cy, cz = positions[container_name]
                    container_dims = object_dims.get(container_name)

                    # Place inside: same XY, Z inside container
                    if container_dims:
                        # Container bottom = cz - container_half_height
                        # Inner Z = bottom + small clearance + obj_half_height
                        container_half_h = container_dims[2] / 2.0
                        container_bottom_z = cz - container_half_h
                        inner_z = container_bottom_z + container_dims[2] * 0.2
                    else:
                        inner_z = cz

                    obj_bbox = obj.get_bounding_box()
                    obj_half_h = (obj_bbox.max_point[2] - obj_bbox.min_point[2]) / 2.0

                    final_z = inner_z + obj_half_h
                    positions[obj.name] = (cx, cy, final_z)
                    obj.set_initial_pose(Pose(
                        position_xyz=(cx, cy, final_z),
                        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                    ))
                    if verbose:
                        print(f"  [Inside] {obj.name} in {container_name} at z={final_z:.3f}")
                break

    success = True
    if verbose:
        print(f"\n[AdaptivePlacer] Placed {len(positions)} objects")
        for name, (x, y, z) in sorted(positions.items()):
            print(f"  {name:30s} -> ({x:.3f}, {y:.3f}, {z:.3f})")

    return AdaptivePlacementResult(
        success=success,
        positions=positions,
        final_loss=0.0,
        attempts=1,
    )


def _yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert yaw angle (radians) to wxyz quaternion."""
    w = math.cos(yaw_rad / 2.0)
    z = math.sin(yaw_rad / 2.0)
    return (w, 0.0, 0.0, z)
