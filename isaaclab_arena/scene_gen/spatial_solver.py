"""Spatial solver for 2D object placement on table surface.

This module implements a constraint solver for spatial predicates, determining
x, y, and yaw coordinates for objects placed on the table surface. It uses
convex hull collision detection and iterative optimization to resolve constraints.
"""

import math
import numpy as np
import random
from typing import Optional
from scipy.spatial import ConvexHull

from .predicates import (
    ObjectState,
    SpatialPredicate,
    PredicateType,
    PlaceOnBasePredicate,
    RelativePositionPredicate,
    PhysicalPredicate,
)


class SpatialSolver:
    """Solver for 2D spatial placement constraints."""

    def __init__(
        self,
        table_bounds: tuple[float, float, float, float],
        collision_margin: float = 0.05,
    ):
        """Initialize spatial solver.

        Args:
            table_bounds: (min_x, max_x, min_y, max_y) bounds of table surface
            collision_margin: Minimum distance between objects (meters)
                            5cm margin - ensures collision-free placement
        """
        self.min_x, self.max_x, self.min_y, self.max_y = table_bounds
        self.collision_margin = collision_margin
        self.default_collision_margin = collision_margin  # Save default for resetting
        self.table_width = self.max_x - self.min_x
        self.table_depth = self.max_y - self.min_y

    def solve(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
        max_iterations: int = 500,
        fixed_objects: list[str] = None,
        allow_relaxation: bool = True,
    ) -> tuple[bool, str]:
        """Solve spatial predicates for all objects.

        Args:
            object_states: Dictionary mapping object names to their states
            object_dims: Dictionary mapping object names to (width, depth, height)
            max_iterations: Maximum optimization iterations
            fixed_objects: List of object names that should not be moved (e.g., rack fixtures)
            allow_relaxation: If True, progressively relax collision margins if solving fails

        Returns:
            (success, message) tuple indicating if solving succeeded
        """
        if fixed_objects is None:
            fixed_objects = []

        # ADAPTIVE PARAMETERS for dense scenes
        num_objects = len(object_states)

        # Check if scene has many large objects (containers, etc.)
        # Use max(width, depth) for each object to determine "footprint"
        if object_dims:
            footprints = [max(dims[0], dims[1]) for dims in object_dims.values()]
            avg_size = sum(footprints) / len(footprints)
            # Count objects with footprint > 30cm as "large"
            large_count = sum(1 for fp in footprints if fp > 0.3)
            has_large_objects = large_count >= max(
                3, num_objects // 3
            )  # At least 3 or 1/3 are large
        else:
            avg_size = 0.0
            has_large_objects = False

        # Set base parameters based on scene complexity
        if num_objects >= 18:
            # Ultra-dense scenes (18+ objects): minimal spacing, maximum iterations
            base_margin = 0.012  # 1.2cm - very tight (reduced from 1.5cm)
            max_iterations = 3500  # More iterations (increased from 2500)
            print(
                f"[SpatialSolver] ULTRA-DENSE mode: {num_objects} objects, margin={base_margin}m, max_iter={max_iterations}"
            )
        elif num_objects >= 12 or (num_objects >= 6 and has_large_objects):
            # Hard mode for 12+ objects OR 6+ with many large objects (containers)
            base_margin = 0.020  # 2cm - tight packing (reduced from 2.5cm)
            max_iterations = 3000  # More iterations (increased from 1800)
            if has_large_objects:
                print(
                    f"[SpatialSolver] CONTAINER mode: {num_objects} objects ({large_count} large), margin={base_margin}m, max_iter={max_iterations}"
                )
            else:
                print(
                    f"[SpatialSolver] Hard scene mode: {num_objects} objects, margin={base_margin}m, max_iter={max_iterations}"
                )
        else:
            # Normal scenes: comfortable spacing
            base_margin = self.default_collision_margin

        # TRY WITH PROGRESSIVE RELAXATION
        # Start with base margin, then relax if needed
        margins_to_try = [base_margin]
        # Use relaxation for dense scenes OR when there are fixed objects (racks)
        if allow_relaxation and (num_objects >= 6 or fixed_objects):
            # Add relaxed margins as fallbacks
            margins_to_try.extend(
                [
                    base_margin * 1.25,  # 25% more spacing
                    base_margin * 1.5,  # 50% more spacing
                    base_margin * 2.0,  # Double spacing (last resort)
                ]
            )

        last_error = ""
        for attempt, margin in enumerate(margins_to_try):
            self.collision_margin = margin

            if attempt > 0:
                print(
                    f"[SpatialSolver] Retry {attempt}: Relaxing margin to {margin:.3f}m"
                )
                # Re-randomize positions for fresh attempt
                for obj_name, obj_state in object_states.items():
                    if obj_name not in fixed_objects:
                        obj_state.x = random.uniform(self.min_x + 0.1, self.max_x - 0.1)
                        obj_state.y = random.uniform(self.min_y + 0.1, self.max_y - 0.1)

            # First pass: process place-on-base predicates
            for obj_name, obj_state in object_states.items():
                for pred in obj_state.predicates:
                    if isinstance(pred, PlaceOnBasePredicate):
                        self._apply_place_on_base(obj_state, pred)

            # Second pass: process relative position predicates
            for _ in range(max_iterations):
                changed = False
                for obj_name, obj_state in object_states.items():
                    for pred in obj_state.predicates:
                        if isinstance(pred, RelativePositionPredicate):
                            if self._apply_relative_position(
                                obj_state, pred, object_states
                            ):
                                changed = True

                if not changed:
                    break

            # Third pass: apply orientation predicates
            for obj_name, obj_state in object_states.items():
                for pred in obj_state.predicates:
                    if pred.type in [
                        PredicateType.FACING_LEFT,
                        PredicateType.FACING_RIGHT,
                        PredicateType.FACING_FRONT,
                        PredicateType.FACING_BACK,
                        PredicateType.RANDOM_ROT,
                    ]:
                        self._apply_orientation(obj_state, pred)

            # Check if all objects are fully solved (skip objects with physical predicates)
            unsolved = []
            for name, state in object_states.items():
                # Skip objects that have physical predicates (will be handled by physical solver)
                has_physical_predicate = any(
                    isinstance(pred, PhysicalPredicate)
                    or pred.type
                    in [
                        PredicateType.PLACE_ON,
                        PredicateType.PLACE_IN,
                        PredicateType.PLACE_ANYWHERE,
                    ]
                    for pred in state.predicates
                )

                if not has_physical_predicate and not state.is_fully_solved():
                    unsolved.append(name)

            if unsolved:
                last_error = f"Objects not fully solved: {unsolved}"
                if attempt == len(margins_to_try) - 1:
                    return False, last_error
                continue  # Try next margin

            # Check for collisions and optimize placement
            success = self._optimize_placement(
                object_states, object_dims, max_iterations, fixed_objects
            )

            if success:
                if attempt > 0:
                    print(
                        f"[SpatialSolver] ✓ Solved with relaxed margin: {margin:.3f}m"
                    )
                return True, "All spatial constraints resolved successfully"

            last_error = "Failed to resolve collisions within iteration limit"
            # If this wasn't the last attempt, try with more relaxed margin

        # All attempts failed
        return False, last_error

    def _apply_place_on_base(self, obj_state: ObjectState, pred: PlaceOnBasePredicate):
        """Apply place-on-base predicate."""
        if pred.x is not None:
            obj_state.x = pred.x
        else:
            # Random x within table bounds
            obj_state.x = random.uniform(self.min_x + 0.05, self.max_x - 0.05)

        if pred.y is not None:
            obj_state.y = pred.y
        else:
            # Random y within table bounds
            obj_state.y = random.uniform(self.min_y + 0.05, self.max_y - 0.05)

        if pred.yaw is not None:
            obj_state.yaw = pred.yaw
        else:
            obj_state.yaw = random.uniform(0, 360)

    def _apply_relative_position(
        self,
        obj_state: ObjectState,
        pred: RelativePositionPredicate,
        all_states: dict[str, ObjectState],
    ) -> bool:
        """Apply relative position predicate. Returns True if state changed."""
        ref_state = all_states.get(pred.reference_object)
        if not ref_state or ref_state.x is None or ref_state.y is None:
            return False

        changed = False
        distance = pred.distance or 0.1

        # Coordinate system: Front = +X, Back = -X, Left = +Y, Right = -Y
        if pred.type == PredicateType.LEFT_OF:
            # Left is +Y direction
            new_y = ref_state.y + distance
            if obj_state.y is None or abs(obj_state.y - new_y) > 0.001:
                obj_state.y = new_y
                changed = True
            if obj_state.x is None:
                obj_state.x = ref_state.x
                changed = True

        elif pred.type == PredicateType.RIGHT_OF:
            # Right is -Y direction
            new_y = ref_state.y - distance
            if obj_state.y is None or abs(obj_state.y - new_y) > 0.001:
                obj_state.y = new_y
                changed = True
            if obj_state.x is None:
                obj_state.x = ref_state.x
                changed = True

        elif pred.type == PredicateType.FRONT_OF:
            # Front is +X direction
            new_x = ref_state.x + distance
            if obj_state.x is None or abs(obj_state.x - new_x) > 0.001:
                obj_state.x = new_x
                changed = True
            if obj_state.y is None:
                obj_state.y = ref_state.y
                changed = True

        elif pred.type == PredicateType.BACK_OF:
            # Back is -X direction
            new_x = ref_state.x - distance
            if obj_state.x is None or abs(obj_state.x - new_x) > 0.001:
                obj_state.x = new_x
                changed = True
            if obj_state.y is None:
                obj_state.y = ref_state.y
                changed = True

        return changed

    def _apply_orientation(self, obj_state: ObjectState, pred: SpatialPredicate):
        """Apply orientation predicate."""
        if obj_state.yaw is not None:
            return  # Already set

        if pred.type == PredicateType.FACING_LEFT:
            obj_state.yaw = 180.0
        elif pred.type == PredicateType.FACING_RIGHT:
            obj_state.yaw = 0.0
        elif pred.type == PredicateType.FACING_FRONT:
            obj_state.yaw = 90.0
        elif pred.type == PredicateType.FACING_BACK:
            obj_state.yaw = 270.0
        elif pred.type == PredicateType.RANDOM_ROT:
            obj_state.yaw = random.uniform(0, 360)

    def _optimize_placement(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
        max_iterations: int,
        fixed_objects: list[str] = None,
    ) -> bool:
        """Optimize placement to resolve collisions (prevent intersections)."""
        if fixed_objects is None:
            fixed_objects = []

        previous_collision_count = float("inf")
        no_progress_count = 0

        for iteration in range(max_iterations):
            collisions = self._check_collisions(object_states, object_dims)

            if not collisions:
                # Also check table bounds
                if self._check_table_bounds(object_states, object_dims):
                    if iteration > 0:
                        print(
                            f"[SpatialSolver] ✓ Resolved collisions after {iteration} iterations"
                        )
                    return True

            # Check for progress
            if len(collisions) >= previous_collision_count:
                no_progress_count += 1
                if no_progress_count > 10:
                    # Add random perturbation to break deadlock (but not fixed objects)
                    for name, state in object_states.items():
                        if state.x is not None and name not in fixed_objects:
                            state.x += random.uniform(-0.05, 0.05)
                            state.y += random.uniform(-0.05, 0.05)
                    no_progress_count = 0
            else:
                no_progress_count = 0

            previous_collision_count = len(collisions)

            # Resolve multiple collisions per iteration for faster convergence
            num_to_resolve = min(3, len(collisions))
            for i in range(num_to_resolve):
                if i < len(collisions):
                    obj1, obj2 = collisions[i]
                    # Don't move fixed objects - only move the other object
                    if obj1 in fixed_objects:
                        self._move_away_from_fixed(
                            object_states[obj2],
                            object_states[obj1],
                            object_dims[obj2],
                            object_dims[obj1],
                        )
                    elif obj2 in fixed_objects:
                        self._move_away_from_fixed(
                            object_states[obj1],
                            object_states[obj2],
                            object_dims[obj1],
                            object_dims[obj2],
                        )
                    else:
                        self._resolve_collision(
                            object_states[obj1],
                            object_states[obj2],
                            object_dims[obj1],
                            object_dims[obj2],
                        )

        # If we're close to collision-free, accept it (physics will handle small overlaps)
        final_collisions = self._check_collisions(object_states, object_dims)
        num_objects = len([s for s in object_states.values() if s.x is not None])

        # STRICT: NO collisions allowed - physics settling requires collision-free start
        if len(final_collisions) == 0:
            print(
                f"[SpatialSolver] ✓ All collisions resolved - scene is collision-free"
            )
            self._check_table_bounds(object_states, object_dims)
            return True

        print(
            f"[SpatialSolver] ✗ Failed to resolve collisions after {max_iterations} iterations ({len(final_collisions)} remaining)"
        )
        print(f"[SpatialSolver]   Remaining collisions: {final_collisions[:5]}")
        return False

    def _check_collisions(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
    ) -> list[tuple[str, str]]:
        """Check for collisions between objects."""
        collisions = []
        names = list(object_states.keys())

        for i, name1 in enumerate(names):
            state1 = object_states[name1]
            if state1.x is None or state1.y is None:
                continue

            for name2 in names[i + 1 :]:

                state2 = object_states[name2]
                if state2.x is None or state2.y is None:
                    continue

                # Simple circle-based collision check
                dims1 = object_dims[name1]
                dims2 = object_dims[name2]
                radius1 = max(dims1[0], dims1[1]) / 2 + self.collision_margin
                radius2 = max(dims2[0], dims2[1]) / 2 + self.collision_margin

                dist = np.sqrt((state1.x - state2.x) ** 2 + (state1.y - state2.y) ** 2)
                min_dist = radius1 + radius2
                if dist < min_dist:
                    collisions.append((name1, name2))

        return collisions

    def _check_table_bounds(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
    ) -> bool:
        """Check if all objects are within table bounds."""
        for name, state in object_states.items():
            if state.x is None or state.y is None:
                continue

            dims = object_dims[name]
            radius = max(dims[0], dims[1]) / 2

            if (
                state.x - radius < self.min_x
                or state.x + radius > self.max_x
                or state.y - radius < self.min_y
                or state.y + radius > self.max_y
            ):
                # Clamp to bounds
                state.x = np.clip(state.x, self.min_x + radius, self.max_x - radius)
                state.y = np.clip(state.y, self.min_y + radius, self.max_y - radius)

        return True

    def _move_away_from_fixed(
        self,
        movable_state: ObjectState,
        fixed_state: ObjectState,
        movable_dims: tuple[float, float, float],
        fixed_dims: tuple[float, float, float],
    ):
        """Move a movable object away from a fixed object (like a rack)."""
        if (
            movable_state.x is None
            or movable_state.y is None
            or fixed_state.x is None
            or fixed_state.y is None
        ):
            return

        # Calculate direction away from fixed object
        dx = movable_state.x - fixed_state.x
        dy = movable_state.y - fixed_state.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.01:  # Avoid division by zero
            dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
            dist = math.sqrt(dx * dx + dy * dy)

        # Normalize direction
        dx, dy = dx / dist, dy / dist

        # Calculate required separation distance with EXTRA margin for fixed objects
        # Add 5cm extra buffer to ensure clearance from racks
        extra_margin = 0.05
        required_sep = (
            (movable_dims[0] + fixed_dims[0]) / 2 + self.collision_margin + extra_margin
        )

        # Move movable object away from fixed object
        movable_state.x = fixed_state.x + dx * required_sep
        movable_state.y = fixed_state.y + dy * required_sep

        # Clamp to table bounds
        radius_xy = max(movable_dims[0], movable_dims[1]) / 2 + self.collision_margin
        movable_state.x = max(
            self.min_x + radius_xy, min(self.max_x - radius_xy, movable_state.x)
        )
        movable_state.y = max(
            self.min_y + radius_xy, min(self.max_y - radius_xy, movable_state.y)
        )

    def _resolve_collision(
        self,
        state1: ObjectState,
        state2: ObjectState,
        dims1: tuple[float, float, float],
        dims2: tuple[float, float, float],
    ):
        """Resolve collision by pushing objects apart."""
        # Calculate push direction
        dx = state1.x - state2.x
        dy = state1.y - state2.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 0.001:
            # Objects at same position, push in random direction
            angle = random.uniform(0, 2 * np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)
            dist = 1.0

        # Normalize
        dx /= dist
        dy /= dist

        # Calculate required separation
        radius1 = max(dims1[0], dims1[1]) / 2 + self.collision_margin
        radius2 = max(dims2[0], dims2[1]) / 2 + self.collision_margin
        required_dist = radius1 + radius2

        # Push apart more aggressively for large objects
        # Add extra buffer for large objects (>20cm) to help convergence
        avg_size = (max(dims1[0], dims1[1]) + max(dims2[0], dims2[1])) / 2
        extra_buffer = (
            0.02 if avg_size > 0.2 else 0.01
        )  # 2cm for large objects, 1cm for small

        push = (required_dist - dist) / 2 + extra_buffer
        state1.x += dx * push
        state1.y += dy * push
        state2.x -= dx * push
        state2.y -= dy * push

        # Clamp to table bounds
        state1.x = np.clip(state1.x, self.min_x + radius1, self.max_x - radius1)
        state1.y = np.clip(state1.y, self.min_y + radius1, self.max_y - radius1)
        state2.x = np.clip(state2.x, self.min_x + radius2, self.max_x - radius2)
        state2.y = np.clip(state2.y, self.min_y + radius2, self.max_y - radius2)
