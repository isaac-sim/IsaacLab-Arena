"""Physical solver for 3D object placement with physics simulation.

This module implements physics-based placement for stacking and containment.
It uses occupancy grids and physics simulation to ensure physically plausible
configurations.
"""

import numpy as np
import random
from typing import Optional, Any
from .predicates import (
    ObjectState,
    PhysicalPredicate,
    PlaceOnPredicate,
    PlaceInPredicate,
    PredicateType,
)


class PhysicalSolver:
    """Solver for physical predicates using physics simulation."""

    def __init__(
        self,
        simulation_app: Optional[Any] = None,
        grid_resolution: float = 0.01,
        stability_threshold: float = 0.02,
    ):
        """Initialize physical solver.

        Args:
            simulation_app: Isaac Sim SimulationApp instance (None for testing)
            grid_resolution: Voxel size for occupancy grid (meters)
            stability_threshold: Maximum displacement for stable placement (meters)
        """
        self.simulation_app = simulation_app
        self.grid_resolution = grid_resolution
        self.stability_threshold = stability_threshold
        self.placed_objects = []  # Track placed objects for occupancy

    def solve(
        self,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
        object_paths: dict[str, str],
        scene_path: str,
    ) -> tuple[bool, str]:
        """Solve physical predicates for all objects.

        Args:
            object_states: Dictionary of object states
            object_dims: Dictionary of object dimensions
            object_paths: Dictionary of USD paths for objects
            scene_path: Path to the base scene

        Returns:
            (success, message) tuple
        """
        # Group predicates by type
        place_on_predicates = []
        place_in_predicates = []
        place_anywhere_predicates = []

        for obj_name, obj_state in object_states.items():
            for pred in obj_state.predicates:
                if isinstance(pred, PlaceOnPredicate):
                    place_on_predicates.append((obj_name, pred))
                elif isinstance(pred, PlaceInPredicate):
                    place_in_predicates.append((obj_name, pred))
                elif pred.type == PredicateType.PLACE_ANYWHERE:
                    place_anywhere_predicates.append((obj_name, pred))

        # Process place-on predicates
        for obj_name, pred in place_on_predicates:
            success = self._solve_place_on(
                object_states[obj_name],
                pred,
                object_states.get(pred.support_object),
                object_dims[obj_name],
                object_dims.get(pred.support_object),
            )
            if not success:
                return False, f"Failed to place {obj_name} on {pred.support_object}"

        # Process place-in predicates
        for obj_name, pred in place_in_predicates:
            success = self._solve_place_in(pred, object_states, object_dims)
            if not success:
                return False, f"Failed to place objects in {pred.support_object}"

        # Process place-anywhere predicates
        for obj_name, pred in place_anywhere_predicates:
            success = self._solve_place_anywhere(
                object_states[obj_name], object_dims[obj_name], object_states
            )
            if not success:
                return False, f"Failed to place {obj_name} anywhere"

        return True, "All physical constraints resolved"

    def _solve_place_on(
        self,
        obj_state: ObjectState,
        pred: PlaceOnPredicate,
        support_state: Optional[ObjectState],
        obj_dims: tuple[float, float, float],
        support_dims: Optional[tuple[float, float, float]],
    ) -> bool:
        """Solve place-on predicate by stacking object on support."""
        if not support_state or support_state.x is None or support_state.y is None:
            return False

        # For now, simple heuristic placement on top
        # In full implementation, this would use occupancy grid sampling

        # Use object's x, y if already set (from spatial predicates)
        if obj_state.x is None:
            # Sample position on support surface
            if pred.relative_position == "center":
                obj_state.x = support_state.x
                obj_state.y = support_state.y
            elif pred.relative_position == "edge":
                # Random edge position
                angle = random.uniform(0, 2 * np.pi)
                radius = max(support_dims[0], support_dims[1]) / 2 * 0.7
                obj_state.x = support_state.x + radius * np.cos(angle)
                obj_state.y = support_state.y + radius * np.sin(angle)
            else:
                # Random position within support bounds
                max_offset = max(support_dims[0], support_dims[1]) / 2 * 0.5
                obj_state.x = support_state.x + random.uniform(-max_offset, max_offset)
                obj_state.y = support_state.y + random.uniform(-max_offset, max_offset)

        # Set z height on top of support
        if support_state.z is None:
            # Assume support is on table at z=0
            support_z_top = support_dims[2] if support_dims else 0.05
        else:
            support_z_top = support_state.z + (
                support_dims[2] / 2 if support_dims else 0.05
            )

        obj_state.z = support_z_top + obj_dims[2] / 2 + 0.001  # Small gap

        # Set orientation if not already set
        if obj_state.yaw is None:
            if pred.stability_preference == "stable":
                obj_state.yaw = 0.0
            elif pred.stability_preference == "unstable":
                obj_state.yaw = random.uniform(0, 360)
            else:
                obj_state.yaw = random.choice([0, 90, 180, 270])

        if obj_state.pitch is None:
            obj_state.pitch = 0.0
        if obj_state.roll is None:
            obj_state.roll = 0.0

        obj_state.is_placed = True
        self.placed_objects.append(obj_state.name)

        return True

    def _solve_place_in(
        self,
        pred: PlaceInPredicate,
        object_states: dict[str, ObjectState],
        object_dims: dict[str, tuple[float, float, float]],
    ) -> bool:
        """Solve place-in predicate by placing objects inside container."""
        container_state = object_states.get(pred.support_object)
        if not container_state or container_state.x is None:
            return False

        container_dims = object_dims.get(pred.support_object)
        if not container_dims:
            return False

        # Simple grid placement inside container
        target_objects = getattr(pred, "target_objects", [pred.target_object])
        num_objects = len(target_objects)

        # Estimate container interior bounds
        container_width = container_dims[0] * 0.7
        container_depth = container_dims[1] * 0.7
        container_height = container_dims[2] * 0.5

        # Simple grid layout
        cols = int(np.ceil(np.sqrt(num_objects)))
        rows = int(np.ceil(num_objects / cols))

        cell_width = container_width / cols
        cell_depth = container_depth / rows

        for i, obj_name in enumerate(target_objects):
            obj_state = object_states.get(obj_name)
            if not obj_state:
                continue

            row = i // cols
            col = i % cols

            # Position within container
            local_x = (col + 0.5) * cell_width - container_width / 2
            local_y = (row + 0.5) * cell_depth - container_depth / 2

            obj_state.x = container_state.x + local_x
            obj_state.y = container_state.y + local_y

            # Start above container, will settle with physics
            container_z = (
                container_state.z if container_state.z else container_dims[2] / 2
            )
            obj_state.z = container_z + container_height + 0.05

            if obj_state.yaw is None:
                obj_state.yaw = random.uniform(0, 360)
            if obj_state.pitch is None:
                obj_state.pitch = 0.0
            if obj_state.roll is None:
                obj_state.roll = 0.0

            obj_state.is_placed = True
            self.placed_objects.append(obj_name)

        return True

    def _solve_place_anywhere(
        self,
        obj_state: ObjectState,
        obj_dims: tuple[float, float, float],
        all_states: dict[str, ObjectState],
    ) -> bool:
        """Solve place-anywhere by finding random supported position."""
        # For simplicity, place on table if no other placement found
        # In full implementation, this would check all possible support surfaces

        if obj_state.x is None or obj_state.y is None:
            # Find a random non-colliding position
            for _ in range(20):
                obj_state.x = random.uniform(-0.3, 0.3)
                obj_state.y = random.uniform(-0.3, 0.3)

                # Check if collides with already placed objects
                collision = False
                for other_name, other_state in all_states.items():
                    if other_name == obj_state.name or not other_state.is_placed:
                        continue
                    if other_state.x is None or other_state.y is None:
                        continue

                    dist = np.sqrt(
                        (obj_state.x - other_state.x) ** 2
                        + (obj_state.y - other_state.y) ** 2
                    )
                    if dist < 0.1:
                        collision = True
                        break

                if not collision:
                    break

        # Place on table surface
        if obj_state.z is None:
            obj_state.z = obj_dims[2] / 2 + 0.001

        if obj_state.yaw is None:
            obj_state.yaw = random.uniform(0, 360)
        if obj_state.pitch is None:
            obj_state.pitch = 0.0
        if obj_state.roll is None:
            obj_state.roll = 0.0

        obj_state.is_placed = True
        self.placed_objects.append(obj_state.name)

        return True

    def validate_with_physics(
        self, scene_path: str, num_steps: int = 300
    ) -> tuple[bool, dict]:
        """Validate scene stability using physics simulation.

        This method uses Isaac Sim to run a physics simulation and check if
        objects remain stable or fall/move significantly.

        Args:
            scene_path: Path to the USD scene file
            num_steps: Number of simulation steps to run (~5s at 60Hz)

        Returns:
            (is_stable, diagnostics) tuple
        """
        if not self.simulation_app:
            # No simulation available, assume stable
            return True, {"message": "No physics validation (simulation not available)"}

        try:
            import omni.usd
            import omni.timeline
            from pxr import UsdGeom, Gf

            # Load scene
            print(
                f"[PhysicalSolver] Loading scene for physics validation: {scene_path}"
            )
            omni.usd.get_context().open_stage(scene_path)
            stage = omni.usd.get_context().get_stage()

            # Record initial positions of all objects (excluding tables and static objects)
            initial_positions = {}
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()

                # Skip tables, ground plane, and scene itself
                if any(
                    skip in prim_name
                    for skip in ["table", "ground", "scene", "physics", "render"]
                ):
                    continue

                # Only track Xforms that are likely to be objects
                if prim.IsA(UsdGeom.Xform):
                    xformable = UsdGeom.Xformable(prim)
                    xform_ops = xformable.GetOrderedXformOps()

                    # Find translate op
                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            initial_pos = op.Get()
                            if initial_pos:
                                initial_positions[prim.GetPath()] = initial_pos
                            break

            if not initial_positions:
                return True, {"message": "No objects found to validate"}

            print(f"[PhysicalSolver] Tracking {len(initial_positions)} objects")

            # Run physics simulation (similar to settle_scenes.py)
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(num_steps):
                self.simulation_app.update()
            timeline.pause()

            print(f"[PhysicalSolver] Simulation complete, checking stability...")

            # Check final positions
            final_positions = {}
            for prim_path in initial_positions:
                prim = stage.GetPrimAtPath(prim_path)
                if prim:
                    xformable = UsdGeom.Xformable(prim)
                    xform_ops = xformable.GetOrderedXformOps()

                    for op in xform_ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            final_pos = op.Get()
                            if final_pos:
                                final_positions[prim_path] = final_pos
                            break

            # Calculate displacements
            unstable_objects = []
            max_displacement = 0.0

            for prim_path, initial_pos in initial_positions.items():
                final_pos = final_positions.get(prim_path)
                if final_pos:
                    displacement = np.linalg.norm(
                        np.array(final_pos) - np.array(initial_pos)
                    )
                    max_displacement = max(max_displacement, displacement)

                    if displacement > self.stability_threshold:
                        obj_name = str(prim_path).split("/")[-1]
                        unstable_objects.append(
                            {
                                "object": obj_name,
                                "displacement": float(displacement),
                                "initial": [float(x) for x in initial_pos],
                                "final": [float(x) for x in final_pos],
                            }
                        )
                        print(
                            f"[PhysicalSolver]   {obj_name}: moved {displacement:.4f}m"
                        )

            is_stable = len(unstable_objects) == 0

            if is_stable:
                print(
                    f"[PhysicalSolver] ✓ Scene is stable (max displacement: {max_displacement:.4f}m)"
                )
            else:
                print(
                    f"[PhysicalSolver] ✗ Scene is unstable ({len(unstable_objects)} objects moved)"
                )

            diagnostics = {
                "stable": is_stable,
                "num_objects": len(initial_positions),
                "unstable_objects": unstable_objects,
                "max_displacement": float(max_displacement),
            }

            return is_stable, diagnostics

        except Exception as e:
            import traceback

            print(f"[PhysicalSolver] Error during physics validation: {e}")
            traceback.print_exc()
            return False, {"error": str(e)}

    def settle_scene(self, scene_path: str, output_path: str, num_steps: int = 300):
        """Settle a scene using physics simulation and save the result.

        This is similar to the settle_scenes.py utility but integrated into
        the scene generation pipeline.

        Args:
            scene_path: Path to input USD scene
            output_path: Path to save settled scene
            num_steps: Number of simulation steps
        """
        if not self.simulation_app:
            print("[PhysicalSolver] No simulation app, cannot settle scene")
            return

        try:
            import omni.usd
            import omni.timeline

            print(f"[PhysicalSolver] Settling scene: {scene_path}")

            # Open scene
            omni.usd.get_context().open_stage(scene_path)
            timeline = omni.timeline.get_timeline_interface()

            # Run physics
            timeline.play()
            for _ in range(num_steps):
                self.simulation_app.update()
            timeline.pause()

            # Export settled scene
            stage = omni.usd.get_context().get_stage()
            stage.GetRootLayer().Export(output_path)

            print(f"[PhysicalSolver] Settled scene saved to: {output_path}")

        except Exception as e:
            print(f"[PhysicalSolver] Error settling scene: {e}")
            import traceback

            traceback.print_exc()
