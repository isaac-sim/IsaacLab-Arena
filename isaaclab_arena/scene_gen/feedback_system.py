"""Feedback system for iterative scene refinement.

This module provides feedback to the LLM agent based on solver results,
enabling iterative improvement of scene generation.
"""

from typing import Optional

from .predicates import ObjectState, PhysicalPredicate, PredicateType


class FeedbackSystem:
    """Generates feedback for LLM agent based on solver results."""

    @staticmethod
    def generate_grammar_feedback(
        object_states: dict[str, ObjectState],
    ) -> Optional[str]:
        """Generate feedback about predicate grammar issues.

        Args:
            object_states: Dictionary of object states

        Returns:
            Feedback string if issues found, None otherwise
        """

        issues = []

        for obj_name, obj_state in object_states.items():
            # Check if object has physical predicates that will handle placement
            has_physical_predicate = any(
                isinstance(pred, PhysicalPredicate)
                or pred.type
                in [
                    PredicateType.PLACE_ON,
                    PredicateType.PLACE_IN,
                    PredicateType.PLACE_ANYWHERE,
                ]
                for pred in obj_state.predicates
            )

            # If object has physical predicates, skip spatial completeness check
            # (physical solver will handle x, y, z, yaw)
            if has_physical_predicate:
                continue

            # For spatially-placed objects, check completeness
            if not obj_state.is_fully_solved():
                missing = []
                if obj_state.x is None:
                    missing.append("x coordinate")
                if obj_state.y is None:
                    missing.append("y coordinate")
                if obj_state.yaw is None:
                    missing.append("yaw orientation")

                issues.append(
                    f"Object '{obj_name}' is not fully specified. "
                    f"Missing: {', '.join(missing)}. "
                    f"Please add predicates to determine these values or use a physical predicate."
                )

        if issues:
            return "GRAMMAR ISSUES:\n" + "\n".join(f"- {issue}" for issue in issues)

        return None

    @staticmethod
    def generate_solver_feedback(
        success: bool,
        message: str,
        collisions: Optional[list[tuple[str, str]]] = None,
        out_of_bounds: Optional[list[str]] = None,
    ) -> str:
        """Generate feedback about solver failures.

        Args:
            success: Whether solving succeeded
            message: Solver message
            collisions: List of (obj1, obj2) collision pairs
            out_of_bounds: List of objects outside table bounds

        Returns:
            Feedback string
        """
        if success:
            return "SOLVER SUCCESS: All constraints resolved successfully."

        feedback_parts = ["SOLVER FAILURE:"]
        feedback_parts.append(f"- {message}")

        if collisions:
            feedback_parts.append("\nCOLLISIONS DETECTED:")
            for obj1, obj2 in collisions:
                feedback_parts.append(
                    f"- '{obj1}' and '{obj2}' are overlapping. "
                    f"Please increase distance or reposition."
                )

        if out_of_bounds:
            feedback_parts.append("\nOBJECTS OUT OF BOUNDS:")
            for obj_name in out_of_bounds:
                feedback_parts.append(
                    f"- '{obj_name}' is outside the table surface. "
                    f"Please adjust position or remove object."
                )

        feedback_parts.append(
            "\nFIX THE PROBLEM:\n"
            "- Use EXPLICIT x,y coordinates for ALL objects (not relative positioning)\n"
            "- Space objects at least 0.20m (20cm) apart\n"
            "- Use a simple grid pattern: row 1 at y=0, row 2 at y=0.25, etc.\n"
            "- Stay within SAFE bounds: X=[0.25 to 0.85], Y=[-0.40 to 0.40]"
        )

        return "\n".join(feedback_parts)

    @staticmethod
    def generate_physics_feedback(is_stable: bool, diagnostics: dict) -> str:
        """Generate feedback about physics validation results.

        Args:
            is_stable: Whether scene is physically stable
            diagnostics: Diagnostics from physics validation

        Returns:
            Feedback string
        """
        if is_stable:
            return "PHYSICS VALIDATION SUCCESS: Scene is stable."

        feedback_parts = ["PHYSICS VALIDATION FAILURE:"]

        if "error" in diagnostics:
            feedback_parts.append(f"- Error: {diagnostics['error']}")
            return "\n".join(feedback_parts)

        unstable_objects = diagnostics.get("unstable_objects", [])
        if unstable_objects:
            feedback_parts.append("\nUNSTABLE OBJECTS (fell or moved significantly):")
            for obj_info in unstable_objects:
                obj_name = obj_info["object"]
                displacement = obj_info["displacement"]
                feedback_parts.append(
                    f"- '{obj_name}' moved {displacement:.3f}m after simulation"
                )

        feedback_parts.append(
            "\nSUGGESTIONS:\n"
            "- For stacked objects, increase support_ratio (0.5-0.8 is more stable)\n"
            "- Prefer 'stable' stability preference for better results\n"
            "- Ensure support objects are large enough to support stacked objects\n"
            "- Avoid stacking too many layers\n"
            "- Place unstable objects inside containers instead"
        )

        return "\n".join(feedback_parts)

    @staticmethod
    def generate_scene_evaluation(
        object_states: dict[str, ObjectState],
        table_bounds: tuple[float, float, float, float],
    ) -> dict:
        """Evaluate the quality of a generated scene.

        Args:
            object_states: Dictionary of object states
            table_bounds: (min_x, max_x, min_y, max_y)

        Returns:
            Dictionary with evaluation metrics
        """
        min_x, max_x, min_y, max_y = table_bounds
        table_area = (max_x - min_x) * (max_y - min_y)

        # Calculate coverage
        num_placed = sum(1 for state in object_states.values() if state.is_placed)

        # Calculate compactness (how clustered objects are)
        placed_positions = [
            (state.x, state.y)
            for state in object_states.values()
            if state.is_placed and state.x is not None
        ]

        compactness = 0.0
        if len(placed_positions) > 1:
            import numpy as np

            center = np.mean(placed_positions, axis=0)
            distances = [
                np.linalg.norm(np.array(pos) - center) for pos in placed_positions
            ]
            compactness = 1.0 / (1.0 + np.mean(distances))

        # Check for container presence
        has_container = any(
            "bowl" in state.name.lower()
            or "bin" in state.name.lower()
            or "box" in state.name.lower()
            or "container" in state.name.lower()
            for state in object_states.values()
            if state.is_placed
        )

        # Diversity score based on object types
        object_types = set(
            state.name.split("_")[0]
            for state in object_states.values()
            if state.is_placed
        )
        diversity = len(object_types) / max(num_placed, 1)

        return {
            "num_objects": num_placed,
            "compactness": float(compactness),
            "has_container": has_container,
            "diversity": float(diversity),
            "coverage": float(
                num_placed / max((table_area * 100), 1)
            ),  # Rough estimate
        }

    @staticmethod
    def generate_success_feedback(evaluation: dict) -> str:
        """Generate feedback for successful scene generation.

        Args:
            evaluation: Scene evaluation metrics

        Returns:
            Feedback string with scene quality assessment
        """
        feedback_parts = ["SCENE GENERATION SUCCESS!"]
        feedback_parts.append(f"\nScene Quality Metrics:")
        feedback_parts.append(f"- Number of objects: {evaluation['num_objects']}")
        feedback_parts.append(f"- Compactness: {evaluation['compactness']:.2f}")
        feedback_parts.append(
            f"- Has container: {'Yes' if evaluation['has_container'] else 'No'}"
        )
        feedback_parts.append(f"- Diversity: {evaluation['diversity']:.2f}")

        # Provide qualitative assessment
        feedback_parts.append("\nQualitative Assessment:")

        if evaluation["num_objects"] < 3:
            feedback_parts.append(
                "- Scene is sparse. Consider adding more objects for richer tasks."
            )
        elif evaluation["num_objects"] > 15:
            feedback_parts.append(
                "- Scene is dense. Good for complex manipulation challenges."
            )
        else:
            feedback_parts.append("- Scene has good object density.")

        if not evaluation["has_container"]:
            feedback_parts.append(
                "- Warning: No container present. Consider adding one for task variety."
            )

        if evaluation["compactness"] < 0.3:
            feedback_parts.append("- Objects are well-distributed across the table.")
        elif evaluation["compactness"] > 0.7:
            feedback_parts.append("- Objects are tightly clustered together.")

        return "\n".join(feedback_parts)
