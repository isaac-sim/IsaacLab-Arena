"""Validate LLM-generated task specs before saving.

Checks:
1. Required fields present
2. Object names exist in the scene
3. Size feasibility (inner object fits in outer)
4. Predicate validity
"""

from __future__ import annotations

from typing import Any

from isaaclab_arena.task_gen.build_prompt import AVAILABLE_PREDICATES


def validate_task(
    task: dict[str, Any],
    scene_object_names: set[str],
    scene_object_dims: dict[str, tuple] | None = None,
) -> tuple[bool, str]:
    """Validate a single task spec dict from LLM output.

    Args:
        task: Task dict from LLM JSON output.
        scene_object_names: Set of valid object names in the scene.
        scene_object_dims: Optional {name: (w, d, h)} for size feasibility checks.

    Returns:
        (is_valid, error_message). error_message is empty if valid.
    """
    errors = []

    # Required fields
    required = ["task_name", "instruction", "category", "difficulty",
                "contact_objects", "goal_relations", "success_conditions"]
    for field in required:
        if field not in task:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, "; ".join(errors)

    # Validate contact_objects exist in scene
    invalid_objects = []
    for obj_name in task.get("contact_objects", []):
        if not isinstance(obj_name, str):
            errors.append(f"contact_object is not a string: {obj_name}")
            continue
        if obj_name not in scene_object_names:
            invalid_objects.append(obj_name)
    if invalid_objects:
        errors.append(f"Objects not in scene: {invalid_objects}")

    # Validate goal_relations reference valid objects
    for rel in task.get("goal_relations", []):
        obj = rel.get("object", "")
        target = rel.get("target", "")
        if isinstance(obj, str) and obj and obj not in scene_object_names:
            errors.append(f"goal_relation object '{obj}' not in scene")
        if isinstance(target, str) and target and target not in scene_object_names and target != "table":
            errors.append(f"goal_relation target '{target}' not in scene")

    # Validate success_conditions reference valid predicates
    for cond in task.get("success_conditions", []):
        pred = cond.get("predicate", "")
        # Allow predicates with "object_" prefix (runtime versions)
        base_pred = pred.replace("object_", "").replace("_and_in_contact", "")
        if pred not in AVAILABLE_PREDICATES and base_pred not in AVAILABLE_PREDICATES:
            # Not a hard error — LLM might use valid compound predicates
            pass

        # Check objects in args
        args = cond.get("args", {})
        for key in ("object", "container", "reference_object", "surface"):
            val = args.get(key, "")
            # LLM might return a list of objects instead of a single string
            if isinstance(val, list):
                for v in val:
                    if v and v not in scene_object_names and v != "table":
                        errors.append(f"success_condition arg '{key}={v}' not in scene")
            elif val and val not in scene_object_names and val != "table":
                errors.append(f"success_condition arg '{key}={val}' not in scene")

    # Size feasibility: check containment tasks
    if scene_object_dims:
        for rel in task.get("goal_relations", []):
            if rel.get("predicate") == "in_container":
                inner = rel.get("object", "")
                outer = rel.get("target", "")
                if inner in scene_object_dims and outer in scene_object_dims:
                    inner_dims = scene_object_dims[inner]
                    outer_dims = scene_object_dims[outer]
                    if max(inner_dims) > min(outer_dims):
                        errors.append(
                            f"Size infeasible: {inner} (max dim {max(inner_dims):.3f}m) "
                            f"doesn't fit in {outer} (min dim {min(outer_dims):.3f}m)"
                        )

    # Validate basic structure
    if not task.get("contact_objects"):
        errors.append("contact_objects is empty")
    if not task.get("goal_relations"):
        errors.append("goal_relations is empty")
    if not task.get("instruction"):
        errors.append("instruction is empty")

    if errors:
        return False, "; ".join(errors)
    return True, ""


def validate_tasks_batch(
    tasks: list[dict[str, Any]],
    scene_object_names: set[str],
    scene_object_dims: dict[str, tuple] | None = None,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Validate a batch of tasks, separating valid from invalid.

    Returns:
        (valid_tasks, invalid_tasks_with_errors)
    """
    valid = []
    invalid = []
    for task in tasks:
        try:
            is_valid, error = validate_task(task, scene_object_names, scene_object_dims)
        except Exception as e:
            is_valid, error = False, f"Validation error: {e}"
        if is_valid:
            valid.append(task)
        else:
            invalid.append((task, error))
    return valid, invalid
