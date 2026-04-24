"""Build LLM prompts for task generation from scene info.

Given parsed scene objects and an RQA category, constructs a prompt
that asks the LLM to generate GoalSpec-compatible JSON task definitions.
"""

from __future__ import annotations

import json
from typing import Any

from isaaclab_arena.task_gen.rqa_templates import get_task_rqa, SPATIAL_VERBS


# Available predicates the LLM can use for goal_relations and success_conditions
AVAILABLE_PREDICATES = {
    "in_container":  "Object centroid is inside an open-top container",
    "on_top":        "Object is stably resting on top of another object/surface",
    "left_of":       "Object is to the left (+Y) of a reference object",
    "right_of":      "Object is to the right (-Y) of a reference object",
    "in_front_of":   "Object is in front (+X) of a reference object",
    "behind":        "Object is behind (-X) a reference object",
    "next_to":       "Object edge-to-edge distance < threshold from reference",
    "center_of":     "Object is XY-centered on a reference surface",
    "between":       "Object is positioned between two reference objects",
    "stacked":       "Objects are stacked vertically in specified order",
    "in_line":       "Objects are arranged in a line along an axis",
    "grabbed":       "Object is in contact with the gripper",
    "dropped":       "Object is NOT in contact with the gripper",
    "upright":       "Object is standing upright (within tolerance)",
    "door_open":     "Appliance door joint is in the open position",
    "door_closed":   "Appliance door joint is in the closed position",
}


def build_task_gen_prompt(
    scene_name: str,
    scene_objects: list[dict[str, Any]],
    categories: list[str],
    difficulty: str = "simple",
    num_tasks_per_category: int = 2,
    existing_tasks: list[str] | None = None,
) -> str:
    """Build LLM prompt for generating task specs from a scene.

    Args:
        scene_name: Name of the scene USD file (without extension).
        scene_objects: Parsed objects from scene_parser.parse_scene().
        categories: RQA categories to generate tasks for.
        difficulty: Target difficulty (simple / moderate / complex).
        num_tasks_per_category: How many tasks to generate per category.
        existing_tasks: List of existing task instructions to avoid duplicates.

    Returns:
        Complete LLM prompt string.
    """
    total_tasks = num_tasks_per_category * len(categories)

    # Format scene objects for the prompt
    objects_info = []
    articulated_objects = []
    for obj in scene_objects:
        dims = obj.get("dims")
        dims_str = f"{dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}m" if dims else "unknown"
        entry = {"name": obj["name"], "dims": dims_str}
        if obj.get("is_articulated"):
            entry["type"] = "articulated"
            articulated_objects.append(obj["name"])
        objects_info.append(entry)

    # Build the prompt
    rqa_info = get_task_rqa(categories)

    prompt = f"""You are a robot task generation expert. Generate {total_tasks} manipulation tasks for a tabletop scene.

**SCENE: {scene_name}**
Objects on the table:
```json
{json.dumps(objects_info, indent=2)}
```

{"Articulated objects (can be opened/closed): " + ", ".join(articulated_objects) if articulated_objects else ""}

**TASK CATEGORIES AND TEMPLATES:**
{rqa_info}

**OUTPUT FORMAT:**
Return a JSON array. Each task must have ALL these fields:
```json
[
  {{
    "task_name": "pick_apple_place_bowl",
    "instruction": "Pick the apple and place it in the bowl",
    "category": "recognition",
    "difficulty": "{difficulty}",
    "contact_objects": ["apple_012", "bowl_ycb_robolab"],
    "goal_relations": [
      {{"predicate": "in_container", "object": "apple_012", "target": "bowl_ycb_robolab", "params": {{}}}}
    ],
    "success_conditions": [
      {{"predicate": "object_in_container", "args": {{"object": "apple_012", "container": "bowl_ycb_robolab", "require_gripper_detached": true}}}}
    ],
    "subtasks": [
      {{"type": "pick_and_place", "object": "apple_012", "target": "bowl_ycb_robolab", "stages": ["grab", "lift", "move", "drop", "verify"]}}
    ],
    "episode_length_s": 40,
    "attributes": ["recognition", "simple"]
  }}
]
```

**AVAILABLE PREDICATES for goal_relations and success_conditions:**
{json.dumps(AVAILABLE_PREDICATES, indent=2)}

**SPATIAL VERBS:** {", ".join(SPATIAL_VERBS)}

**RULES:**
- Generate {num_tasks_per_category} task(s) for EACH category: {", ".join(categories)}
- Generate {total_tasks} tasks total.
- Difficulty: {difficulty}
  - simple: 1-2 objects, 1-2 steps, 1-2 success conditions
  - moderate: 2-3 objects, 2-4 steps, 2-3 success conditions (prefer multi-step tasks)
  - complex: 3+ objects, 4+ steps, 3+ success conditions (long-horizon with ordering/spatial constraints)
- PREFER longer-horizon tasks over single-object grab tasks. Each success condition should check a DIFFERENT aspect (e.g. object position AND gripper released AND spatial relation).
- Object names MUST exactly match the scene objects listed above.
- Do NOT invent objects not in the scene.
- contact_objects: list ALL objects the robot needs to touch.
- goal_relations: the desired spatial state AFTER the task is done (consumed by A* planner).
- success_conditions: runtime checks to determine if the task succeeded.
- subtasks: decomposition for progress tracking. Each subtask scores 1/N of total.
- episode_length_s: 20s per object manipulated (min 20s, max 120s).
- task_name: snake_case, descriptive, unique.
- Each task must be physically feasible given the object sizes.
- Do NOT place large objects inside small ones.
- Vary tasks across the templates — avoid repetitive patterns.
{"- Do NOT duplicate these existing tasks: " + json.dumps(existing_tasks) if existing_tasks else ""}

Return ONLY the JSON array. No markdown, no explanation."""

    return prompt


def build_fix_prompt(original_prompt: str, bad_output: str, error_msg: str) -> str:
    """Build a prompt asking the LLM to fix its previous invalid output."""
    return f"""{original_prompt}

**YOUR PREVIOUS OUTPUT HAD ERRORS:**
{error_msg}

**IMPORTANT**: If the error is "Size infeasible", do NOT place that object inside that container. Use a DIFFERENT task type instead (e.g. place ON TOP, place NEXT TO, or use a different container/surface). Do NOT repeat the same infeasible combination.

If the error mentions "not in scene", only use exact object names from the scene list above. Do NOT invent position names like "original_pos".

Fix the errors and return a valid JSON array. Keep the same task structure."""
