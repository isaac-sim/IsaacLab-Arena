"""Task generator orchestrator: scene USD -> LLM -> validated GoalSpec outputs.

Pipeline:
    scene.usda -> scene_parser -> build_prompt -> LLM -> validate -> GoalSpec JSON + Python task class

Usage:
    gen = TaskGenerator()
    specs = gen.generate_tasks_for_scene("scene.usda", categories=["recognition", "spatial"])
    gen.generate_tasks_for_folder("scenes/", output_dir="tasks/")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from isaaclab_arena.task_gen.build_prompt import build_task_gen_prompt, build_fix_prompt
from isaaclab_arena.task_gen.goal_spec import GoalSpec, GoalRelation, SuccessCondition, SubtaskSpec, ObjectState
from isaaclab_arena.task_gen.rqa_templates import get_categories_for_scene, ARTICULATED_CATEGORIES
from isaaclab_arena.task_gen.task_validation import validate_tasks_batch


class TaskGenerator:
    """Orchestrates LLM-driven task generation from scene USDs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aws/anthropic/bedrock-claude-opus-4-6",
        base_url: str = "https://inference-api.nvidia.com",
        output_dir: Optional[str] = None,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.getenv("NV_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set NV_API_KEY or pass api_key.")

        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "generated")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_tasks_for_scene(
        self,
        usd_path: str,
        categories: list[str] | None = None,
        difficulty: str = "simple",
        num_tasks_per_category: int = 2,
        asset_manager=None,
    ) -> list[GoalSpec]:
        """Generate tasks for a single scene USD.

        Args:
            usd_path: Path to the .usda scene file.
            categories: RQA categories. If None, auto-detects from scene content.
            difficulty: Target difficulty level.
            num_tasks_per_category: Tasks to generate per category.
            asset_manager: Optional ArenaAssetManager for dims/articulated lookup.

        Returns:
            List of validated GoalSpec objects.
        """
        from isaaclab_arena.task_gen.scene_parser import parse_scene

        scene_name = Path(usd_path).stem
        print(f"\n[TaskGen] Scene: {scene_name}")
        print(f"[TaskGen] Difficulty: {difficulty}")

        # 1. Parse scene
        scene_objects = parse_scene(usd_path, asset_manager=asset_manager)
        print(f"[TaskGen] Found {len(scene_objects)} objects")

        if len(scene_objects) < 2:
            print(f"[TaskGen] Too few objects ({len(scene_objects)}), skipping")
            return []

        # 2. Determine categories
        has_articulated = any(obj.get("is_articulated") for obj in scene_objects)
        if categories is None:
            categories = get_categories_for_scene(has_articulated)
        else:
            # Filter out articulated categories if no articulated objects
            if not has_articulated:
                categories = [c for c in categories if c not in ARTICULATED_CATEGORIES]

        print(f"[TaskGen] Categories: {categories}")

        # 3. Build initial state from scene objects
        scene_object_names = set()
        scene_object_dims = {}
        initial_state = {}
        for obj in scene_objects:
            name = obj["name"]
            scene_object_names.add(name)
            if obj.get("dims"):
                scene_object_dims[name] = obj["dims"]
            initial_state[name] = ObjectState(
                position=obj["position"],
                rotation=obj.get("rotation", (1, 0, 0, 0)),
                dims=obj.get("dims"),
            )

        # 4. Check for existing tasks (avoid duplicates)
        scene_task_dir = self.output_dir / scene_name
        existing_instructions = []
        if scene_task_dir.exists():
            for f in scene_task_dir.glob("*.json"):
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                        existing_instructions.append(d.get("instruction", ""))
                except Exception:
                    pass

        # 5. Call LLM in batches of categories to avoid token truncation.
        # Each batch generates tasks for 3-4 categories (fits in output limit).
        MAX_CATEGORIES_PER_CALL = 4
        all_valid_specs = []

        for batch_start in range(0, len(categories), MAX_CATEGORIES_PER_CALL):
            batch_cats = categories[batch_start:batch_start + MAX_CATEGORIES_PER_CALL]

            for attempt in range(self.max_retries):
                prompt = build_task_gen_prompt(
                    scene_name=scene_name,
                    scene_objects=scene_objects,
                    categories=batch_cats,
                    difficulty=difficulty,
                    num_tasks_per_category=num_tasks_per_category,
                    existing_tasks=[s.instruction for s in all_valid_specs]
                        + (existing_instructions if existing_instructions else []),
                )

                if attempt > 0:
                    print(f"[TaskGen] Retry {attempt} of {self.max_retries - 1} "
                          f"(categories: {batch_cats})")

                try:
                    raw_output = self._call_llm(prompt)
                    tasks = self._parse_llm_output(raw_output)
                    print(f"[TaskGen] LLM returned {len(tasks)} tasks for {batch_cats}")

                    # 6. Validate
                    valid, invalid = validate_tasks_batch(
                        tasks, scene_object_names, scene_object_dims,
                    )
                    print(f"[TaskGen] Valid: {len(valid)}, Invalid: {len(invalid)}")

                    for task, error in invalid:
                        print(f"  [Invalid] {task.get('task_name', '?')}: {error}")

                    # 7. Convert valid tasks to GoalSpec
                    for task in valid:
                        spec = self._task_dict_to_goal_spec(
                            task, scene_name, initial_state,
                        )
                        all_valid_specs.append(spec)

                    if not invalid:
                        break  # This batch succeeded fully

                    # Narrow retry to only failed categories
                    failed_cats = {t.get("category") for t, _ in invalid}
                    batch_cats = [c for c in batch_cats if c in failed_cats]
                    if not batch_cats:
                        break

                except Exception as e:
                    print(f"[TaskGen] Error: {e}")
                    if attempt == self.max_retries - 1:
                        break

        # 8. Deduplicate by task_name
        seen = set()
        unique_specs = []
        for spec in all_valid_specs:
            if spec.task_name not in seen:
                seen.add(spec.task_name)
                unique_specs.append(spec)

        # 9. Save outputs
        for spec in unique_specs:
            spec.scene = os.path.basename(usd_path)
            path = spec.save_json(self.output_dir / scene_name)
            print(f"[TaskGen] Saved: {path}")

        print(f"[TaskGen] Generated {len(unique_specs)} tasks for {scene_name}")
        return unique_specs

    @staticmethod
    def get_difficulty_schedule(num_objects: int) -> dict[str, int]:
        """Auto-select difficulties and tasks-per-difficulty based on object count.

        More objects → more difficulties available, biased toward moderate/complex.

        Args:
            num_objects: Number of objects in the scene.

        Returns:
            Dict mapping difficulty → num_tasks_per_category for that difficulty.
        """
        if num_objects >= 10:
            # Full spread, heavy on moderate/complex
            return {"simple": 1, "moderate": 2, "complex": 2}
        elif num_objects >= 6:
            # Simple + moderate, more moderate
            return {"simple": 1, "moderate": 2}
        else:
            # Only simple and moderate (light)
            return {"simple": 1, "moderate": 1}

    def generate_all_tasks_for_scene(
        self,
        usd_path: str,
        categories: list[str] | None = None,
        asset_manager=None,
    ) -> list[GoalSpec]:
        """Generate tasks across auto-selected difficulties for a scene.

        Uses object count to determine which difficulties to generate,
        biased toward moderate and complex tasks.
        """
        from isaaclab_arena.task_gen.scene_parser import parse_scene

        scene_objects = parse_scene(usd_path, asset_manager=asset_manager)
        schedule = self.get_difficulty_schedule(len(scene_objects))

        print(f"[TaskGen] {Path(usd_path).stem}: {len(scene_objects)} objects → "
              f"schedule: {schedule}")

        all_specs = []
        for difficulty, num_per_cat in schedule.items():
            specs = self.generate_tasks_for_scene(
                usd_path,
                categories=categories,
                difficulty=difficulty,
                num_tasks_per_category=num_per_cat,
                asset_manager=asset_manager,
            )
            all_specs.extend(specs)

        return all_specs

    def generate_tasks_for_folder(
        self,
        scene_folder: str,
        categories: list[str] | None = None,
        auto_difficulty: bool = True,
        difficulty: str = "simple",
        num_tasks_per_category: int = 2,
        asset_manager=None,
    ) -> dict[str, list[GoalSpec]]:
        """Generate tasks for all scenes in a folder.

        Args:
            auto_difficulty: If True, auto-select difficulties per scene based
                on object count (recommended). If False, use fixed difficulty.
        """
        scene_folder = Path(scene_folder)
        usd_files = sorted(scene_folder.glob("*.usda"))

        print(f"[TaskGen] Found {len(usd_files)} scenes in {scene_folder}")

        results = {}
        total_tasks = 0
        for i, usd_file in enumerate(usd_files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(usd_files)}] {usd_file.name}")
            print(f"{'='*60}")

            if auto_difficulty:
                specs = self.generate_all_tasks_for_scene(
                    str(usd_file),
                    categories=categories,
                    asset_manager=asset_manager,
                )
            else:
                specs = self.generate_tasks_for_scene(
                    str(usd_file),
                    categories=categories,
                    difficulty=difficulty,
                    num_tasks_per_category=num_tasks_per_category,
                    asset_manager=asset_manager,
                )
            results[usd_file.stem] = specs
            total_tasks += len(specs)

        print(f"\n[TaskGen] Total: {total_tasks} tasks from {len(usd_files)} scenes")
        return results

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return raw text output."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a robot task generation expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=16000,
        )
        return response.choices[0].message.content

    def _parse_llm_output(self, content: str) -> list[dict]:
        """Parse LLM output to extract JSON task array."""
        content = content.strip()

        # Strip markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        # Try direct parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "tasks" in result:
                return result["tasks"]
            return [result]
        except json.JSONDecodeError:
            pass

        # Find JSON array in text
        bracket_start = content.find("[")
        if bracket_start != -1:
            depth = 0
            for i in range(bracket_start, len(content)):
                if content[i] == "[":
                    depth += 1
                elif content[i] == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[bracket_start:i + 1])
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"Failed to parse LLM output as JSON array: {content[:200]}...")

    def _task_dict_to_goal_spec(
        self,
        task: dict,
        scene_name: str,
        initial_state: dict[str, ObjectState],
    ) -> GoalSpec:
        """Convert a validated task dict to a GoalSpec dataclass."""
        goal_relations = []
        for r in task.get("goal_relations", []):
            obj = r.get("object", "")
            if isinstance(obj, list):
                obj = obj[0] if obj else ""
            target = r.get("target", "")
            if isinstance(target, list):
                target = target[0] if target else ""
            goal_relations.append(GoalRelation(
                predicate=r.get("predicate", "on_top"),
                object=obj,
                target=target,
                params=r.get("params", {}),
            ))

        success_conditions = []
        for c in task.get("success_conditions", []):
            args = c.get("args", {})
            # Flatten any list values in args to strings
            clean_args = {}
            for k, v in args.items():
                if isinstance(v, list):
                    clean_args[k] = v[0] if v else ""
                else:
                    clean_args[k] = v
            success_conditions.append(SuccessCondition(
                predicate=c.get("predicate", ""),
                args=clean_args,
            ))

        subtasks = []
        for s in task.get("subtasks", []):
            obj = s.get("object", s.get("objects", ""))
            if isinstance(obj, list):
                obj = obj[0] if obj else ""
            subtasks.append(SubtaskSpec(
                type=s.get("type", "pick_and_place"),
                object=obj,
                target=s.get("target", ""),
                stages=s.get("stages", ["grab", "lift", "move", "drop", "verify"]),
            ))

        # Filter initial_state to only include contact objects
        task_initial_state = {
            name: state for name, state in initial_state.items()
            if name in set(task.get("contact_objects", []))
        }

        return GoalSpec(
            task_name=task["task_name"],
            instruction=task["instruction"],
            scene=scene_name,
            category=task["category"],
            difficulty=task["difficulty"],
            contact_objects=task["contact_objects"],
            initial_state=task_initial_state,
            goal_relations=goal_relations,
            success_conditions=success_conditions,
            subtasks=subtasks,
            episode_length_s=task.get("episode_length_s", 60),
            attributes=task.get("attributes", [task["category"], task["difficulty"]]),
        )
