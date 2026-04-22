# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a registered ArenaEnvironment file from a ResolvedScene.

Takes the initial_scene_graph from a resolved scene and writes a Python
module that instantiates the background + embodiment + items, applies the
initial relations, and wires them into an IsaacLabArenaEnvironment.

Task selection is driven off ``resolved.goal_added`` / ``resolved.goal_removed``:

  * Exactly one ``on`` / ``in`` goal between two resolved items →
    ``PickAndPlaceTask`` (subject becomes ``pick_up_object``, target becomes
    ``destination_location``). ``goal_removed`` relations are implicitly
    negated by the pick — they are recorded as comments for traceability.
  * Anything else (multi-relation goals, unsupported kinds, target resolves
    to the background) → ``NoTask`` fallback with explicit ``TODO`` comments
    for each unmapped goal so the caller can wire them in manually.

``next_to`` / ``at_position`` initial placements are still emitted as
``TODO`` comments (see ``_SUPPORTED_INITIAL_KINDS``); the ``In`` relation
class is also still missing (see the TODO in
``isaaclab_arena.relations.relations``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .resolver import ResolvedScene
from .schema import SceneSpec

# Relation kinds the writer knows how to translate into Arena `add_relation`
# calls. Anything else is emitted as a commented-out placeholder.
_SUPPORTED_INITIAL_KINDS = {"on"}

# Goal-relation kinds that map to ``PickAndPlaceTask``. ``on`` and ``in`` both
# describe "put subject at target"; the task's contact-sensor success predicate
# handles both identically.
_PICK_AND_PLACE_KINDS = {"on", "in"}

# Default episode length for auto-generated pick_and_place envs. Matches the
# hand-authored examples (e.g. pick_and_place_maple_table_environment).
_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S = 20.0

# Short verb code per goal-relation kind. Chosen so the generated env name
# reads like "avocadoPnPbowltable" rather than a 40-character description.
_VERB_CODES: dict[str, str] = {
    "on": "PnP",
    "in": "PnP",
    "next_to": "Place",
    "at_position": "Move",
    "is_anchor": "Anchor",
}


def _safe_var(name: str) -> str:
    v = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    if not v:
        return "obj"
    if v[0].isdigit():
        v = "_" + v
    return v


def _compact(name: str) -> str:
    """Lowercase, strip non-alphanumerics — e.g. 'red_bell_pepper' -> 'redbellpepper'."""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower() if name else ""


def _env_name(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """Build a compact env name from the goal diff, e.g. 'avocadoPnPbowltable'.

    Subject of the first goal_added relation drives the primary object; the
    target drives the secondary; the background anchors the support surface.
    Falls back to the first resolved item if goal_added is empty, which the
    schema validator already treats as an error case.
    """
    if resolved.goal_added:
        diff = resolved.goal_added[0]
        verb = _VERB_CODES.get(diff["kind"], diff["kind"].capitalize())
        primary = _compact(diff["subject"])
        secondary = _compact(diff["target"])
    else:
        verb = "Env"
        primary = _compact(next(iter(resolved.items), "llm"))
        secondary = ""
    support = _compact(resolved.background.name if resolved.background else spec.background)
    return f"{primary}{verb}{secondary}{support}" or "llmEnv"


def _class_name(env_name: str, resolved: ResolvedScene, spec: SceneSpec) -> str:
    """CamelCase variant of the env name, with the verb code preserved."""
    if resolved.goal_added:
        diff = resolved.goal_added[0]
        verb = _VERB_CODES.get(diff["kind"], diff["kind"].capitalize())
        primary = _compact(diff["subject"]).capitalize()
        secondary = _compact(diff["target"]).capitalize()
    else:
        verb = "Env"
        primary = _compact(next(iter(resolved.items), "llm")).capitalize()
        secondary = ""
    support = _compact(resolved.background.name if resolved.background else spec.background).capitalize()
    return f"{primary}{verb}{secondary}{support}Environment"


def write_env(resolved: ResolvedScene, spec: SceneSpec, out_path: str | Path) -> Path:
    """Render the env module and return the final path.

    ``out_path`` may be either an explicit ``*.py`` file or a directory —
    in the latter case the filename is derived from the env name so the
    generated module on disk matches the registered env name.
    """
    env_slug = _env_name(resolved, spec)
    class_name = _class_name(env_slug, resolved, spec)

    out_path = Path(out_path)
    if out_path.suffix != ".py":
        out_path = out_path / f"{env_slug}.py"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    item_vars: dict[str, str] = {}  # instance_name -> python var
    for instance_name in resolved.items:
        item_vars[instance_name] = f"{_safe_var(instance_name)}_obj"

    background_name = resolved.background.name if resolved.background else spec.background
    task_plan = _plan_task(resolved, spec, item_vars, background_name)

    body = _render_module(
        class_name=class_name,
        env_name=env_slug,
        task_description=spec.task_description,
        background_name=background_name,
        embodiment_default=resolved.embodiment_name,
        items=[(item_vars[n], cls.name) for n, cls in resolved.items.items()],
        initial_relations=_render_relations(resolved, spec, item_vars),
        asset_list_src=["background", "ground_plane", "light", *item_vars.values()],
        task_plan=task_plan,
    )

    out_path.write_text(body)
    return out_path


def _plan_task(
    resolved: ResolvedScene,
    spec: SceneSpec,
    item_vars: dict[str, str],
    background_name: str,
) -> dict[str, Any]:
    """Decide which task to emit based on the resolved goal diff.

    Returns a dict with:

      * ``kind``          — ``"pick_and_place"`` or ``"no_task"``
      * ``task_import``   — single ``from ... import ...`` line to inline in
        ``get_env`` (matches the style of the other imports there)
      * ``task_expr``     — source for the task instance passed to
        ``IsaacLabArenaEnvironment(..., task=<expr>)``; may span multiple lines
        and must already be indented for placement at call-site column.
      * ``goal_comments`` — lines (each already prefixed with leading
        whitespace) documenting ``goal_added`` / ``goal_removed`` entries —
        success contract for pick_and_place, or manual TODOs for NoTask.
      * ``header_note``   — one-liner explaining what was emitted; goes in the
        module docstring so ``head -n 20 foo.py`` is self-explanatory.
    """
    # Try to fit PickAndPlaceTask: exactly one on/in goal between resolved items
    # (not the background — PickAndPlaceTask distinguishes background_scene
    # from destination_location, and uses the destination for contact-sensor
    # filtering, which a whole-scene background generally does not support).
    if len(resolved.goal_added) == 1:
        g = resolved.goal_added[0]
        if g["kind"] in _PICK_AND_PLACE_KINDS:
            subj_var = item_vars.get(g["subject"])
            tgt_var = None
            if g["target"] != background_name:
                tgt_var = item_vars.get(g["target"])
            if subj_var is not None and tgt_var is not None:
                return _pick_and_place_plan(resolved, spec, g, subj_var, tgt_var)

    return _no_task_plan(resolved)


def _pick_and_place_plan(
    resolved: ResolvedScene,
    spec: SceneSpec,
    goal: dict[str, Any],
    subject_var: str,
    target_var: str,
) -> dict[str, Any]:
    comments = [
        "        # goal_added (enforced by PickAndPlaceTask success predicate):"
        f"  {goal['kind']}({goal['subject']}, {goal['target']})"
    ]
    for r in resolved.goal_removed:
        comments.append(
            "        # goal_removed (implicitly negated when the pick succeeds):"
            f"  {r['kind']}({r['subject']}, {r['target']})"
        )
    return {
        "kind": "pick_and_place",
        "task_import": "from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask",
        "task_expr": (
            "PickAndPlaceTask(\n"
            f"                pick_up_object={subject_var},\n"
            f"                destination_location={target_var},\n"
            "                background_scene=background,\n"
            f"                episode_length_s={_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S},\n"
            f"                task_description={spec.task_description!r},\n"
            "            )"
        ),
        "goal_comments": comments,
        "header_note": (
            f"PickAndPlaceTask: pick_up={subject_var.removesuffix('_obj')}, "
            f"destination={target_var.removesuffix('_obj')}"
        ),
    }


def _no_task_plan(resolved: ResolvedScene) -> dict[str, Any]:
    """Fallback when the goal diff does not fit a known task shape."""
    added = [(r["kind"], r["subject"], r["target"]) for r in resolved.goal_added]
    removed = [(r["kind"], r["subject"], r["target"]) for r in resolved.goal_removed]
    reason: str
    if not resolved.goal_added:
        reason = "no goal_added relations in resolved scene"
    elif len(resolved.goal_added) > 1:
        reason = f"multi-relation goal not yet supported ({len(resolved.goal_added)} added)"
    else:
        g = resolved.goal_added[0]
        if g["kind"] not in _PICK_AND_PLACE_KINDS:
            reason = f"goal kind {g['kind']!r} has no task mapping yet"
        else:
            reason = (
                f"goal {g['kind']}({g['subject']}, {g['target']}) does not resolve "
                "to two distinct items (target may be the background or unresolved)"
            )
    comments = [f"        # NoTask fallback — {reason}."]
    for kind, subj, tgt in added:
        comments.append(f"        # TODO(goal_added): {kind}({subj}, {tgt}) — wire success predicate")
    for kind, subj, tgt in removed:
        comments.append(f"        # TODO(goal_removed): {kind}({subj}, {tgt}) — wire negation")
    return {
        "kind": "no_task",
        "task_import": "from isaaclab_arena.tasks.no_task import NoTask",
        "task_expr": "NoTask()",
        "goal_comments": comments,
        "header_note": f"NoTask fallback ({reason}). Goal diff preserved as TODO comments.",
    }


def _render_relations(resolved: ResolvedScene, spec: SceneSpec, item_vars: dict[str, str]) -> list[str]:
    """Translate initial_scene_graph into add_relation call source lines."""
    lines: list[str] = []
    for rel in resolved.initial_scene_graph:
        subject_var = item_vars.get(rel["subject"])
        target_name = rel["target"]

        if subject_var is None:
            lines.append(f"        # SKIPPED (unknown subject {rel['subject']!r}): {rel}")
            continue

        if rel["kind"] == "on":
            target_var = "background" if target_name == spec.background else item_vars.get(target_name)
            if target_var is None:
                lines.append(f"        # SKIPPED (unknown target {target_name!r}): {rel}")
                continue
            lines.append(f"        {subject_var}.add_relation(On({target_var}, clearance_m=0.02))")
        else:
            lines.append(f"        # TODO({rel['kind']}): no generator support yet for this relation kind. raw={rel}")
    return lines


def _render_module(
    *,
    class_name: str,
    env_name: str,
    task_description: str,
    background_name: str,
    embodiment_default: str,
    items: list[tuple[str, str]],
    initial_relations: list[str],
    asset_list_src: list[str],
    task_plan: dict[str, Any],
) -> str:
    # TODO: Support multiple instances of the same library asset (e.g. two
    # bananas in one scene). Right now each item gets the asset's registered
    # name by default, so two bananas would collide on the scene-level name.
    # When we need this, re-introduce instance_name="..." on the emitted
    # line and derive a unique suffix from the LLM's instance_name / query
    # (e.g. "banana_1", "banana_2"). The resolver already keeps items keyed
    # by instance_name, so this is a generator-side fix only.
    item_decls = "\n".join(
        f'        {var} = self.asset_registry.get_asset_by_name("{asset}")()' for var, asset in items
    )
    relations_src = "\n".join(initial_relations) if initial_relations else "        # (no initial relations)"
    asset_list = ", ".join(asset_list_src)

    goal_comments = "\n".join(task_plan["goal_comments"])
    task_import = task_plan["task_import"]
    task_expr = task_plan["task_expr"]
    task_header_note = task_plan["header_note"]

    return f'''# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-generated by isaaclab_arena.llm_env_gen.

Task: {task_description}

Task wiring: {task_header_note}
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class {class_name}(ExampleEnvironmentBase):
    """{task_description}"""

    name: str = "{env_name}"

    def get_env(self, args_cli: argparse.Namespace) -> "IsaacLabArenaEnvironment":
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        {task_import}
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("{background_name}")()
        background.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068)))
        background.add_relation(IsAnchor())
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))
        light = self.asset_registry.get_asset_by_name("light")()

        # No kwargs — works for both no_embodiment and robot embodiments whose
        # optional flags (enable_cameras, etc.) default to safe values.
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

{item_decls}

{relations_src}

        scene = Scene(assets=[{asset_list}])

{goal_comments}

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task={task_expr},
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="{embodiment_default}")
'''
