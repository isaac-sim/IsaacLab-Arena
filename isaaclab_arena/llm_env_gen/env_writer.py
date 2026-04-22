# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a registered ArenaEnvironment file from a ResolvedScene.

Takes the initial_scene_graph from a resolved scene and writes a Python
module that instantiates the background + embodiment + items, applies the
initial relations, and wires them into an IsaacLabArenaEnvironment with
NoTask (the null task used for non-goal-driven scene walkthroughs).

The `in` / goal relations are intentionally NOT materialized here — NoTask
has no success predicate, and there is no In relation class yet (see the
TODO in isaaclab_arena.relations.relations). To wire up goal checking,
swap NoTask for a real task and implement the In relation first.
"""

from __future__ import annotations

import re
from pathlib import Path

from .resolver import ResolvedScene
from .schema import SceneSpec

# Relation kinds the writer knows how to translate into Arena `add_relation`
# calls. Anything else is emitted as a commented-out placeholder.
_SUPPORTED_INITIAL_KINDS = {"on"}

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

    body = _render_module(
        class_name=class_name,
        env_name=env_slug,
        task_description=spec.task_description,
        background_name=resolved.background.name if resolved.background else spec.background,
        embodiment_default=resolved.embodiment_name,
        items=[(item_vars[n], cls.name) for n, cls in resolved.items.items()],
        initial_relations=_render_relations(resolved, spec, item_vars),
        asset_list_src=["background", "ground_plane", "light", *item_vars.values()],
    )

    out_path.write_text(body)
    return out_path


def _render_relations(
    resolved: ResolvedScene, spec: SceneSpec, item_vars: dict[str, str]
) -> list[str]:
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
            lines.append(
                f"        # TODO({rel['kind']}): no generator support yet for this relation kind. raw={rel}"
            )
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
) -> str:
    # TODO: Support multiple instances of the same library asset (e.g. two
    # bananas in one scene). Right now each item gets the asset's registered
    # name by default, so two bananas would collide on the scene-level name.
    # When we need this, re-introduce instance_name="..." on the emitted
    # line and derive a unique suffix from the LLM's instance_name / query
    # (e.g. "banana_1", "banana_2"). The resolver already keeps items keyed
    # by instance_name, so this is a generator-side fix only.
    item_decls = "\n".join(
        f'        {var} = self.asset_registry.get_asset_by_name("{asset}")()'
        for var, asset in items
    )
    relations_src = "\n".join(initial_relations) if initial_relations else "        # (no initial relations)"
    asset_list = ", ".join(asset_list_src)

    return f'''# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-generated by isaaclab_arena.llm_env_gen.

Task: {task_description}

NoTask is used as a placeholder — the initial scene graph is materialized
via add_relation() calls on each item, but there is no success predicate.
Swap NoTask for a real task (and implement the In relation) to wire up
goal checking.
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
        from isaaclab_arena.tasks.no_task import NoTask
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

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="{embodiment_default}")
'''
