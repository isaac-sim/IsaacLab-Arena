# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Decide placement + task dispatch for a resolved scene.

``propose_placement`` is a pure transform from (ResolvedScene, SceneSpec)
into a ``Placement`` dataclass bundle — no source-string emission, no
file I/O. The env_writer consumes the bundle and renders it to a Python
module. Keeping the two concerns separate lets us:

  * Unit-test placement logic without parsing generated Python.
  * Insert feasibility gates (IK reachability, motion-plan validity)
    between propose and write, rejecting or tweaking a Placement before
    any file hits disk.
  * Swap the renderer later (e.g. emit a JSON scene description
    instead) without touching the proposer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .resolver import ResolvedScene
from .schema import SceneSpec

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

# Goal-relation kinds that map to ``PickAndPlaceTask``. ``on`` and ``in`` both
# describe "put subject at target"; the task's contact-sensor success predicate
# handles both identically.
_PICK_AND_PLACE_KINDS = {"on", "in"}

# Default episode length for auto-generated pick_and_place envs. Matches the
# hand-authored examples (e.g. pick_and_place_maple_table_environment).
_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S = 20.0

# Safety inset (meters) applied to the runtime tabletop bbox so items do
# not spawn right up against the edge.
_DEFAULT_TABLETOP_MARGIN_M = 0.05

# Per-background tabletop anchor spec. Maps a background's registered
# name to either:
#   * None  — the background asset itself is used as the tabletop anchor
#             (it's a standalone table USD, no sub-prim needed).
#   * str   — a USD prim path (may contain {ENV_REGEX_NS}) for a
#             sub-prim that represents the tabletop surface. The writer
#             emits an ObjectReference to that prim and uses it as the
#             On-parent + bbox source instead of the whole background.
#
# TODO: the bbox returned by ``get_world_bounding_box`` on the plain
# ``table`` background does NOT account for the 90° Z rotation applied
# via set_initial_pose, so PositionLimits derived from it clamps the
# wrong axes. Until that is fixed in the asset layer, prefer an
# ObjectReference to a tabletop sub-prim where available, and treat
# rotation-free backgrounds as the low-risk case.
_BACKGROUND_TABLETOP_ANCHOR: dict[str, str | None] = {
    # Standalone tables — background itself is the tabletop.
    "table": None,
    # Compound / wrapped backgrounds — use the tabletop sub-prim so the
    # bbox stays clean (matches pick_and_place_maple_table_environment /
    # gr1_table_multi_object_no_collision patterns).
    "maple_table_robolab": "{ENV_REGEX_NS}/maple_table_robolab/table",
    "table_maple_robolab": "{ENV_REGEX_NS}/table_maple_robolab/table",
    "office_table": "{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
    "kitchen": "{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static/Counter_Top_A",
}

# Short verb code per goal-relation kind. Chosen so the generated env name
# reads like "avocadoPnPbowltable" rather than a 40-character description.
_VERB_CODES: dict[str, str] = {
    "on": "PnP",
    "in": "PnP",
    "next_to": "Place",
    "at_position": "Move",
    "is_anchor": "Anchor",
}


# ---------------------------------------------------------------------------
# Placement data classes
# ---------------------------------------------------------------------------


@dataclass
class TabletopAnchorPlan:
    """How the generated env should anchor the tabletop surface."""

    kind: Literal["none", "background", "reference"]
    anchor_var: str  # "background" or "tabletop_anchor"
    prim_path: str | None  # USD prim path when kind == "reference"
    emit_position_limits: bool  # whether items get bbox-derived PL
    margin_m: float = _DEFAULT_TABLETOP_MARGIN_M

    def header_note(self, background_name: str) -> str:
        if self.kind == "none":
            return (
                f"background {background_name!r} not in _BACKGROUND_TABLETOP_ANCHOR "
                "— falling back to On(background) without PositionLimits"
            )
        descriptor = "background" if self.kind == "background" else "ObjectReference sub-prim"
        return (
            f"tabletop anchor = {self.anchor_var} ({descriptor}); "
            "PositionLimits derived from get_world_bounding_box()"
        )


@dataclass
class RelationSpec:
    """Structured view of one add_relation call to be rendered in the env."""

    kind: Literal["on", "position_limits", "at_position", "unsupported"]
    # On:
    on_target_var: str | None = None
    on_clearance_m: float = 0.02
    # PositionLimits: when source == "bbox", runtime-derived; when "static", use explicit bounds.
    pl_source: Literal["bbox", "static"] = "bbox"
    # Unsupported: keep the raw relation dict + a reason so it can be emitted as a TODO.
    raw_relation: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class PlacementItem:
    var_name: str  # Python variable in the generated env
    asset_name: str  # registered asset name for AssetRegistry.get_asset_by_name
    instance_name: str  # key used by the resolver / LLM
    relations: list[RelationSpec] = field(default_factory=list)


@dataclass
class TaskPlan:
    """What task to instantiate and how to annotate it."""

    kind: Literal["pick_and_place", "no_task"]
    task_import: str  # single ``from ... import ...`` line
    task_expr: str  # source for the task instance; may span multiple lines
    goal_comments: list[str]  # source lines documenting goal_added/removed
    header_note: str  # one-liner for the module docstring


@dataclass
class Placement:
    """Everything the env_writer needs to render an env module."""

    env_name: str
    class_name: str
    task_description: str
    background_name: str
    embodiment_default: str
    items: list[PlacementItem]
    tabletop_anchor_plan: TabletopAnchorPlan
    task_plan: TaskPlan
    extra_scene_assets: list[str]  # extra asset vars in Scene(assets=[...])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def propose_placement(resolved: ResolvedScene, spec: SceneSpec) -> Placement:
    """Turn a resolved scene into a Placement with no I/O side effects."""
    background_name = resolved.background.name if resolved.background else spec.background

    env_name = _derive_env_name(resolved, spec)
    class_name = _derive_class_name(resolved, spec)

    item_vars: dict[str, str] = {
        instance_name: f"{_safe_var(instance_name)}_obj" for instance_name in resolved.items
    }

    tabletop_plan = _plan_tabletop_anchor(background_name)

    items = _propose_items(resolved, spec, item_vars, tabletop_plan)

    task_plan = _plan_task(resolved, spec, item_vars, background_name)

    extra_assets: list[str] = []
    if tabletop_plan.kind == "reference":
        extra_assets.append("tabletop_anchor")

    return Placement(
        env_name=env_name,
        class_name=class_name,
        task_description=spec.task_description,
        background_name=background_name,
        embodiment_default=resolved.embodiment_name,
        items=items,
        tabletop_anchor_plan=tabletop_plan,
        task_plan=task_plan,
        extra_scene_assets=extra_assets,
    )


# ---------------------------------------------------------------------------
# Name / identifier helpers
# ---------------------------------------------------------------------------


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


def _derive_env_name(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """Compact env name from the goal diff, e.g. 'avocadoPnPbowltable'."""
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


def _derive_class_name(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """CamelCase variant with each segment capitalized, verb code preserved."""
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


# ---------------------------------------------------------------------------
# Tabletop anchor planning
# ---------------------------------------------------------------------------


def _plan_tabletop_anchor(background_name: str) -> TabletopAnchorPlan:
    """Decide whether to anchor to the background, an ObjectReference, or nothing."""
    if background_name not in _BACKGROUND_TABLETOP_ANCHOR:
        return TabletopAnchorPlan(
            kind="none", anchor_var="background", prim_path=None, emit_position_limits=False
        )
    prim = _BACKGROUND_TABLETOP_ANCHOR[background_name]
    if prim is None:
        return TabletopAnchorPlan(
            kind="background", anchor_var="background", prim_path=None, emit_position_limits=True
        )
    return TabletopAnchorPlan(
        kind="reference", anchor_var="tabletop_anchor", prim_path=prim, emit_position_limits=True
    )


# ---------------------------------------------------------------------------
# Item + relation planning
# ---------------------------------------------------------------------------


def _propose_items(
    resolved: ResolvedScene,
    spec: SceneSpec,
    item_vars: dict[str, str],
    tabletop_plan: TabletopAnchorPlan,
) -> list[PlacementItem]:
    """Build one PlacementItem per resolved item with structured relations."""
    items: list[PlacementItem] = []
    for instance_name, cls in resolved.items.items():
        items.append(
            PlacementItem(
                var_name=item_vars[instance_name],
                asset_name=cls.name,
                instance_name=instance_name,
                relations=[],
            )
        )
    # Walk the initial scene graph once; append relation specs to the
    # matching item. Items with no applicable relation get none (their
    # spot in the scene is purely the On from elsewhere, or the default
    # spawn pose).
    by_instance: dict[str, PlacementItem] = {i.instance_name: i for i in items}
    anchor_var = tabletop_plan.anchor_var

    for rel in resolved.initial_scene_graph:
        subj = rel["subject"]
        item = by_instance.get(subj)
        if item is None:
            # Unknown subject — record on the first item so the comment
            # survives rendering; this is strictly a defensive path.
            continue
        if rel["kind"] == "on":
            tgt = rel["target"]
            if tgt == spec.background:
                target_var = anchor_var
                is_tabletop = True
            else:
                target_var = item_vars.get(tgt)
                is_tabletop = False
            if target_var is None:
                item.relations.append(
                    RelationSpec(
                        kind="unsupported",
                        raw_relation=rel,
                        reason=f"unknown target {tgt!r}",
                    )
                )
                continue
            item.relations.append(RelationSpec(kind="on", on_target_var=target_var))
            if is_tabletop and tabletop_plan.emit_position_limits:
                item.relations.append(RelationSpec(kind="position_limits", pl_source="bbox"))
        else:
            item.relations.append(
                RelationSpec(
                    kind="unsupported",
                    raw_relation=rel,
                    reason=f"relation kind {rel['kind']!r} has no generator support yet",
                )
            )
    return items


# ---------------------------------------------------------------------------
# Task planning
# ---------------------------------------------------------------------------


def _plan_task(
    resolved: ResolvedScene,
    spec: SceneSpec,
    item_vars: dict[str, str],
    background_name: str,
) -> TaskPlan:
    """Decide which task to emit based on the resolved goal diff."""
    # Try PickAndPlaceTask: exactly one on/in goal between two resolved
    # items (not the background — PickAndPlaceTask uses the destination
    # for contact-sensor filtering, and the whole-scene background is a
    # poor candidate).
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
) -> TaskPlan:
    comments = [
        "        # goal_added (enforced by PickAndPlaceTask success predicate):"
        f"  {goal['kind']}({goal['subject']}, {goal['target']})"
    ]
    for r in resolved.goal_removed:
        comments.append(
            "        # goal_removed (implicitly negated when the pick succeeds):"
            f"  {r['kind']}({r['subject']}, {r['target']})"
        )
    return TaskPlan(
        kind="pick_and_place",
        task_import="from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask",
        task_expr=(
            "PickAndPlaceTask(\n"
            f"                pick_up_object={subject_var},\n"
            f"                destination_location={target_var},\n"
            "                background_scene=background,\n"
            f"                episode_length_s={_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S},\n"
            f"                task_description={spec.task_description!r},\n"
            "            )"
        ),
        goal_comments=comments,
        header_note=(
            f"PickAndPlaceTask: pick_up={subject_var.removesuffix('_obj')}, "
            f"destination={target_var.removesuffix('_obj')}"
        ),
    )


def _no_task_plan(resolved: ResolvedScene) -> TaskPlan:
    added = [(r["kind"], r["subject"], r["target"]) for r in resolved.goal_added]
    removed = [(r["kind"], r["subject"], r["target"]) for r in resolved.goal_removed]
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
    return TaskPlan(
        kind="no_task",
        task_import="from isaaclab_arena.tasks.no_task import NoTask",
        task_expr="NoTask()",
        goal_comments=comments,
        header_note=f"NoTask fallback ({reason}). Goal diff preserved as TODO comments.",
    )
