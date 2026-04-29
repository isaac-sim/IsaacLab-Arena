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

import math
import random
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

# Goal-relation kinds that map to OpenDoorTask / CloseDoorTask.
_OPEN_DOOR_KINDS = frozenset({"open"})
_CLOSE_DOOR_KINDS = frozenset({"closed"})

# Default episode lengths for auto-generated envs.
_DEFAULT_PICK_AND_PLACE_EPISODE_LENGTH_S = 20.0
_DEFAULT_OPEN_DOOR_EPISODE_LENGTH_S = 30.0

# Safety inset (meters) applied to the runtime tabletop bbox so items do
# not spawn right up against the edge. 0.15 m (~half a hand-span) keeps
# items well inside the table even at the corners and pulls the
# placement pool's "best" layouts toward the centre, which gives the
# robot a much better chance at IK reachability.
_DEFAULT_TABLETOP_MARGIN_M = 0.15

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

# Background → short family name used in the env slug. Different variants
# of the same physical surface (e.g. maple_table_robolab / office_table /
# plain table) all collapse to "table" so the generated env name stays
# readable ("avocadoPnPbowltable") regardless of which specific USD we
# pick. Backgrounds not in this map contribute their full registered
# name (compacted) — tweak here when a new family shows up.
_BACKGROUND_NAME_ALIASES: dict[str, str] = {
    "table": "table",
    "maple_table_robolab": "table",
    "table_maple_robolab": "table",
    "office_table": "table",
    "packing_table": "table",
    "kitchen": "kitchen",
    "kitchen_with_open_drawer": "kitchen",
    "lightwheel_robocasa_kitchen": "kitchen",
}

# Short verb code per goal-relation kind. Chosen so the generated env name
# reads like "avocadoPnPbowltable" rather than a 40-character description.
_VERB_CODES: dict[str, str] = {
    "on": "PnP",
    "in": "PnP",
    "next_to": "Place",
    "at_position": "Move",
    "is_anchor": "Anchor",
    "open": "Open",
    "closed": "Close",
}

# Yaw (radians, world frame) applied via RotateAroundSolution so that an
# openable appliance's door faces the robot at the world origin. The
# standard background is placed at (0.5, 0.0, 0.0) rotated 90° Z.
# -π/2 matches the GR1 microwave env orientation (door toward -X ≈ robot).
_APPLIANCE_FACING_YAW: dict[tuple[str, str], float] = {
    ("kitchen", "microwave"): -math.pi / 2,
}
_DEFAULT_APPLIANCE_FACING_YAW: float = -math.pi / 2

# Backgrounds whose tabletop sub-prim lacks RigidBodyAPI. The anchor
# ObjectReference must omit object_type=ObjectType.RIGID in these cases.
_BACKGROUND_TABLETOP_ANCHOR_BASE_TYPE: frozenset[str] = frozenset({"kitchen"})


# Edge-name → yaw quaternion (qx, qy, qz, qw) so the robot faces the
# table centre from the chosen edge. Four cardinal orientations cover
# the full perimeter of a rectangular tabletop.
_EDGE_ROTATION_XYZW: dict[str, tuple[float, float, float, float]] = {
    "x_min": (0.0, 0.0, 0.0, 1.0),               # faces +x (0°)
    "x_max": (0.0, 0.0, 1.0, 0.0),               # faces -x (180°)
    "y_min": (0.0, 0.0, 0.7071068, 0.7071068),   # faces +y (+90°)
    "y_max": (0.0, 0.0, -0.7071068, 0.7071068),  # faces -y (-90°)
}

# Robot-base sits this far outside the sampled edge. Large enough that
# the robot-stand footprint does not intersect the table, small enough
# that objects in the middle stay within Franka's ~0.85 m reach
# envelope.
_ROBOT_EDGE_OFFSET_M = 0.1

# Edge-fraction sampling range: drawn uniformly from this band so the
# robot lands somewhere across the middle two-thirds of the chosen
# edge but stays clear of either corner.
_ROBOT_EDGE_FRACTION_RANGE = (0.3, 0.7)


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
    # "RIGID" → emit object_type=ObjectType.RIGID; "BASE" → omit it (prim has no RigidBodyAPI).
    anchor_object_type: str = "RIGID"

    def header_note(self, background_name: str) -> str:
        if self.kind == "none":
            return (
                f"background {background_name!r} not in _BACKGROUND_TABLETOP_ANCHOR "
                "— falling back to On(background) without PositionLimits"
            )
        descriptor = "background" if self.kind == "background" else "ObjectReference sub-prim"
        return (
            f"tabletop anchor = {self.anchor_var} ({descriptor}); PositionLimits derived from get_world_bounding_box()"
        )


@dataclass
class RelationSpec:
    """Structured view of one add_relation call to be rendered in the env."""

    kind: Literal["on", "in", "not", "position_limits", "at_position", "rotate_around_solution", "unsupported"]
    # On / In: parent variable to reference. On also carries a clearance.
    on_target_var: str | None = None
    on_clearance_m: float = 0.02
    in_target_var: str | None = None
    # Not: the wrapped spec whose satisfaction we forbid. Only "on" and
    # "in" inner kinds are rendered today.
    inner: RelationSpec | None = None
    # PositionLimits: when source == "bbox", runtime-derived; when "static", use explicit bounds.
    pl_source: Literal["bbox", "static"] = "bbox"
    # RotateAroundSolution: yaw applied on top of the solver-found position.
    rotate_yaw_rad: float = 0.0
    # Unsupported: keep the raw relation dict + a reason so it can be emitted as a TODO.
    raw_relation: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class GoalBindingSpec:
    """Where an item is expected to end up — derived from goal_added."""

    kind: Literal["on", "in"]
    target_var: str  # Python var in the generated env
    target_name: str  # instance_name (or background name) for logs


@dataclass
class PlacementItem:
    var_name: str  # Python variable in the generated env
    asset_name: str  # registered asset name for AssetRegistry.get_asset_by_name
    instance_name: str  # key used by the resolver / LLM
    relations: list[RelationSpec] = field(default_factory=list)
    goal_binding: GoalBindingSpec | None = None  # populated by block_initial_goal_satisfaction
    # True for articulated objects with an open/close goal (e.g. microwave).
    # The proposer adds RotateAroundSolution to orient their door toward the robot.
    is_openable: bool = False


@dataclass
class TaskPlan:
    """What task to instantiate and how to annotate it."""

    kind: Literal["pick_and_place", "open_door", "close_door", "no_task"]
    task_import: str  # single ``from ... import ...`` line
    task_expr: str  # source for the task instance; may span multiple lines
    goal_comments: list[str]  # source lines documenting goal_added/removed
    header_note: str  # one-liner for the module docstring


@dataclass
class RobotPlacement:
    """Where to plant the robot base along one of the four tabletop edges.

    Sampled as ``(edge, fraction)``: the edge is drawn uniformly from
    ``{x_min, x_max, y_min, y_max}`` and the fraction along that edge
    is drawn from ``Uniform(0, 1)``. At render time the generated env
    plants the robot base ``offset_m`` outside that edge, at the
    sampled fraction between the two corners. The base yaw is one of
    four cardinals (per-edge) so the gripper is axis-aligned and
    points across the table at the objects.

    Only meaningful when the env exposes a usable tabletop bbox
    (``TabletopAnchorPlan.emit_position_limits`` is True).
    """

    edge: Literal["x_min", "x_max", "y_min", "y_max"]
    fraction: float  # in [0, 1] — interpolated between the two corners of the edge
    offset_m: float  # outward offset from the edge (meters)
    z_m: float  # world-Z of the robot base (meters)
    rotation_xyzw: tuple[float, float, float, float]


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
    robot_placement: RobotPlacement | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def block_initial_goal_satisfaction(placement: Placement, resolved: ResolvedScene) -> Placement:
    """Attach final-graph-aware constraints that keep the initial state
    from already satisfying any ``goal_added`` relation.

    For every entry in ``resolved.goal_added``:

      * Record a :class:`GoalBindingSpec` on the subject's
        ``PlacementItem`` so downstream consumers (task wiring,
        feasibility gates) can see where it is supposed to end up.
      * Append a ``RelationSpec(kind="not", inner=<on|in> spec)`` so
        the generated env calls ``subject.add_relation(Not(In(target)))``.
        The solver's ``NotRelationLossStrategy`` pushes the initial
        placement out of the goal region.

    Missing / unresolvable targets are skipped with a trace-friendly
    ``unsupported`` RelationSpec so the caller can see the reason in
    the generated file. Returns the mutated Placement for convenience;
    mutation is in-place.
    """
    item_vars = {i.instance_name: i.var_name for i in placement.items}
    by_instance: dict[str, PlacementItem] = {i.instance_name: i for i in placement.items}

    for goal in resolved.goal_added:
        kind = goal["kind"]
        subj = goal["subject"]
        tgt = goal["target"]

        # Open/close door goals are handled by the task, not placement constraints.
        if kind in _OPEN_DOOR_KINDS | _CLOSE_DOOR_KINDS:
            continue

        item = by_instance.get(subj)
        if item is None:
            # Unresolved subject — let the proposer's original trace carry it.
            continue

        # Target may be another item or the background — resolve once.
        if tgt == placement.background_name:
            target_var = placement.tabletop_anchor_plan.anchor_var
        else:
            target_var = item_vars.get(tgt)

        if target_var is None:
            item.relations.append(
                RelationSpec(
                    kind="unsupported",
                    raw_relation=goal,
                    reason=f"block_initial_goal_satisfaction: unknown target {tgt!r}",
                )
            )
            continue

        # Explicit kind dispatch. Each branch builds the inner spec that
        # Not(...) will wrap; unsupported kinds fall through to the else.
        if kind == "in":
            inner = RelationSpec(kind="in", in_target_var=target_var)
        elif kind == "on":
            inner = RelationSpec(kind="on", on_target_var=target_var)
        else:
            item.relations.append(
                RelationSpec(
                    kind="unsupported",
                    raw_relation=goal,
                    reason=f"block_initial_goal_satisfaction: no Not(...) mapping for kind {kind!r}",
                )
            )
            continue

        # Record where the item is supposed to end up (first goal wins
        # when multiple goals touch the same item — Layer 2b can refine).
        if item.goal_binding is None:
            item.goal_binding = GoalBindingSpec(kind=kind, target_var=target_var, target_name=tgt)

        item.relations.append(RelationSpec(kind="not", inner=inner))

    return placement


def propose_placement(
    resolved: ResolvedScene, spec: SceneSpec, attempt: int = 0, seed: int | None = None
) -> Placement:
    """Turn a resolved scene into a Placement with no I/O side effects.

    ``attempt`` selects which robot-placement sample to draw — bumping it
    yields a different angular sample so the auto-retry driver can
    re-sample after an IK-feasibility failure without reseeding the rest
    of the pipeline.

    ``seed`` is folded into the robot-placement RNG seed so a different
    value (typically the user-controlled ``--seed``) yields a different
    angular sample for the same (env_name, attempt) pair — useful when
    the default seed keeps placing the robot at an unhelpful corner.
    """
    background_name = resolved.background.name if resolved.background else spec.background

    env_name = _derive_env_name(resolved, spec)
    class_name = _derive_class_name(resolved, spec)

    item_vars: dict[str, str] = {instance_name: f"{_safe_var(instance_name)}_obj" for instance_name in resolved.items}

    tabletop_plan = _plan_tabletop_anchor(background_name)

    items = _propose_items(resolved, spec, item_vars, tabletop_plan)

    task_plan = _plan_task(resolved, spec, item_vars, background_name)

    extra_assets: list[str] = []
    if tabletop_plan.kind == "reference":
        extra_assets.append("tabletop_anchor")

    robot_placement = (
        _propose_robot_placement(env_name, attempt, seed=seed)
        if tabletop_plan.emit_position_limits
        else None
    )

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
        robot_placement=robot_placement,
    )


def _propose_robot_placement(env_name: str, attempt: int = 0, seed: int | None = None) -> RobotPlacement:
    """Sample a robot placement along one of the four tabletop edges.

    Each edge is drawn with equal probability from
    ``{x_min, x_max, y_min, y_max}`` and the position along the edge
    is drawn from ``Uniform(*_ROBOT_EDGE_FRACTION_RANGE)`` — currently
    [0.3, 0.7], so the robot can land anywhere in the middle two
    thirds of the edge while still avoiding the corners. The seed is
    ``f"{env_name}#{attempt}#{seed}"`` so a fixed
    (env_name, attempt, seed) triple stays reproducible; bumping
    ``attempt`` walks through alternative samples when IK feasibility
    fails, and varying ``--seed`` shifts the whole sequence.

    The base rotation is one of four cardinal yaws (per-edge), so the
    gripper is always axis-aligned with the side the robot is on and
    points across the table at the objects.
    """
    rng = random.Random(f"{env_name}#{attempt}#{seed}")
    edge = rng.choice(list(_EDGE_ROTATION_XYZW.keys()))
    fraction = rng.uniform(*_ROBOT_EDGE_FRACTION_RANGE)
    return RobotPlacement(
        edge=edge,
        fraction=fraction,
        offset_m=_ROBOT_EDGE_OFFSET_M,
        z_m=0.0,
        rotation_xyzw=_EDGE_ROTATION_XYZW[edge],
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


def _background_family(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """Short alias for the background, used in env / class names."""
    bg_name = resolved.background.name if resolved.background else spec.background
    return _BACKGROUND_NAME_ALIASES.get(bg_name, _compact(bg_name))


def _derive_env_name(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """Compact env name from the goal diff, e.g. 'avocadoPnPbowltable'."""
    if resolved.goal_added:
        diff = resolved.goal_added[0]
        verb = _VERB_CODES.get(diff["kind"], diff["kind"].capitalize())
        primary = _compact(diff["subject"])
        secondary = _compact(diff["target"] or "")  # target is None for open/closed unary goals
    else:
        verb = "Env"
        primary = _compact(next(iter(resolved.items), "llm"))
        secondary = ""
    return f"{primary}{verb}{secondary}{_background_family(resolved, spec)}" or "llmEnv"


def _derive_class_name(resolved: ResolvedScene, spec: SceneSpec) -> str:
    """CamelCase variant with each segment capitalized, verb code preserved."""
    if resolved.goal_added:
        diff = resolved.goal_added[0]
        verb = _VERB_CODES.get(diff["kind"], diff["kind"].capitalize())
        primary = _compact(diff["subject"]).capitalize()
        secondary = _compact(diff["target"] or "").capitalize()  # target is None for open/closed
    else:
        verb = "Env"
        primary = _compact(next(iter(resolved.items), "llm")).capitalize()
        secondary = ""
    return f"{primary}{verb}{secondary}{_background_family(resolved, spec).capitalize()}Environment"


# ---------------------------------------------------------------------------
# Tabletop anchor planning
# ---------------------------------------------------------------------------


def _plan_tabletop_anchor(background_name: str) -> TabletopAnchorPlan:
    """Decide whether to anchor to the background, an ObjectReference, or nothing."""
    if background_name not in _BACKGROUND_TABLETOP_ANCHOR:
        return TabletopAnchorPlan(kind="none", anchor_var="background", prim_path=None, emit_position_limits=False)
    prim = _BACKGROUND_TABLETOP_ANCHOR[background_name]
    anchor_object_type = "BASE" if background_name in _BACKGROUND_TABLETOP_ANCHOR_BASE_TYPE else "RIGID"
    if prim is None:
        return TabletopAnchorPlan(
            kind="background",
            anchor_var="background",
            prim_path=None,
            emit_position_limits=True,
            anchor_object_type=anchor_object_type,
        )
    return TabletopAnchorPlan(
        kind="reference",
        anchor_var="tabletop_anchor",
        prim_path=prim,
        emit_position_limits=True,
        anchor_object_type=anchor_object_type,
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
    background_name = resolved.background.name if resolved.background else spec.background

    # Items that are the subject of an open/close door goal need is_openable=True
    # so that RotateAroundSolution is added to orient the door toward the robot.
    openable_subjects = {
        r["subject"]
        for r in resolved.goal_added + resolved.goal_removed
        if r["kind"] in _OPEN_DOOR_KINDS | _CLOSE_DOOR_KINDS
    }

    items: list[PlacementItem] = []
    for instance_name, cls in resolved.items.items():
        items.append(
            PlacementItem(
                var_name=item_vars[instance_name],
                asset_name=cls.name,
                instance_name=instance_name,
                relations=[],
                is_openable=(instance_name in openable_subjects),
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
        if rel["kind"] in _OPEN_DOOR_KINDS | _CLOSE_DOOR_KINDS:
            # State annotations (open/closed) — not placement relations; skip.
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
        elif rel["kind"] == "in":
            tgt = rel["target"]
            # In targets a container item, not the background. In clamps
            # XY to the container's footprint and lets gravity resolve Z
            # on the first physics tick, so no PositionLimits is emitted.
            target_var = item_vars.get(tgt)
            if target_var is None:
                item.relations.append(
                    RelationSpec(
                        kind="unsupported",
                        raw_relation=rel,
                        reason=f"unknown target {tgt!r} for In",
                    )
                )
                continue
            item.relations.append(RelationSpec(kind="in", in_target_var=target_var))
        else:
            item.relations.append(
                RelationSpec(
                    kind="unsupported",
                    raw_relation=rel,
                    reason=f"relation kind {rel['kind']!r} has no generator support yet",
                )
            )

    # For openable items that received an On(tabletop_anchor) relation, append
    # RotateAroundSolution so the door faces the robot (at world origin).
    for item in items:
        if item.is_openable and any(r.kind == "on" for r in item.relations):
            facing_yaw = _APPLIANCE_FACING_YAW.get((background_name, item.asset_name), _DEFAULT_APPLIANCE_FACING_YAW)
            item.relations.append(RelationSpec(kind="rotate_around_solution", rotate_yaw_rad=facing_yaw))

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
    if len(resolved.goal_added) == 1:
        g = resolved.goal_added[0]
        # OpenDoorTask / CloseDoorTask: unary goal on an openable item.
        if g["kind"] in _OPEN_DOOR_KINDS:
            subj_var = item_vars.get(g["subject"])
            if subj_var is not None:
                return _open_door_plan(resolved, spec, g, subj_var)
        elif g["kind"] in _CLOSE_DOOR_KINDS:
            subj_var = item_vars.get(g["subject"])
            if subj_var is not None:
                return _close_door_plan(resolved, spec, g, subj_var)
        # PickAndPlaceTask: exactly one on/in goal between two resolved
        # items (not the background — PickAndPlaceTask uses the destination
        # for contact-sensor filtering, and the whole-scene background is a
        # poor candidate).
        elif g["kind"] in _PICK_AND_PLACE_KINDS:
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


def _open_door_plan(
    resolved: ResolvedScene,
    spec: SceneSpec,
    goal: dict[str, Any],
    subject_var: str,
) -> TaskPlan:
    comments = [f"        # goal_added: open({goal['subject']}, None) — enforced by OpenDoorTask success predicate"]
    for r in resolved.goal_removed:
        comments.append(
            f"        # goal_removed (implied when door is opened): {r['kind']}({r['subject']}, {r['target']})"
        )
    return TaskPlan(
        kind="open_door",
        task_import="from isaaclab_arena.tasks.open_door_task import OpenDoorTask",
        task_expr=(
            "OpenDoorTask(\n"
            f"                openable_object={subject_var},\n"
            f"                episode_length_s={_DEFAULT_OPEN_DOOR_EPISODE_LENGTH_S},\n"
            f"                task_description={spec.task_description!r},\n"
            "            )"
        ),
        goal_comments=comments,
        header_note=f"OpenDoorTask: openable={subject_var.removesuffix('_obj')}",
    )


def _close_door_plan(
    resolved: ResolvedScene,
    spec: SceneSpec,
    goal: dict[str, Any],
    subject_var: str,
) -> TaskPlan:
    comments = [f"        # goal_added: closed({goal['subject']}, None) — enforced by CloseDoorTask success predicate"]
    for r in resolved.goal_removed:
        comments.append(
            f"        # goal_removed (implied when door is closed): {r['kind']}({r['subject']}, {r['target']})"
        )
    return TaskPlan(
        kind="close_door",
        task_import="from isaaclab_arena.tasks.close_door_task import CloseDoorTask",
        task_expr=(
            "CloseDoorTask(\n"
            f"                openable_object={subject_var},\n"
            f"                episode_length_s={_DEFAULT_OPEN_DOOR_EPISODE_LENGTH_S},\n"
            f"                task_description={spec.task_description!r},\n"
            "            )"
        ),
        goal_comments=comments,
        header_note=f"CloseDoorTask: openable={subject_var.removesuffix('_obj')}",
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
        if g["kind"] in _OPEN_DOOR_KINDS | _CLOSE_DOOR_KINDS:
            reason = f"goal kind {g['kind']!r} — subject {g['subject']!r} did not resolve to a known item"
        elif g["kind"] not in _PICK_AND_PLACE_KINDS:
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
