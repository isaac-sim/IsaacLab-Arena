"""GoalSpec goal_relations → Arena Relation instances.

Consumed at scene setup time to place objects before simulation. Sibling
module `predicate_map.py` handles the same predicate vocabulary for a
different purpose (runtime success check). Shared predicate-name handling
lives in `predicate_vocab.py`.

Arena coordinate system (from llm_agent.py):
- +X = front (toward robot), -X = back
- +Y = left, -Y = right

NextTo(parent, side=S): places CHILD on side S of PARENT.
So `obj.add_relation(NextTo(target, side=POSITIVE_Y))` → obj is LEFT of target.

Stacking chains: the LLM emits pairwise `on_top` predicates per pair.
Stack A→B→C produces: [on_top(B,A), on_top(C,B)] → two On() relations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.env_gen.predicate_vocab import normalize_predicate

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


def predicate_to_relations(
    predicate: str,
    obj,
    target=None,
    ref2=None,
    distance_m: float = 0.1,
) -> list:
    """Convert a goal_relation predicate to Arena Relation instances.

    Args:
        predicate: Predicate name (e.g. "in_container", "left_of").
        obj: The Object being manipulated.
        target: Target Object (first reference).
        ref2: Second reference Object (for "between").
        distance_m: Spacing for proximity relations.

    Returns:
        List of Arena Relation instances to add via obj.add_relation(rel).
    """
    from isaaclab_arena.relations.relations import (
        On, Inside, NextTo, AtPosition, Side,
    )

    norm = normalize_predicate(predicate)
    relations = []

    # Containment: object inside container
    if norm in ("in_container", "inside"):
        relations.append(Inside(target))

    # Surface placement: on top of / stacked on
    elif norm in ("on_top", "on", "stacked"):
        relations.append(On(target))

    # Directional — Arena axes: +X=front, +Y=left
    elif norm == "left_of":
        relations.append(NextTo(target, side=Side.POSITIVE_Y, distance_m=distance_m))
    elif norm == "right_of":
        relations.append(NextTo(target, side=Side.NEGATIVE_Y, distance_m=distance_m))
    elif norm == "in_front_of":
        relations.append(NextTo(target, side=Side.POSITIVE_X, distance_m=distance_m))
    elif norm == "behind":
        relations.append(NextTo(target, side=Side.NEGATIVE_X, distance_m=distance_m))
    elif norm == "next_to":
        relations.append(NextTo(target, side=Side.POSITIVE_Y, distance_m=distance_m))

    # Centered on a surface
    elif norm == "center_of":
        relations.append(On(target))
        # Note: ObjectPlacer's On relation already biases toward center
        # For exact centering, caller should add AtPosition with target's XY

    # Between two references: approximate midpoint
    elif norm == "between" and ref2 is not None:
        relations.append(NextTo(target, side=Side.POSITIVE_Y, distance_m=distance_m / 2))
        relations.append(NextTo(ref2, side=Side.NEGATIVE_Y, distance_m=distance_m / 2))

    # In-line: caller should chain these manually per object
    elif norm == "in_line":
        relations.append(NextTo(target, side=Side.POSITIVE_Y, distance_m=distance_m))

    # Exact world position
    elif norm == "at_position":
        if isinstance(target, (tuple, list)):
            relations.append(AtPosition(
                x=target[0], y=target[1],
                z=target[2] if len(target) > 2 else None,
            ))

    return relations


def apply_goal_relations_to_objects(
    goal_spec,
    object_map: dict,
    anchor_name: str = "table",
) -> None:
    """Apply all goal_relations from a GoalSpec to their Arena Objects.

    Adds baseline On(table) for every manipulable + specific goal relations.
    The ObjectPlacer will then solve for valid positions.

    Args:
        goal_spec: GoalSpec instance with goal_relations.
        object_map: Dict mapping object name → Arena Object instance.
        anchor_name: Name of the table/anchor object in object_map.
    """
    from isaaclab_arena.relations.relations import On, IsAnchor

    # Mark table as anchor (fixed during solving)
    if anchor_name in object_map:
        object_map[anchor_name].add_relation(IsAnchor())

    # Baseline: every non-anchor object is on the table
    table = object_map.get(anchor_name)
    if table:
        for name, obj in object_map.items():
            if name != anchor_name:
                obj.add_relation(On(table))

    # Apply goal_relations as additional constraints
    for rel in goal_spec.goal_relations:
        predicate = rel.predicate
        obj_name = rel.object
        target_name = rel.target

        obj = object_map.get(obj_name)
        target = object_map.get(target_name)
        if not obj:
            print(f"[PredicateToRelation] WARNING: object '{obj_name}' not found")
            continue
        if not target and target_name != "table":
            print(f"[PredicateToRelation] WARNING: target '{target_name}' not found")
            continue

        new_rels = predicate_to_relations(
            predicate=predicate,
            obj=obj,
            target=target or table,
        )
        for r in new_rels:
            obj.add_relation(r)
