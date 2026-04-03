"""Translate LLM-generated predicates into Arena Objects with both:
1. RoboLab ObjectStates (for spatial solver) — position solving
2. Arena Relations (On, NextTo, Inside) — stored for env gen reuse

LLM predicate → RoboLab ObjectState + Arena Relation
─────────────────────────────────────────────────────
place-on-base (x,y) → ObjectState.x/y + On(table) + AtPosition(x,y)
left-of (ref, dist)  → RelativePositionPredicate + On(table) + NextTo(ref, +Y)
right-of             → RelativePositionPredicate + On(table) + NextTo(ref, -Y)
front-of             → RelativePositionPredicate + On(table) + NextTo(ref, +X)
back-of              → RelativePositionPredicate + On(table) + NextTo(ref, -X)
place-on (support)   → PlaceOnPredicate + On(support)
place-in (container) → PlaceInPredicate + Inside(container)
place-anywhere       → ObjectState (random) + On(table)
random-rot           → ObjectState.yaw + RotateAroundSolution
facing-front/back/.. → ObjectState.yaw + RotateAroundSolution
"""

from __future__ import annotations

import math
import random
from typing import Optional

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.relations.relations import (
    AtPosition,
    Inside,
    IsAnchor,
    NextTo,
    On,
    RotateAroundSolution,
    Side,
)
from isaaclab_arena.scene_gen.arena_asset_manager import ArenaAssetManager, DEFAULT_TABLE_BOUNDS
from isaaclab_arena.scene_gen.predicates import (
    ObjectState,
    PlaceOnBasePredicate,
    PlaceOnPredicate,
    PlaceInPredicate,
    RelativePositionPredicate,
    PredicateType,
)

DEFAULT_CLEARANCE_M = 0.02


def translate_predicates(
    llm_result: dict,
    table_asset,
    asset_manager: ArenaAssetManager,
) -> list:
    """Translate LLM predicates into Arena Objects.

    Each object gets:
    - `_llm_position` dict for the spatial solver
    - `_object_state` ObjectState for the spatial solver
    - Arena Relations (On, NextTo, Inside, etc.) for env gen reuse

    Args:
        llm_result: Dict from LLMAgent {"objects": [...], "predicates": [...]}.
        table_asset: The table Object (must have IsAnchor).
        asset_manager: ArenaAssetManager for dims lookup.

    Returns:
        List of Arena Object instances ready for adaptive_placer.
    """
    registry = AssetRegistry()

    selected_objects = llm_result.get("objects", [])
    predicates = llm_result.get("predicates", [])

    # Instantiate Arena Objects
    obj_map: dict[str, object] = {}
    instance_count: dict[str, int] = {}  # Track duplicates

    for obj_data in selected_objects:
        name = obj_data["name"]

        # Fix LLM duplicate naming: "peach_003_1" → try "peach_003"
        # The LLM appends _N when it wants multiple instances of the same object
        original_name = name
        registry_name = name

        try:
            cls = registry.get_asset_by_name(registry_name)
        except (AssertionError, KeyError):
            # Try stripping trailing _N suffix (LLM duplicate naming)
            import re
            stripped = re.sub(r'_\d+$', '', name)
            if stripped != name:
                try:
                    cls = registry.get_asset_by_name(stripped)
                    registry_name = stripped
                except (AssertionError, KeyError):
                    print(f"[PredicateTranslator] WARNING: '{name}' (tried '{stripped}') not in registry, skipping")
                    continue
            else:
                print(f"[PredicateTranslator] WARNING: '{name}' not in registry, skipping")
                continue

        # Handle duplicate instances: give unique instance_name
        instance_count[registry_name] = instance_count.get(registry_name, 0) + 1
        count = instance_count[registry_name]

        try:
            if count > 1:
                obj = cls(instance_name=f"{registry_name}_{count}")
            else:
                obj = cls()
        except TypeError:
            # Some classes don't accept instance_name
            obj = cls()
        obj._llm_position = {}
        obj._object_state = ObjectState(name=original_name)
        obj_map[original_name] = obj

    # Process predicates
    for pred in predicates:
        pred_type = pred.get("type", "")
        obj_name = pred.get("object", "")

        if obj_name not in obj_map:
            continue

        obj = obj_map[obj_name]
        state = obj._object_state

        # --- place-on-base: direct table placement ---
        if pred_type == "place-on-base":
            x = pred.get("x")
            y = pred.get("y")
            if x is not None:
                state.x = float(x)
                obj._llm_position['x'] = float(x)
            if y is not None:
                state.y = float(y)
                obj._llm_position['y'] = float(y)
            # Arena relations
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))
            if x is not None and y is not None:
                obj.add_relation(AtPosition(x=float(x), y=float(y)))

        # --- Rotation predicates ---
        elif pred_type == "random-rot":
            yaw = random.uniform(0, 360)
            state.yaw = yaw
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=math.radians(yaw)))

        elif pred_type == "facing-front":
            state.yaw = 0.0
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0))

        elif pred_type == "facing-back":
            state.yaw = 180.0
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=math.pi))

        elif pred_type == "facing-left":
            state.yaw = 90.0
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=math.pi / 2))

        elif pred_type == "facing-right":
            state.yaw = 270.0
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=-math.pi / 2))

        # --- Relative spatial predicates ---
        elif pred_type == "left-of":
            ref_name = pred.get("reference", pred.get("ref", ""))
            distance = float(pred.get("distance", 0.15))
            ref_obj = obj_map.get(ref_name)
            # RoboLab predicate (for spatial solver)
            state.predicates.append(RelativePositionPredicate(
                type=PredicateType.LEFT_OF, target_object=obj_name,
                reference_object=ref_name, distance=distance))
            # Arena relations
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))
            if ref_obj:
                obj.add_relation(NextTo(ref_obj, side=Side.POSITIVE_Y, distance_m=distance))

        elif pred_type == "right-of":
            ref_name = pred.get("reference", pred.get("ref", ""))
            distance = float(pred.get("distance", 0.15))
            ref_obj = obj_map.get(ref_name)
            state.predicates.append(RelativePositionPredicate(
                type=PredicateType.RIGHT_OF, target_object=obj_name,
                reference_object=ref_name, distance=distance))
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))
            if ref_obj:
                obj.add_relation(NextTo(ref_obj, side=Side.NEGATIVE_Y, distance_m=distance))

        elif pred_type == "front-of":
            ref_name = pred.get("reference", pred.get("ref", ""))
            distance = float(pred.get("distance", 0.15))
            ref_obj = obj_map.get(ref_name)
            state.predicates.append(RelativePositionPredicate(
                type=PredicateType.FRONT_OF, target_object=obj_name,
                reference_object=ref_name, distance=distance))
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))
            if ref_obj:
                obj.add_relation(NextTo(ref_obj, side=Side.POSITIVE_X, distance_m=distance))

        elif pred_type == "back-of":
            ref_name = pred.get("reference", pred.get("ref", ""))
            distance = float(pred.get("distance", 0.15))
            ref_obj = obj_map.get(ref_name)
            state.predicates.append(RelativePositionPredicate(
                type=PredicateType.BACK_OF, target_object=obj_name,
                reference_object=ref_name, distance=distance))
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))
            if ref_obj:
                obj.add_relation(NextTo(ref_obj, side=Side.NEGATIVE_X, distance_m=distance))

        # --- Physical predicates ---
        elif pred_type == "place-on":
            support_name = pred.get("support", "")
            support_obj = obj_map.get(support_name)
            # RoboLab predicate
            state.predicates.append(PlaceOnPredicate(
                target_object=obj_name,
                support_object=support_name))
            # Arena relation
            if support_obj:
                obj.add_relation(On(support_obj, clearance_m=DEFAULT_CLEARANCE_M))
            else:
                obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))

        elif pred_type == "place-in":
            container_name = pred.get("container", "")
            container_obj = obj_map.get(container_name)
            # RoboLab predicate
            state.predicates.append(PlaceInPredicate(
                target_objects=[obj_name],
                container=container_name))
            # Arena relation
            if container_obj:
                obj.add_relation(Inside(container_obj, clearance_m=DEFAULT_CLEARANCE_M))
            else:
                obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))

        elif pred_type == "place-anywhere":
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))

        else:
            print(f"[PredicateTranslator] WARNING: Unknown predicate '{pred_type}'")

    # Ensure all objects have at least On(table)
    for name, obj in obj_map.items():
        rels = obj.get_relations()
        has_spatial = any(isinstance(r, (On, Inside, NextTo, AtPosition)) for r in rels)
        if not has_spatial:
            obj.add_relation(On(table_asset, clearance_m=DEFAULT_CLEARANCE_M))

        # Ensure yaw is set
        state = obj._object_state
        if state.yaw is None:
            state.yaw = random.uniform(0, 360)
            obj.add_relation(RotateAroundSolution(
                roll_rad=0.0, pitch_rad=0.0, yaw_rad=math.radians(state.yaw)))

    return list(obj_map.values())
