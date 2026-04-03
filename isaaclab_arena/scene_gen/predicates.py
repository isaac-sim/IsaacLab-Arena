"""Predicate definitions for scene generation.

This module defines spatial and physical predicates used to specify object placement
and relationships in generated scenes. These predicates are resolved by the spatial
and physical solvers to create valid scene configurations.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from enum import Enum


class PredicateType(Enum):
    """Types of predicates for scene generation."""

    # Spatial predicates (2D placement in x-y plane)
    LEFT_OF = "left-of"
    RIGHT_OF = "right-of"
    FRONT_OF = "front-of"
    BACK_OF = "back-of"
    PLACE_ON_BASE = "place-on-base"
    ALIGN_LEFT = "align-left"
    ALIGN_RIGHT = "align-right"
    ALIGN_FRONT = "align-front"
    ALIGN_BACK = "align-back"
    ALIGN_CENTER_LR = "align-center-lr"
    ALIGN_CENTER_FB = "align-center-fb"
    FACING_LEFT = "facing-left"
    FACING_RIGHT = "facing-right"
    FACING_FRONT = "facing-front"
    FACING_BACK = "facing-back"
    RANDOM_ROT = "random-rot"

    # Physical predicates (3D placement with physics)
    PLACE_ON = "place-on"
    PLACE_IN = "place-in"
    PLACE_ANYWHERE = "place-anywhere"


@dataclass
class Predicate:
    """Base class for all predicates."""

    type: PredicateType
    target_object: str

    def __repr__(self):
        return f"{self.type.value}({self.target_object})"


@dataclass
class SpatialPredicate(Predicate):
    """Spatial predicate for 2D placement."""

    reference_object: Optional[str] = None
    distance: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    yaw: Optional[float] = None

    def __repr__(self):
        parts = [f"{self.type.value}({self.target_object}"]
        if self.reference_object:
            parts.append(f", ref={self.reference_object}")
        if self.distance is not None:
            parts.append(f", dist={self.distance:.3f}")
        if self.x is not None:
            parts.append(f", x={self.x:.3f}")
        if self.y is not None:
            parts.append(f", y={self.y:.3f}")
        if self.yaw is not None:
            parts.append(f", yaw={self.yaw:.1f}")
        parts.append(")")
        return "".join(parts)


@dataclass
class PhysicalPredicate(Predicate):
    """Physical predicate for 3D placement with physics."""

    support_object: Optional[str] = None
    support_ratio: float = 0.5  # Ratio of contact area to bottom area
    stability_preference: Literal["stable", "unstable", "neutral"] = "stable"
    relative_position: Optional[str] = None  # e.g., "center", "edge", "left-edge"

    def __repr__(self):
        parts = [f"{self.type.value}({self.target_object}"]
        if self.support_object:
            parts.append(f", support={self.support_object}")
        parts.append(f", ratio={self.support_ratio:.2f}")
        parts.append(f", stability={self.stability_preference}")
        if self.relative_position:
            parts.append(f", pos={self.relative_position}")
        parts.append(")")
        return "".join(parts)


@dataclass
class PlaceOnBasePredicate(SpatialPredicate):
    """Place object on the base surface (table)."""

    def __init__(
        self,
        target_object: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        yaw: Optional[float] = None,
    ):
        super().__init__(
            type=PredicateType.PLACE_ON_BASE,
            target_object=target_object,
            x=x,
            y=y,
            yaw=yaw,
        )


@dataclass
class RelativePositionPredicate(SpatialPredicate):
    """Place object relative to another in 2D."""

    def __init__(
        self,
        target_object: str,
        reference_object: str,
        direction: PredicateType,
        distance: float = 0.1,
    ):
        super().__init__(
            type=direction,
            target_object=target_object,
            reference_object=reference_object,
            distance=distance,
        )


@dataclass
class PlaceOnPredicate(PhysicalPredicate):
    """Place object on top of another object."""

    def __init__(
        self,
        target_object: str,
        support_object: str,
        support_ratio: float = 0.5,
        stability_preference: Literal["stable", "unstable", "neutral"] = "stable",
        relative_position: Optional[str] = None,
    ):
        super().__init__(
            type=PredicateType.PLACE_ON,
            target_object=target_object,
            support_object=support_object,
            support_ratio=support_ratio,
            stability_preference=stability_preference,
            relative_position=relative_position,
        )


@dataclass
class PlaceInPredicate(PhysicalPredicate):
    """Place objects inside a container."""

    def __init__(self, target_objects: list[str], container: str):
        # For place-in, we'll use the first object as target and store the rest
        super().__init__(
            type=PredicateType.PLACE_IN,
            target_object=target_objects[0] if target_objects else "",
            support_object=container,
        )
        self.target_objects = target_objects


@dataclass
class ObjectState:
    """Represents the state of an object during scene generation."""

    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    is_placed: bool = False
    predicates: list[Predicate] = field(default_factory=list)

    def is_fully_solved(self) -> bool:
        """Check if object has all required coordinates determined."""
        return self.x is not None and self.y is not None and self.yaw is not None

    def __repr__(self):
        status = "PLACED" if self.is_placed else "PENDING"
        coords = []
        if self.x is not None:
            coords.append(f"x={self.x:.3f}")
        if self.y is not None:
            coords.append(f"y={self.y:.3f}")
        if self.z is not None:
            coords.append(f"z={self.z:.3f}")
        if self.yaw is not None:
            coords.append(f"yaw={self.yaw:.1f}")
        coord_str = ", ".join(coords) if coords else "not positioned"
        return f"ObjectState({self.name}, {status}, {coord_str})"


def parse_predicates_from_dict(pred_dict: dict) -> Predicate:
    """Parse a predicate from a dictionary returned by LLM."""
    pred_type_str = pred_dict.get("type", "")
    target = pred_dict.get("object", "")

    try:
        pred_type = PredicateType(pred_type_str)
    except ValueError:
        raise ValueError(f"Unknown predicate type: {pred_type_str}")

    # Spatial predicates
    if pred_type == PredicateType.PLACE_ON_BASE:
        return PlaceOnBasePredicate(
            target_object=target,
            x=pred_dict.get("x"),
            y=pred_dict.get("y"),
            yaw=pred_dict.get("yaw"),
        )

    elif pred_type in [
        PredicateType.LEFT_OF,
        PredicateType.RIGHT_OF,
        PredicateType.FRONT_OF,
        PredicateType.BACK_OF,
    ]:
        return RelativePositionPredicate(
            target_object=target,
            reference_object=pred_dict.get("reference", ""),
            direction=pred_type,
            distance=pred_dict.get("distance", 0.1),
        )

    elif pred_type in [
        PredicateType.ALIGN_LEFT,
        PredicateType.ALIGN_RIGHT,
        PredicateType.ALIGN_FRONT,
        PredicateType.ALIGN_BACK,
        PredicateType.ALIGN_CENTER_LR,
        PredicateType.ALIGN_CENTER_FB,
    ]:
        return SpatialPredicate(
            type=pred_type,
            target_object=target,
            reference_object=pred_dict.get("reference"),
        )

    # Physical predicates
    elif pred_type == PredicateType.PLACE_ON:
        return PlaceOnPredicate(
            target_object=target,
            support_object=pred_dict.get("support", ""),
            support_ratio=pred_dict.get("support_ratio", 0.5),
            stability_preference=pred_dict.get("stability", "stable"),
            relative_position=pred_dict.get("position"),
        )

    elif pred_type == PredicateType.PLACE_IN:
        return PlaceInPredicate(
            target_objects=pred_dict.get("objects", [target]),
            container=pred_dict.get("container", ""),
        )

    elif pred_type == PredicateType.PLACE_ANYWHERE:
        return PhysicalPredicate(type=pred_type, target_object=target)

    else:
        # Generic spatial predicate
        return SpatialPredicate(
            type=pred_type,
            target_object=target,
            reference_object=pred_dict.get("reference"),
            yaw=pred_dict.get("yaw"),
        )
