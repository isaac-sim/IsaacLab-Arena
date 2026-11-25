# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Rigid assemblies and passive mechanisms for the short G1 motions."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Mechanism:
    """Constrain a body to a slide (metres) or hinge (degrees) about its origin."""

    kind: Literal["slide", "hinge"]
    axis: Literal["X", "Y", "Z"]
    limits: tuple[float, float] | None

    parent: str | None = None
    """Another body in the same workcell, or the fixed workcell frame."""


@dataclass(frozen=True)
class Body:
    """Collect matching workcell shapes into a single rigid body about a named pivot."""

    name: str
    shapes: str
    pivot: str
    mechanism: Mechanism | None = None
    stand: bool = False

    mass_kg: float = 0.4


def body(name, shapes=None, pivot=None, mechanism=None, stand=False, mass_kg=0.4):
    """Use the same-named shape as the pivot for a single-piece work item."""
    return Body(name, shapes or name, pivot or name, mechanism, stand, mass_kg)


SLIDE_BUTTON = Mechanism("slide", "Y", (0, 0.012))
LOAD = (body("load", stand=True),)
BODIES = {
    "reverse_gap": LOAD,
    "shelf_bay": LOAD + tuple(body(name) for name in ("stock_a", "stock_b", "stock_c")),
    "partition_delivery": LOAD,
    "cabinet_reach": (body("button", "ResetPad", "ResetPad", SLIDE_BUTTON),),
    "tote_extraction": (body("tote", "Tote/* Grip", "Tote"),),
    "hanging_pan": (
        body("pan", "HandleSide0* HandleTop0 Handle0 Pan0", "Pan0"),
        body("neighbor", "HandleSide1* HandleTop1 Handle1 Pan1", "Pan1"),
    ),
    "valve_turn": (body("lever", "Lever Knob", "Spindle", Mechanism("hinge", "Y", (-65, 0))),),
    "seam_inspection": (body("probe", "ProbeGrip ProbeTip", "ProbeTip", stand=True),),
    "panel_catches": (
        body("left", "LeftCatch", "LeftCatch", SLIDE_BUTTON),
        body("right", "Catch1", "Catch1", SLIDE_BUTTON),
    ),
    "dishwasher": (body("plate", "HeldPlate", "HeldPlate", stand=True), body("neighbor", "Neighbor", "Neighbor")),
    "billet_insertion": (body("billet", "Billet", "Billet", stand=True),),
    "rail_support": (body("crate", "Crate", "Crate", stand=True),),
    "stack_bracing": (body("spacer", "Spacer PullTab", "Spacer"), body("crate", "UpperCrate", "UpperCrate")),
    "worktop_clamp": (
        body("clamp", "Spine ClampArm* Screw UpperJaw", "Spine"),
        body("jaw", "LowerJaw", "LowerJaw", Mechanism("slide", "Z", (0, 0.075), "clamp")),
    ),
    "loaded_drawer": (
        body("drawer", "Drawer/*", "Drawer", Mechanism("slide", "Y", (0, 0.25))),
        body("tray", "InnerTray/* ToolHandle", "InnerTray"),
    ),
    "pipe_bracket": (body("bracket", "Bracket/*", "Bracket", stand=True),),
    "panel_removal": (body("panel", "Worktop", "Worktop"),),
    "hatch_transfer": (
        body("tray", "ServingTray/Base ServingTray/End* ServingTray/Side*", "ServingTray"),
        body("cup0", "ServingTray/Cup0", "ServingTray/Cup0"),
        body("cup1", "ServingTray/Cup1", "ServingTray/Cup1"),
    ),
    "probe_withdrawal": (body("probe", "ProbeShaft BentTip ProbeHandle", "BentTip", stand=True),),
    "display_cover": (
        body("backing", "AdheredCover", "AdheredCover"),
        body("cover", "FreeCover", "PeelHinge", Mechanism("hinge", "X", (-90, 90), "backing")),
    ),
    "scoop_pour": (body("scoop", "Scoop/Base Scoop/End* Scoop/Side* Scoop/Handle", "Scoop", stand=True),) + tuple(
        body(f"pellet{i}", f"Scoop/Pellet{i}", f"Scoop/Pellet{i}", mass_kg=0.015) for i in range(6)
    ),
    "panel_corner": (body("panel", "HeldWorktop", "HeldWorktop", stand=True),),
    "aisle_retreat": (body("chassis", "Chassis Grip*", "Chassis", stand=True),),
    "platform_step": (body("crate", "HeldCrate", "HeldCrate", stand=True),),
    "beam_clearance": (body("load", "TallItem", "TallItem", stand=True),),
    "trolley_dock": (body("trolley", "Trolley/Deck Trolley/Grip Trolley/Upright*", "Trolley"),) + tuple(
        body(f"wheel{i}{j}", f"Trolley/Wheel{i}{j}", f"Trolley/Wheel{i}{j}", Mechanism("hinge", "X", None, "trolley"))
        for i in range(2)
        for j in range(2)
    ),
    "pillar_pass": (body("case", "InstrumentCase CaseGrip GripFoot*", "InstrumentCase", stand=True),),
    "closing_door": (
        body("parcel", "Parcel", "Parcel", stand=True),
        body("mover", "Door/*", "Door"),
    ),
    "moving_carriage": (
        body("part", "Carriage/Part Carriage/Handle*", "Carriage/Part"),
        body("mover", "Carriage/Deck", "Carriage"),
    ),
    "descending_hood": (
        body("tray", "Tray/*", "Tray"),
        body("mover", "Hood/*", "Hood"),
    ),
}
