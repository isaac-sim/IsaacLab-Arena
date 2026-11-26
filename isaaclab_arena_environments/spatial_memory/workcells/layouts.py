# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collision inserts and shared guards for the two layouts illustrated in the G1 guide."""

from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class WorkcellLayout:
    """Position one decision-relevant collision feature inside a shared set of guards."""

    instruction: str
    feature: str
    size: Vec3
    centers: tuple[Vec3, Vec3]
    guards: tuple[tuple[str, Vec3, Vec3], ...] = ()


LAYOUTS = {
    "reverse_gap": WorkcellLayout(
        "Back through this gap.",
        "JambFoot",
        (0.18, 0.4, 1.16),
        ((-1.15, 0.163, 0.58), (-1.15, 0.683, 0.58)),
        guards=(
            ("LeftJambPost", (-1.15, -0.0969776, 0.63), (0.18, 0.12, 1.26)),
            ("RightJambPost", (-1.15, 0.9430225, 0.63), (0.18, 0.12, 1.26)),
        ),
    ),
    "shelf_bay": WorkcellLayout(
        "Seat the carton in this bay.", "BayStop", (0.09, 0.39, 0.2), ((-1.99, -0.74, 0.855), (-1.99, -0.255, 0.855))
    ),
    "partition_delivery": WorkcellLayout(
        "Cross this marked threshold.", "ThresholdCurb", (0.14, 0.12, 0.16), ((-1.38, 0.25, 0.08), (-1.38, 0.63, 0.08))
    ),
    "cabinet_reach": WorkcellLayout(
        "Touch the rear pad.",
        "DoorSideReturn",
        (0.12, 0.08, 0.15),
        ((-0.38, -0.04, 1.04), (-0.38, 0.09, 1.04)),
        guards=(("CabinetSide", (-0.45, 0.065, 1.095), (0.035, 0.46, 0.53)),),
    ),
    "tote_extraction": WorkcellLayout(
        "Withdraw the tote onto the ledge.",
        "UndershelfCleat",
        (0.38, 0.055, 0.065),
        ((0, -0.04, 1.176), (0, 0.12, 1.176)),
    ),
    "hanging_pan": WorkcellLayout(
        "Unhook the pan.",
        "HookKeeper",
        (0.058, 0.025, 0.035),
        ((-0.138, 0.095, 1.36), (-0.082, 0.095, 1.36)),
        guards=(("RackShelf", (-0.03, 0.05, 1.5), (0.88, 0.53, 0.032)),),
    ),
    "valve_turn": WorkcellLayout(
        "Turn the recessed lever to its stop.",
        "GuardRib",
        (0.055, 0.095, 0.065),
        ((-0.09, 0.08, 1.26), (0.06, 0.08, 1.26)),
        guards=(
            ("ValveGuardFlange", (0, -0.115, 1.305), (0.6, 0.22, 0.044)),
            ("FrontGuardLip", (0, -0.205, 1.256), (0.6, 0.035, 0.092)),
        ),
    ),
    "seam_inspection": WorkcellLayout(
        "Run the probe along the entire seam in one pass.",
        "RearSeam",
        (0.024, 0.1, 0.028),
        ((0.282, 0.095, 1.025), (0.282, 0.095, 1.09)),
    ),
    "panel_catches": WorkcellLayout(
        "Depress the two catches together.",
        "LeftCatch",
        (0.037, 0.025, 0.046),
        ((-0.286, 0.036, 1.06), (-0.214, 0.036, 1.06)),
        guards=(
            ("LeftServiceHood", (-0.25, -0.105, 1.15), (0.23, 0.2, 0.036)),
            ("CatchMount", (-0.25, 0.058, 1.06), (0.14, 0.03, 0.055)),
        ),
    ),
    "dishwasher": WorkcellLayout(
        "Seat this plate in this rack slot.",
        "RackFinger",
        (0.025, 0.035, 0.15),
        ((-0.1, 0.068, 0.962), (-0.1, 0.022, 0.962)),
        guards=(("FingerRunner", (-0.1, 0.09, 0.88), (0.043, 0.19, 0.026)),),
    ),
    "billet_insertion": WorkcellLayout(
        "Insert the billet to the machine stop.",
        "InnerJaw",
        (0.09, 0.09, 0.12),
        ((-0.065, 0.09, 1.002), (0.065, 0.09, 1.002)),
        guards=(("PortalApron", (0, -0.105, 1.305), (0.6, 0.27, 0.045)),),
    ),
    "rail_support": WorkcellLayout(
        "Set the crate on both rails.", "RailEnd", (0.071, 0.095, 0.14), ((0.23, 0.245, 0.89), (0.23, -0.205, 0.89))
    ),
    "stack_bracing": WorkcellLayout(
        "Support the upper crate and pull the spacer free.",
        "RearFoot",
        (0.18, 0.24, 0.2),
        ((-0.19, 0.04, 0.92), (-0.19, 0.12, 0.92)),
    ),
    "worktop_clamp": WorkcellLayout(
        "Tighten the clamp until the worktop is secure.",
        "BearingPad",
        (0.085, 0.085, 0.025),
        ((-0.12, -0.48, 0.807), (0.11, -0.48, 0.807)),
    ),
    "loaded_drawer": WorkcellLayout(
        "Close this drawer.", "CrossbarLug", (0.11, 0.07, 0.07), ((0.05, -0.12, 1.05), (0.25, -0.12, 1.05))
    ),
    "pipe_bracket": WorkcellLayout(
        "Pass the rigid U-bracket around this pipe.",
        "PipeStop",
        (0.044, 0.055, 0.08),
        ((-0.045, 0.14, 1.07), (0.045, 0.14, 1.07)),
        guards=(("RearStopMount", (0, 0.18, 1.07), (0.18, 0.042, 0.085)),),
    ),
    "panel_removal": WorkcellLayout(
        "Rotate the panel out of the frame.",
        "OverhangCleat",
        (0.11, 0.16, 0.06),
        ((0.07, 0.17, 1.14), (0.29, 0.17, 1.14)),
    ),
    "hatch_transfer": WorkcellLayout(
        "Slide the tray through the hatch.",
        "GuideFoot",
        (0.09, 0.06, 0.075),
        ((0.23, 0.085, 0.86), (0.38, 0.085, 0.86)),
    ),
    "probe_withdrawal": WorkcellLayout(
        "Pull the probe free.",
        "BaffleStop",
        (0.05, 0.045, 0.1),
        ((-0.12, 0.075, 1.11), (-0.045, 0.075, 1.11)),
        guards=(
            ("BaffleStopRail", (-0.08, 0.11, 1.11), (0.18, 0.035, 0.11)),
            ("BaffleRailAnchor", (-0.16, 0.16, 1.11), (0.045, 0.13, 0.1)),
        ),
    ),
    "display_cover": WorkcellLayout(
        "Peel the rigid-backed cover off the display.",
        "HoodRib",
        (0.11, 0.07, 0.05),
        ((-0.13, 0.04, 1.208), (0.1, 0.04, 1.208)),
    ),
    "scoop_pour": WorkcellLayout(
        "Tip the scoop into this chute.",
        "HopperLip",
        (0.12, 0.045, 0.055),
        ((-0.08, 0.118, 1.12), (0.085, 0.118, 1.12)),
        guards=(("FarRimRail", (0, 0.17, 1.12), (0.33, 0.08, 0.055)),),
    ),
    "panel_corner": WorkcellLayout(
        "Turn this long load through the corner.",
        "ReturnBumper",
        (0.27, 0.14, 0.18),
        ((-0.8, -0.285, 0.88), (-0.39, -0.06, 0.88)),
    ),
    "aisle_retreat": WorkcellLayout(
        "Back one chassis length out of the aisle.",
        "RackProjection",
        (0.14, 0.19, 0.12),
        ((-0.41, -0.41, 0.94), (-0.41, -0.1, 0.94)),
        guards=(("RackBracket", (-0.49, -0.26, 0.94), (0.055, 0.48, 0.13)),),
    ),
    "platform_step": WorkcellLayout(
        "Step onto this platform with the crate level.",
        "StepCurb",
        (0.12, 0.12, 0.09),
        ((-0.18, 0.02, 0.225), (0.15, 0.02, 0.225)),
    ),
    "beam_clearance": WorkcellLayout(
        "Duck beneath this beam carrying a tall item.",
        "BeamDrop",
        (0.22, 0.26, 0.1),
        ((-0.13, 0.04, 1.44), (0.13, 0.04, 1.44)),
    ),
    "trolley_dock": WorkcellLayout(
        "Pivot this cart into its berth.", "WheelStop", (0.11, 0.12, 0.1), ((-0.18, 0.3, 0.05), (0.18, 0.3, 0.05))
    ),
    "pillar_pass": WorkcellLayout(
        "Move sideways past this pillar.",
        "PlinthToe",
        (0.16, 0.15, 0.11),
        ((-0.17, -0.35, 0.055), (0.17, -0.35, 0.055)),
    ),
    "closing_door": WorkcellLayout(
        "Carry the parcel through the closing door.",
        "JambBumper",
        (0.13, 0.12, 0.15),
        ((-0.43, 0.13, 0.9), (-0.43, 0.3, 0.9)),
        guards=(("JambReturnPlate", (-0.5, 0.2, 0.9), (0.06, 0.36, 0.19)),),
    ),
    "moving_carriage": WorkcellLayout(
        "Take the part from the crossing carriage.",
        "GuardFoot",
        (0.1, 0.085, 0.09),
        ((-0.08, 0.15, 0.97), (0.12, 0.15, 0.97)),
        guards=(
            ("GuardBracket", (0.02, 0.211, 0.97), (0.12, 0.105, 0.09)),
            ("GuardTopFlange", (0.02, 0.16, 1.055), (0.46, 0.17, 0.035)),
            ("GuardFrontLip", (0.02, 0.084, 1.006), (0.46, 0.023, 0.112)),
        ),
    ),
    "descending_hood": WorkcellLayout(
        "Draw the tray clear of the descending hood.",
        "TrayStop",
        (0.09, 0.09, 0.075),
        ((-0.16, 0.16, 0.86), (0.16, 0.16, 0.86)),
    ),
}


# Touching these surfaces is an unintended collision. Other inserts are supports, targets or intended stops.
FORBIDDEN_CONTACT = frozenset({
    "reverse_gap",
    "shelf_bay",
    "partition_delivery",
    "cabinet_reach",
    "tote_extraction",
    "valve_turn",
    "dishwasher",
    "billet_insertion",
    "loaded_drawer",
    "pipe_bracket",
    "panel_removal",
    "hatch_transfer",
    "probe_withdrawal",
    "display_cover",
    "scoop_pour",
    "panel_corner",
    "aisle_retreat",
    "platform_step",
    "beam_clearance",
    "pillar_pass",
    "closing_door",
    "moving_carriage",
    "descending_hood",
})

# Start the carrier beside the load at the entrance, in the same short approach as the illustrations.
PILOT_STARTS = {
    "reverse_gap": ((-0.72, 0.42, 0.78), (0.0, 0.0, 0.0, 1.0), (-0.24, 0.42, 1.0)),
    "shelf_bay": ((-0.89, -0.5, 0.78), (0.0, 0.0, 1.0, 0.0), (-1.46, -0.5, 0.96)),
    "partition_delivery": ((-0.53, 0.47, 0.78), (0.0, 0.0, 1.0, 0.0), (-0.96, 0.465, 0.97)),
}

# Each pair names a fixed bearing surface and the body that must actually load it.
REQUIRED_SUPPORT = {
    "shelf_bay": (("dest_table", "load"),),
    "rail_support": (("Rail0", "crate"), ("Rail1", "crate")),
    "worktop_clamp": (("BearingPad", "jaw"), ("Worktop", "clamp")),
}
