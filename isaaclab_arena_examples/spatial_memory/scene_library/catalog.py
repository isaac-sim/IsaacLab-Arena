# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The thirty short motion scenes, in field-guide order."""

import random
import torch
from functools import partial

from isaaclab_arena_environments.spatial_memory.g1_adapter import G1_FLOOR_Z_M, G1Adapter
from isaaclab_arena_environments.spatial_memory.parts import PALETTE
from isaaclab_arena_environments.spatial_memory.tasks.partition_delivery import PartitionDeliveryLogic
from isaaclab_arena_environments.spatial_memory.tasks.reverse_gap import ReverseGapLogic
from isaaclab_arena_environments.spatial_memory.tasks.shelf_bay import ShelfBayLogic
from isaaclab_arena_examples.spatial_memory.scene_library import locomotion, manipulation


def _pilot(logic_type, scene, rng):
    """Use the pilot's own seeded fixtures, with floor-relative Z in the exported asset."""
    from pxr import Gf, UsdGeom

    logic = logic_type(None, G1Adapter())
    layout = logic.sample_layout(torch.Generator().manual_seed(rng.getrandbits(32)))
    for spec, pose in zip(logic.part_specs(), layout.part_poses_e, strict=True):
        x, y, z, qx, qy, qz, qw = pose.tolist()
        path = scene.box(spec.name, (x, y, z - G1_FLOOR_Z_M), spec.size_xyz, PALETTE[spec.color])
        node = UsdGeom.Xformable(scene.stage.GetPrimAtPath(path))
        # Insert orientation before nonuniform scaling, preserving the pilot's rigid transform.
        translation, _, scale = node.GetOrderedXformOps()
        orient = node.AddOrientOp()
        orient.Set(Gf.Quatf(qw, qx, qy, qz))
        node.SetXformOpOrder([translation, orient, scale])


PILOT_SCENES = {
    "reverse_gap": partial(_pilot, ReverseGapLogic),
    "shelf_bay": partial(_pilot, ShelfBayLogic),
    "partition_delivery": partial(_pilot, PartitionDeliveryLogic),
}
SCENES = dict(PILOT_SCENES)
SCENES.update(
    (builder.__name__, builder)
    for builder in (
        manipulation.cabinet_reach,
        manipulation.tote_extraction,
        manipulation.hanging_pan,
        manipulation.valve_turn,
        manipulation.seam_inspection,
        manipulation.panel_catches,
        manipulation.dishwasher,
        manipulation.billet_insertion,
        manipulation.rail_support,
        manipulation.stack_bracing,
        manipulation.worktop_clamp,
        manipulation.loaded_drawer,
        manipulation.pipe_bracket,
        manipulation.panel_removal,
        manipulation.hatch_transfer,
        manipulation.probe_withdrawal,
        manipulation.display_cover,
        manipulation.scoop_pour,
        locomotion.panel_corner,
        locomotion.aisle_retreat,
        locomotion.platform_step,
        locomotion.beam_clearance,
        locomotion.trolley_dock,
        locomotion.pillar_pass,
        locomotion.closing_door,
        locomotion.moving_carriage,
        locomotion.descending_hood,
    )
)


def build_stage(name: str, seed: int):
    """Build a self-contained USD stage, sampling each scene independently from its name and seed."""
    from pxr import Usd

    from isaaclab_arena_examples.spatial_memory.scene_library.geometry import Workcell

    stage = Usd.Stage.CreateInMemory()
    workcell = Workcell(stage)
    SCENES[name](workcell, random.Random(f"{seed}/{name}"))
    if name in PILOT_SCENES:
        # The legacy loading lanes open along X. Look into the bays for the second view.
        workcell.finish(approach=(2.8, -0.2, 1.15))
    else:
        workcell.finish()
    return stage
