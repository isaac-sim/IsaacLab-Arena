# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static-base G1 pick-and-place environment (WBC stands the robot in place; no nav).

This is a same-shelf-only variant of ``galileo_g1_locomanip_pick_and_place``: same
``galileo_locomanip`` background, same OpenXR retargeter, same 23-D action layout. The
default embodiment is ``g1_wbc_agile_pink`` (AGILE end-to-end velocity policy) instead of
``g1_wbc_pink`` (HOMIE stand+walk pair) -- the static task never walks, so the AGILE
single-policy backend is a better fit than HOMIE's stand/walk model split. The
``g1_wbc_pink`` embodiment is still accepted via ``--embodiment`` for users who want
HOMIE behaviour. The other differences from the locomanip env are:

1. The destination plate sits on the *same* shelf as the apple (within arm's reach), so
   the robot never needs to drive its base anywhere -- WBC just holds the standing pose.
2. The apple's spawn pose is randomized per episode within ``APPLE_SPAWN_XY_RANGE_M``
   (XY only) so recorded demos have spatial variation; the destination plate stays at a
   fixed pose so the place target is identical across episodes.
"""

from __future__ import annotations

import argparse
import warnings
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Pose tuning constants (all values empirically validated -- see commit history for the
# manual procedure: spawn the destination plate at the locomanip apple z (0.0707) and
# read the final z after gravity-settle; spawn the apple and read its USD AABB):
#
# - SHELF_SURFACE_Z: measured by gravity-settling the plate (whose USD origin sits at
#   its bottom, i.e. BBox min_z = 0); the settled z = -0.030 in env-local frame is the
#   actual shelf-top z in the galileo_locomanip scene.
# - SHELF_AIRGAP: keeps PhysX from spawning objects in collider penetration with the
#   shelf on the first sim tick (which would otherwise launch them upward).
SHELF_SURFACE_Z = -0.030
SHELF_AIRGAP = 0.005
# Cuboid center: top face = SHELF_SURFACE_Z. This assumes procedural_table height is 0.04 m.
SHELF_SUPPORT_PATCH_CENTER = (0.62, 0.0, SHELF_SURFACE_Z - 0.02)

# Object XY spawn pose (env-local frame, shelf-relative). X mirrors the locomanip env
# (the only on-shelf X we have ground-truth data for via the brown_box flow). The pickup
# Y also mirrors locomanip (Y=0.18); the destination is offset -0.24 m in Y so the
# plate's 30 cm footprint clears the apple without collision. Earlier we tried Y=0.30
# for the apple but a smoke test showed it rolls off the shelf edge from there.
PICK_UP_OBJECT_SPAWN_XY = (0.5785, 0.18)
DESTINATION_SPAWN_XY = (0.5785, -0.06)

# Half-range of the apple's per-episode XY randomization at reset, in metres. Mirrors
# the locomanip env's ``XY_RANGE_M = 0.025`` but tightened to 0.020 because the static
# variant places the destination plate on the *same* shelf, so the spawn workspace is
# narrower (apple at Y=0.18, plate's +Y edge at ~0.09 -> 9 cm headroom; 2 cm jitter
# leaves 7 cm minimum gap to the plate). Without this jitter every recorded demo has
# the apple at the exact same XY, which limits the spatial variation a finetuned policy
# can generalize over. The destination plate is left at a fixed Pose so the place
# target is identical across episodes.
APPLE_SPAWN_XY_RANGE_M = 0.020

# Per-asset Z offset from the asset's USD origin to its bottom face. Added on top of
# ``SHELF_SURFACE_Z + SHELF_AIRGAP`` so the asset's *bottom* lands on the shelf rather
# than its USD origin (which may sit anywhere inside the AABB depending on how the
# asset was authored). Measured from each asset's USD AABB. Assets not in this table
# are spawned with no Z compensation -- callers passing arbitrary ``--object`` /
# ``--destination`` values are expected to verify the resulting spawn pose visually.
_USD_ORIGIN_ABOVE_BOTTOM_M: dict[str, float] = {
    "apple_01_objaverse_robolab": 0.0171,  # BBox min_z = -0.019, max_z = 0.049
    "clay_plates_hot3d_robolab": 0.0,  # USD origin at plate bottom (BBox min_z = 0)
}

TUNED_PICK_UP_OBJECT_NAME = "apple_01_objaverse_robolab"
TUNED_DESTINATION_NAME = "clay_plates_hot3d_robolab"

# Per-asset uniform scale matching the tuned pick-up / destination pair. Assets not in
# this table spawn at scale=(1.0, 1.0, 1.0) with a one-shot warning so the user knows
# the resulting size may need visual verification.
_TUNED_SCALES: dict[str, tuple[float, float, float]] = {
    TUNED_PICK_UP_OBJECT_NAME: (0.009, 0.009, 0.009),
    TUNED_DESTINATION_NAME: (0.5, 0.5, 0.5),
}


def _shelf_spawn_z(asset_name: str) -> float:
    """Return the env-local Z to spawn ``asset_name`` flush on the shelf surface.

    Falls back to ``SHELF_SURFACE_Z + SHELF_AIRGAP`` (no USD-origin compensation) for
    assets we have not measured, with a one-shot warning so the user knows the spawn
    pose may need visual verification.
    """
    if asset_name in _USD_ORIGIN_ABOVE_BOTTOM_M:
        return SHELF_SURFACE_Z + SHELF_AIRGAP + _USD_ORIGIN_ABOVE_BOTTOM_M[asset_name]
    warnings.warn(
        "galileo_g1_static_pick_and_place: no measured USD-origin offset for "
        f"'{asset_name}'; spawning at shelf surface with no compensation. Verify "
        "the asset's bottom face actually lands on the shelf.",
        stacklevel=2,
    )
    return SHELF_SURFACE_Z + SHELF_AIRGAP


def _asset_scale(asset_name: str) -> tuple[float, float, float]:
    """Return the tuned uniform scale for ``asset_name``, or 1.0 with a warning."""
    if asset_name in _TUNED_SCALES:
        return _TUNED_SCALES[asset_name]
    warnings.warn(
        "galileo_g1_static_pick_and_place: no measured scale for "
        f"'{asset_name}'; spawning at scale=(1.0, 1.0, 1.0). Verify visually.",
        stacklevel=2,
    )
    return (1.0, 1.0, 1.0)


@register_environment
class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on the locomanip warehouse shelf.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "galileo_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        # Reuse the locomanip background USD: it bakes in lighting and provides the same
        # shelf-in-front-of-robot geometry the locomanip env was tuned against.
        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        # The imported shelf mesh has uneven/perforated collision in the task region:
        # small objects can fall through parts of the visible shelf. Add an invisible
        # kinematic cuboid flush with the shelf top so task objects see a clean support.
        # This has only reproduced in GPU simulation; CPU runs have not shown the issue.
        shelf_support = self.asset_registry.get_asset_by_name("procedural_table")(
            instance_name="static_pick_place_shelf_support",
            prim_path="{ENV_REGEX_NS}/static_pick_place_shelf_support",
        )
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)(scale=_asset_scale(args_cli.object))
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)(
            scale=_asset_scale(args_cli.destination)
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Robot pose mirrors the locomanip env exactly so the WBC controller stands the
        # robot up in the same shelf-relative spot. The controller dynamically lifts the
        # pelvis to ~z=0.74 at runtime; init_state.pos.z=0 is correct.
        embodiment.set_initial_pose(Pose(position_xyz=(0.3, 0.08, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        shelf_support.set_initial_pose(
            Pose(position_xyz=SHELF_SUPPORT_PATCH_CENTER, rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
        destination_x, destination_y = DESTINATION_SPAWN_XY
        pick_up_object_z = _shelf_spawn_z(args_cli.object)
        # ``PoseRange`` registers a ``randomize_object_pose`` reset event so the apple's
        # XY is resampled every episode within ``APPLE_SPAWN_XY_RANGE_M``. Z and rotation
        # are pinned (rpy_min == rpy_max) so the object always lands flush on the shelf
        # in its authored orientation; we only randomize XY. This gives recorded demos
        # spatial variation that lets a finetuned policy generalize over the spawn range.
        pick_up_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(
                    pick_up_object_x - APPLE_SPAWN_XY_RANGE_M,
                    pick_up_object_y - APPLE_SPAWN_XY_RANGE_M,
                    pick_up_object_z,
                ),
                position_xyz_max=(
                    pick_up_object_x + APPLE_SPAWN_XY_RANGE_M,
                    pick_up_object_y + APPLE_SPAWN_XY_RANGE_M,
                    pick_up_object_z,
                ),
                rpy_min=(0.0, 0.0, 0.0),
                rpy_max=(0.0, 0.0, 0.0),
            )
        )
        destination.set_initial_pose(
            Pose(
                position_xyz=(destination_x, destination_y, _shelf_spawn_z(args_cli.destination)),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf and place it onto the "
                f"{destination_label} on the same shelf next to it."
            )

        scene = Scene(assets=[background, shelf_support, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=30.0,
                task_description=task_description,
                # Mirror the locomanip env's success thresholds so metrics are comparable.
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=TUNED_PICK_UP_OBJECT_NAME)
        parser.add_argument("--destination", type=str, default=TUNED_DESTINATION_NAME)
        # Default embodiment is g1_wbc_agile_pink: AGILE end-to-end velocity policy for
        # whole-body balance + PinkIK upper body. The static task never walks, so AGILE's
        # single-policy backend is a better fit than HOMIE's stand+walk split (which
        # ``g1_wbc_pink`` ships). Same 23-D action layout and OpenXR retargeter as the
        # locomanip env -- the only knob that flips is which lower-body ONNX policy gets
        # loaded by the WBC factory. ``g1_wbc_pink`` is still accepted as an override
        # for users who specifically want HOMIE.
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--task_description",
            type=str,
            default=None,
            help=(
                "Override the natural-language task description. Defaults to a template "
                "derived from --object and --destination."
            ),
        )
