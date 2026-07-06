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
2. The apple's spawn pose is configured through ``APPLE_SPAWN_XY_RANGE_M`` (XY only);
   the branch sets that range to zero for deterministic demos. The destination plate
   stays at a fixed pose so the place target is identical across episodes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
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
SHELF_SUPPORT_PATCH_SIZE = (0.8, 1.5, 0.04)
# Cuboid center: top face = SHELF_SURFACE_Z.
SHELF_SUPPORT_PATCH_CENTER = (0.62, 0.0, SHELF_SURFACE_Z - SHELF_SUPPORT_PATCH_SIZE[2] / 2.0)

# Object XY spawn pose (env-local frame, table-relative). X mirrors the locomanip env
# (the only on-table X we have ground-truth data for via the brown_box flow). Y values
# are tuned for the static apple-to-plate setup: both objects sit left on the table,
# and the plate is close enough to reduce unnecessary reach while still clearing the
# apple.
PICK_UP_OBJECT_SPAWN_XY = (0.5785, 0.27)
DESTINATION_SPAWN_XY = (0.5785, 0.06)

# Half-range of the apple's per-episode XY randomization at reset, in metres. Mirrors
# the locomanip env's ``XY_RANGE_M = 0.025`` but tightened to 0.020 because the static
# variant places the destination plate on the *same* shelf, so the spawn workspace is
# narrower (apple at Y=0.18, plate's +Y edge at ~0.09 -> 9 cm headroom; 2 cm jitter
# leaves 7 cm minimum gap to the plate). Without this jitter every recorded demo has
# the apple at the exact same XY, which limits the spatial variation a finetuned policy
# can generalize over. The destination plate is left at a fixed Pose so the place
# target is identical across episodes.
APPLE_SPAWN_XY_RANGE_M = 0.0

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

# The sim scene includes these boxes on the shelf/table workspace used by
# the static pick-and-place task, where they can block or clutter the
# apple-to-plate interaction area.
_BACKGROUND_PRIMS_TO_DEACTIVATE: tuple[str, ...] = (
    "galileo_locomanip/BackgroundAssets/boxes/jetson_orin_06",
    "galileo_locomanip/BackgroundAssets/boxes/jetson_orin_03",
    "galileo_locomanip/BackgroundAssets/boxes/hesai_box_06",
)


def _deactivate_background_prims(env, env_ids, prim_relative_paths: tuple[str, ...]) -> None:
    """Deactivate selected referenced background prims before simulation starts."""
    del env_ids
    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for prim_relative_path in prim_relative_paths:
            prim_path = f"{env_prim_path}/{prim_relative_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.OverridePrim(prim_path).SetActive(False)
            else:
                warnings.warn(
                    f"_deactivate_background_prims: prim not found at '{prim_path}'; "
                    "the background asset may still be visible.",
                    stacklevel=1,
                )


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


@dataclass
class GalileoG1StaticPickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the static-base Galileo G1 pick-and-place environment."""

    enable_cameras: bool = False
    object: str = TUNED_PICK_UP_OBJECT_NAME
    destination: str = TUNED_DESTINATION_NAME
    embodiment: str = "g1_wbc_agile_pink"
    """Use AGILE whole-body control by default; ``g1_wbc_pink`` selects HOMIE instead."""
    teleop_device: str | None = None
    task_description: str = "move the apple to the plate"
    lock_waist: bool = True
    """Keep the waist out of Pink IK unless extended arm reach is required."""


@register_environment
class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase[GalileoG1StaticPickAndPlaceEnvironmentCfg]):
    """G1 (WBC-balanced, no nav) pick-and-place on the locomanip warehouse shelf.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "galileo_g1_static_pick_and_place"
    _legacy_argparse_cfg_type = GalileoG1StaticPickAndPlaceEnvironmentCfg

    def build(self, cfg: GalileoG1StaticPickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab import sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange
        from isaaclab_arena_environments.mdp.galileo_g1_static_pick_and_place.robot_configs import (
            G1_STATIC_FINGER_DYNAMIC_FRICTION,
            G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            G1_STATIC_FINGER_PRIM_NAME_MARKERS,
            G1_STATIC_FINGER_STATIC_FRICTION,
            G1_STATIC_OPEN_ARM_JOINT_POS,
        )

        # Reuse the locomanip background USD: it bakes in lighting and provides the same
        # shelf-in-front-of-robot geometry the locomanip env was tuned against.
        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()

        class StaticShelfSupport(Object):
            def __init__(self):
                spawner_cfg = sim_utils.CuboidCfg(
                    size=SHELF_SUPPORT_PATCH_SIZE,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                    visible=False,
                )
                super().__init__(
                    name="static_pick_place_shelf_support",
                    prim_path="{ENV_REGEX_NS}/static_pick_place_shelf_support",
                    object_type=ObjectType.BASE,
                    spawner_cfg=spawner_cfg,
                    initial_pose=Pose(
                        position_xyz=SHELF_SUPPORT_PATCH_CENTER,
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                    tags=["background", "procedural"],
                )

        # The imported shelf mesh has uneven/perforated collision in the task region:
        # small objects can fall through parts of the visible shelf. Add an invisible
        # static cuboid flush with the shelf top so task objects see a clean support.
        shelf_support = StaticShelfSupport()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)(scale=_asset_scale(cfg.object))
        destination = self.asset_registry.get_asset_by_name(cfg.destination)(scale=_asset_scale(cfg.destination))
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            lock_waist=cfg.lock_waist,
        )
        embodiment.set_finger_contact_friction(
            material_path=G1_STATIC_FINGER_FRICTION_MATERIAL_PATH,
            static_friction=G1_STATIC_FINGER_STATIC_FRICTION,
            dynamic_friction=G1_STATIC_FINGER_DYNAMIC_FRICTION,
            prim_name_markers=G1_STATIC_FINGER_PRIM_NAME_MARKERS,
        )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Robot pose is tuned for the same-shelf static task: slightly forward toward
        # the table while preserving the lateral offset that keeps both arms usable.
        # The controller dynamically lifts the pelvis to ~z=0.74 at runtime;
        # init_state.pos.z=0 is correct.
        embodiment.set_initial_pose(Pose(position_xyz=(0.25, 0.08, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        embodiment.set_joint_initial_pos(G1_STATIC_OPEN_ARM_JOINT_POS)
        pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
        destination_x, destination_y = DESTINATION_SPAWN_XY
        pick_up_object_z = _shelf_spawn_z(cfg.object)
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
                position_xyz=(destination_x, destination_y, _shelf_spawn_z(cfg.destination)),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        task_description = cfg.task_description

        def env_cfg_callback(env_cfg):
            from isaaclab.managers import EventTermCfg

            # The source galileo_locomanip USD includes boxes that sit in the
            # static task workspace. Deactivate the referenced prims at startup so
            # they are absent from the composed scene for every cloned environment.
            env_cfg.events.deactivate_static_pick_place_background_prims = EventTermCfg(
                func=_deactivate_background_prims,
                mode="prestartup",
                params={"prim_relative_paths": _BACKGROUND_PRIMS_TO_DEACTIVATE},
            )
            # The GR00T policy consumes camera observations. Force one RTX sensor
            # refresh after reset so the next policy query does not see the
            # previous episode's final rendered frame.
            env_cfg.num_rerenders_on_reset = 1
            return env_cfg

        scene = Scene(assets=[background, shelf_support, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=6.0,
                task_description=task_description,
                # Mirror the locomanip env's success thresholds so metrics are comparable.
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )
