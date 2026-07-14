# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Galileo static apple-to-plate scene helpers.

These helpers are intentionally robot-agnostic: they configure the background,
shelf support, apple, and destination plate used by the static apple task. Robot
pose, robot controller setup, teleop, policy, and task-specific controller hooks
belong in each environment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class StaticAppleSceneAssets:
    """Assets composing the Galileo static apple-to-plate scene."""

    background: Any
    shelf_support: Any
    pick_up_object: Any
    destination: Any


def asset_scale(asset_name: str, *, warning_prefix: str) -> tuple[float, float, float]:
    """Return the tuned uniform scale for ``asset_name``, or 1.0 with a warning."""
    if asset_name in _TUNED_SCALES:
        return _TUNED_SCALES[asset_name]
    warnings.warn(
        f"{warning_prefix}: no measured scale for '{asset_name}'; spawning at scale=(1.0, 1.0, 1.0). Verify visually.",
        stacklevel=2,
    )
    return (1.0, 1.0, 1.0)


def shelf_spawn_z(asset_name: str, *, warning_prefix: str) -> float:
    """Return the env-local Z to spawn ``asset_name`` flush on the shelf surface."""
    if asset_name in _USD_ORIGIN_ABOVE_BOTTOM_M:
        return SHELF_SURFACE_Z + SHELF_AIRGAP + _USD_ORIGIN_ABOVE_BOTTOM_M[asset_name]
    warnings.warn(
        f"{warning_prefix}: no measured USD-origin offset for '{asset_name}'; "
        "spawning at shelf surface with no compensation. Verify the asset's "
        "bottom face actually lands on the shelf.",
        stacklevel=2,
    )
    return SHELF_SURFACE_Z + SHELF_AIRGAP


def build_static_apple_scene_assets(
    asset_registry,
    *,
    pick_up_object_name: str,
    destination_name: str,
    warning_prefix: str,
    apple_spawn_xy_range_m: float = APPLE_SPAWN_XY_RANGE_M,
) -> StaticAppleSceneAssets:
    """Build and pose the Galileo static apple-to-plate scene assets."""

    from isaaclab import sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.utils.pose import Pose, PoseRange

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

    background = asset_registry.get_asset_by_name("galileo_locomanip")()
    shelf_support = StaticShelfSupport()
    pick_up_object = asset_registry.get_asset_by_name(pick_up_object_name)(
        scale=asset_scale(pick_up_object_name, warning_prefix=warning_prefix)
    )
    destination = asset_registry.get_asset_by_name(destination_name)(
        scale=asset_scale(destination_name, warning_prefix=warning_prefix)
    )

    pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
    destination_x, destination_y = DESTINATION_SPAWN_XY
    pick_up_object.set_initial_pose(
        PoseRange(
            position_xyz_min=(
                pick_up_object_x - apple_spawn_xy_range_m,
                pick_up_object_y - apple_spawn_xy_range_m,
                shelf_spawn_z(pick_up_object_name, warning_prefix=warning_prefix),
            ),
            position_xyz_max=(
                pick_up_object_x + apple_spawn_xy_range_m,
                pick_up_object_y + apple_spawn_xy_range_m,
                shelf_spawn_z(pick_up_object_name, warning_prefix=warning_prefix),
            ),
            rpy_min=(0.0, 0.0, 0.0),
            rpy_max=(0.0, 0.0, 0.0),
        )
    )
    destination.set_initial_pose(
        Pose(
            position_xyz=(destination_x, destination_y, shelf_spawn_z(destination_name, warning_prefix=warning_prefix)),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )
    return StaticAppleSceneAssets(
        background=background,
        shelf_support=shelf_support,
        pick_up_object=pick_up_object,
        destination=destination,
    )


def make_static_apple_task_description(pick_up_object_name: str, destination_name: str) -> str:
    object_label = pick_up_object_name.replace("_", " ")
    destination_label = destination_name.replace("_", " ")
    return (
        f"Pick up the {object_label} from the shelf and place it onto the "
        f"{destination_label} on the same shelf next to it."
    )


def deactivate_background_prims(env, env_ids, prim_relative_paths: tuple[str, ...]) -> None:
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
                    f"deactivate_background_prims: prim not found at '{prim_path}'; "
                    "the background asset may still be visible.",
                    stacklevel=1,
                )


def make_static_apple_env_cfg_callback(*, num_rerenders_on_reset: int | None = None):
    """Return an env-cfg callback for static apple scene cleanup."""

    def env_cfg_callback(env_cfg):
        from isaaclab.managers import EventTermCfg

        env_cfg.events.deactivate_static_pick_place_background_prims = EventTermCfg(
            func=deactivate_background_prims,
            mode="prestartup",
            params={"prim_relative_paths": _BACKGROUND_PRIMS_TO_DEACTIVATE},
        )
        if num_rerenders_on_reset is not None:
            env_cfg.num_rerenders_on_reset = num_rerenders_on_reset
        return env_cfg

    return env_cfg_callback
