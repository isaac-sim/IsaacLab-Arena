# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Assets used by the ported Isaac CAP gear-mesh environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import DomeLight
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset, register_asset_factory
from isaaclab_arena.utils.pose import Pose

_SHARED_ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/gear_assembly/assets"
_GEAR_MESH_ASSET_ROOT = (
    f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/latest/gear_assembly/assets"
)

FR3_WORKCELL_TABLE_USD_PATH = f"{_SHARED_ASSET_ROOT}/industrial__fr3_workcell_table/industrial__fr3_workcell_table.usda"
HDR_SHADOW_RECEIVER_USD_PATH = (
    f"{_SHARED_ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"
)
GEAR_MESH_ASSET_PATHS = {
    name: f"{_GEAR_MESH_ASSET_ROOT}/industrial__gear_mesh_{name}/industrial__gear_mesh_{name}.usda"
    for name in ("16t", "20t", "24t", "board", "mat")
}


def _normalize_initial_pose(
    initial_pose: Pose | Mapping[str, Sequence[float]] | None,
) -> Pose | None:
    """Normalize graph/YAML pose mappings to Arena poses."""
    if initial_pose is None or isinstance(initial_pose, Pose):
        return initial_pose
    return Pose(
        position_xyz=tuple(float(value) for value in initial_pose["position_xyz"]),
        rotation_xyzw=tuple(float(value) for value in initial_pose["rotation_xyzw"]),
    )


class GearInsertionRigidObject(Object):
    """Spawn a committed gear USD without enabling unused contact sensors."""

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=False),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)


class GearMeshBoardObject(Object):
    """Spawn the board articulation and retain its USD-authored joint drives."""

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        return self._add_initial_pose_to_cfg(
            ArticulationCfg(
                prim_path=self.prim_path,
                spawn=self._get_spawn_cfg(activate_contact_sensors=False),
                actuators={
                    "pinion_drive": ImplicitActuatorCfg(
                        joint_names_expr=["pinion_joint"],
                        stiffness=0.0,
                        damping=4.0,
                        effort_limit_sim=1.5,
                        armature=3.0e-4,
                    ),
                    "button_spring": ImplicitActuatorCfg(
                        joint_names_expr=["button_joint"],
                        stiffness=400.0,
                        damping=4.0,
                        armature=1.0e-3,
                    ),
                },
                **self.asset_cfg_addon,
            )
        )


def _make_gear_mesh_gear(
    teeth: int,
    instance_name: str,
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
) -> Object:
    gear = GearInsertionRigidObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.RIGID,
        usd_path=str(GEAR_MESH_ASSET_PATHS[f"{teeth}t"]),
        initial_pose=_normalize_initial_pose(initial_pose),
    )
    gear.disable_reset_pose()
    return gear


def _make_gear_mesh_board(
    teeth: int,
    instance_name: str,
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
) -> Object:
    board = GearMeshBoardObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.ARTICULATION,
        usd_path=str(GEAR_MESH_ASSET_PATHS["board"]),
        initial_pose=_normalize_initial_pose(initial_pose),
        spawn_cfg_addon={"variants": {"layout": f"board_{teeth}"}},
    )
    board.disable_reset_pose()
    return board


@register_asset_factory(name="industrial__gear_mesh_16t", object_type=ObjectType.RIGID)
def make_gear_mesh_16t(instance_name: str = "gear_a", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_gear(16, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_20t", object_type=ObjectType.RIGID)
def make_gear_mesh_20t(instance_name: str = "gear_a", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_gear(20, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_24t", object_type=ObjectType.RIGID)
def make_gear_mesh_24t(instance_name: str = "gear_a", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_gear(24, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_board_16", object_type=ObjectType.ARTICULATION)
def make_gear_mesh_board_16(instance_name: str = "board", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_board(16, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_board_20", object_type=ObjectType.ARTICULATION)
def make_gear_mesh_board_20(instance_name: str = "board", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_board(20, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_board_24", object_type=ObjectType.ARTICULATION)
def make_gear_mesh_board_24(instance_name: str = "board", initial_pose=None, **_ignored: Any) -> Object:
    return _make_gear_mesh_board(24, instance_name, initial_pose)


@register_asset_factory(name="industrial__gear_mesh_mat", object_type=ObjectType.RIGID)
def make_gear_mesh_mat(
    instance_name: str = "gear_mat",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    mat = GearInsertionRigidObject(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.RIGID,
        usd_path=str(GEAR_MESH_ASSET_PATHS["mat"]),
        initial_pose=_normalize_initial_pose(initial_pose),
    )
    mat.disable_reset_pose()
    return mat


@register_asset
class IndustrialFr3WorkcellTable(Background):
    """Cap's FR3 workcell table background."""

    name = "industrial__fr3_workcell_table"
    tags: ClassVar[list[str]] = ["background"]
    usd_path = FR3_WORKCELL_TABLE_USD_PATH
    object_min_z = 0.0

    def __init__(
        self,
        initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            usd_path=self.usd_path,
            object_min_z=self.object_min_z,
            initial_pose=_normalize_initial_pose(initial_pose),
            tags=self.tags,
            **kwargs,
        )


@register_asset
class IndustrialHdrShadowReceiver(Object):
    """Invisible collision-free floor anchor for the HDR scene."""

    name = "industrial__hdr_shadow_receiver"
    tags: ClassVar[list[str]] = ["floor", "visual"]

    def __init__(
        self,
        instance_name: str = "hdr_shadow_receiver",
        ground_z: float = 0.0,
        **kwargs: Any,
    ) -> None:
        initial_pose = kwargs.pop("initial_pose", None)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", {}) or {})
        spawn_cfg_addon["visible"] = False
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, ground_z + 0.0005))
        super().__init__(
            name=instance_name,
            usd_path=HDR_SHADOW_RECEIVER_USD_PATH,
            object_type=ObjectType.BASE,
            initial_pose=_normalize_initial_pose(initial_pose),
            spawn_cfg_addon=spawn_cfg_addon,
            tags=self.tags,
            **kwargs,
        )


@register_asset
class IndustrialEmptyWarehouseDomeLight(DomeLight):
    """Cap's shared empty-warehouse HDR dome light."""

    name = "industrial__empty_warehouse_dome_light"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            spawner_cfg=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=1500.0,
                texture_file=EmptyWarehouseHDRRobolab.texture_file,
                texture_format=EmptyWarehouseHDRRobolab.texture_format,
                visible_in_primary_ray=True,
            ),
            **kwargs,
        )
