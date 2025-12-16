# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Any, Union

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from pxr import Gf, Usd, UsdGeom

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_set import RigidObjectSet
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.phyx_utils import add_contact_report

AssetCfg = Union[AssetBaseCfg, RigidObjectCfg, ArticulationCfg, ContactSensorCfg]


class Scene:

    def __init__(self, assets: list[Asset, RigidObjectSet] | None = None):
        self.assets: dict[str, Asset | RigidObjectSet] = {}
        # We add these here so a user can override them if they want.
        self.observation_cfg = None
        self.events_cfg = None
        self.termination_cfg = None
        self.rewards_cfg = None
        self.curriculum_cfg = None
        self.commands_cfg = None
        if assets is not None:
            self.add_assets(assets)

    def add_asset(self, asset: Asset | RigidObjectSet):
        """Add an asset to the scene.

        Args:
            asset: An Asset instance or a dictionary of Assets. If a dictionary is provided,
                   the keys will be used as the names of the assets and the values will be the list of assets.
        """
        if not isinstance(asset, Asset | RigidObjectSet):
            raise ValueError(f"Invalid asset type: {type(asset)}")

        if asset.name is None:
            print("Asset name is None. Skipping asset.")
            return
        # if name already exists, overwrite
        self.assets[asset.name] = asset

    def add_assets(self, assets: list[Asset | RigidObjectSet]):
        for asset in assets:
            self.add_asset(asset)

    def get_scene_cfg(self) -> Any:
        """Returns a configclass containing all the scene elements."""
        # Combine the configs into a configclass.
        fields: list[tuple[str, type, AssetCfg]] = []
        for asset in self.assets.values():
            for asset_cfg_name, asset_cfg in asset.get_object_cfg().items():
                fields.append((asset_cfg_name, type(asset_cfg), asset_cfg))
        NewConfigClass = make_configclass("SceneCfg", fields)
        new_config_class = NewConfigClass()
        return new_config_class

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_rewards_cfg(self) -> Any:
        return self.rewards_cfg

    def get_curriculum_cfg(self) -> Any:
        return self.curriculum_cfg

    def get_commands_cfg(self) -> Any:
        return self.commands_cfg

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg

    def export_to_usd(self, output_path: pathlib.Path, root_prim_path: str = "/World") -> None:
        """Exports the scene to a USD file.

        The resulting USD file will contain a root prim at the given root_prim_path,
        and each asset in the scene will be added as a child of the root prim.
        The pose of each asset prim will be set to the initial pose of the asset.
        Note that the resulting USD is flattened, so all asset references are resolved.

        Args:
            output_path: The path to the USD file to export to.
            root_prim_path: The path for a root prim added to the scene.
                All assets will be added as children of this prim.
        """
        export_scene_to_usd(self, output_path, root_prim_path)


def export_scene_to_usd(scene: Scene, output_path: pathlib.Path, root_prim_path: str = "/World") -> None:
    """Exports the scene to a USD file.

    The resulting USD file will contain a root prim at the given root_prim_path,
    and each asset in the scene will be added as a child of the root prim.
    The pose of each asset prim will be set to the initial pose of the asset.
    Note that the resulting USD is flattened, so all asset references are resolved.

    Args:
        scene: The scene to export.
        output_path: The path to the USD file to export to.
        root_prim_path: The path to the root prim in the USD file.
    """
    # Create a new stage for composition
    stage_out = Usd.Stage.CreateInMemory()
    # Add the root/default prim
    world = stage_out.DefinePrim(root_prim_path, "Xform")
    stage_out.SetDefaultPrim(world)
    # Add each asset to the stage, under the root prim
    for asset in scene.assets.values():
        _create_prim_from_asset(stage_out, asset)
    # Flatten
    flattened_layer = stage_out.Flatten()
    # Save to a file
    flattened_layer.Export(output_path.as_posix())


def _create_prim_from_asset(stage: Usd.Stage, asset: Asset) -> None:
    """Adds a prim to the stage for the given asset.

    This is used internally by the scene.export_to_usd method.
    For the passed asset, this method will create a prim at the given stage,
    and reference the asset USD file.
    The pose of the prim will be set to the initial pose of the asset.

    Args:
        stage: The stage to add the prim to.
        asset: The asset to add to the stage.
    """
    assert isinstance(asset, Object)
    # Get the default prim path
    default_prim_path = stage.GetDefaultPrim().GetPath()
    assert default_prim_path is not None
    # Construct the path for the asset prim
    asset_path = str(default_prim_path) + "/" + asset.name
    # Create the prim and reference the asset USD file.
    prim = stage.DefinePrim(asset_path, "Xform")
    prim.GetReferences().AddReference(asset.usd_path)
    # Apply a contact reporter API this is a rigid object
    if asset.object_type == ObjectType.RIGID:
        add_contact_report(prim)
    # Adding the pose
    prim_xform = UsdGeom.Xform(prim)
    # We're going to overwrite the pose, but we need to match the floating point precision
    # of the existing prim pose. So we have to do some detection.
    trans_double = _is_double_precision(prim_xform.GetTranslateOp())
    orient_double = _is_double_precision(prim_xform.GetOrientOp())
    scale_double = _is_double_precision(prim_xform.GetScaleOp())
    # Add the transform
    prim_xform.ClearXformOpOrder()
    if asset.initial_pose is not None:
        t = Gf.Vec3d(asset.initial_pose.position_xyz) if trans_double else Gf.Vec3f(asset.initial_pose.position_xyz)
        r = (
            Gf.Quatd(*asset.initial_pose.rotation_wxyz)
            if orient_double
            else Gf.Quatf(*asset.initial_pose.rotation_wxyz)
        )
        t_precision = UsdGeom.XformOp.PrecisionDouble if trans_double else UsdGeom.XformOp.PrecisionFloat
        r_precision = UsdGeom.XformOp.PrecisionDouble if orient_double else UsdGeom.XformOp.PrecisionFloat
        prim_xform.AddTranslateOp(precision=t_precision).Set(t)
        prim_xform.AddOrientOp(precision=r_precision).Set(r)
    s = Gf.Vec3d(asset.scale) if scale_double else Gf.Vec3f(asset.scale)
    s_precision = UsdGeom.XformOp.PrecisionDouble if scale_double else UsdGeom.XformOp.PrecisionFloat
    prim_xform.AddScaleOp(precision=s_precision).Set(s)


def _is_double_precision(op: UsdGeom.XformOp) -> bool | None:
    # Detect if the op is None or doesn't contain precision.
    # In this case we default to float precision.
    if not op:
        return False
    return op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
