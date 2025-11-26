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
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.configclass import make_configclass

AssetCfg = Union[AssetBaseCfg, RigidObjectCfg, ArticulationCfg, ContactSensorCfg]


class Scene:

    def __init__(self, assets: list[Asset] | None = None):
        self.assets: dict[str, Asset] = {}
        # We add these here so a user can override them if they want.
        self.observation_cfg = None
        self.events_cfg = None
        self.termination_cfg = None
        self.rewards_cfg = None
        self.curriculum_cfg = None
        self.commands_cfg = None
        if assets is not None:
            self.add_assets(assets)

    def add_asset(self, asset: Asset):
        assert asset.name is not None, "Asset with the same name already exists"
        self.assets[asset.name] = asset

    def add_assets(self, assets: list[Asset]):
        for asset in assets:
            self.add_asset(asset)

    def get_scene_cfg(self) -> Any:
        """Returns a configclass containing all the scene elements."""
        # Combine the configs into a configclass.
        fields: list[tuple[str, type, AssetCfg]] = []
        for asset in self.assets.values():
            for asset_cfg_name, asset_cfg in asset.get_cfgs().items():
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
    # Add the transform
    prim_xform = UsdGeom.Xform(prim)
    prim_xform.ClearXformOpOrder()
    if asset.initial_pose is not None:
        prim_xform.AddTranslateOp().Set(Gf.Vec3f(asset.initial_pose.position_xyz))
        prim_xform.AddOrientOp().Set(Gf.Quatf(*asset.initial_pose.rotation_wxyz))
    prim_xform.AddScaleOp().Set(Gf.Vec3f(asset.scale))
