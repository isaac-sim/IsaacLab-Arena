# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turnable appliances loaded from pinned asset packages."""

import hashlib
import math
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.simready_central_asset_loader import (
    _download_central_asset_package_to_cache,
    is_central_asset_url,
)
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections


@register_asset
class TurnableAppliance(Object, Turnable):
    """A countertop articulation with a named rotary control and discrete settings."""

    registry_name = "turnable_appliance"
    tags = ["object", "turnable", "appliance"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        turnable_joint_name: str,
        min_level_angle_deg: float,
        max_level_angle_deg: float,
        num_levels: int,
        usd_path: str | None = None,
        lightwheel_file_version: str | None = None,
        knob_damping: float | None = None,
        initial_joint_positions: dict[str, float] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """Load one USD source and expose its rotary joint through Turnable.

        Args:
            turnable_joint_name: Revolute joint controlling the appliance setting.
            min_level_angle_deg: First active setting, above the reset angle of zero.
            max_level_angle_deg: Last active setting.
            num_levels: Number of discrete active settings.
            usd_path: Local USD path or revision-pinned SimReady Central URL.
            lightwheel_file_version: Pinned Lightwheel USD file version, instead of usd_path.
            knob_damping: Passive rotary damping in N m s/rad, with zero stiffness and velocity.
            initial_joint_positions: Initial joint positions in radians or meters; unspecified joints start at zero.
            name: Scene object name.
            **kwargs: Additional Object configuration.
        """
        assert bool(usd_path) != bool(lightwheel_file_version), "Specify exactly one appliance source"
        if lightwheel_file_version:
            from lightwheel_sdk.loader import object_loader

            usd_path, _, _ = acquire_lightwheel_asset(
                object_loader,
                object_loader.acquire_by_file_version,
                description=f"Lightwheel appliance {lightwheel_file_version}",
                file_version_id=lightwheel_file_version,
            )
        elif is_central_asset_url(usd_path):
            usd_path = _download_central_asset_package_to_cache(usd_path)
        usd_path = _write_appliance_wrapper(usd_path, turnable_joint_name, knob_damping)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        # A fixed appliance base preserves countertop placement while its controls move.
        spawn_cfg_addon.setdefault("fix_root_link", True)
        super().__init__(
            name=name or self.registry_name,
            usd_path=usd_path,
            object_type=ObjectType.ARTICULATION,
            turnable_joint_name=turnable_joint_name,
            min_level_angle_deg=min_level_angle_deg,
            max_level_angle_deg=max_level_angle_deg,
            num_levels=num_levels,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)


def _write_appliance_wrapper(
    usd_path: str,
    joint_name: str,
    knob_damping: float | None = None,
    joint_type=UsdPhysics.RevoluteJoint,
) -> str:
    """Write a physics-enabled reference layer without changing the source package.

    Args:
        usd_path: Downloaded USD entrypoint.
        joint_name: Required enabled control joint.
        knob_damping: Optional passive rotary damping in N m s/rad.
        joint_type: USD joint schema required for the control.

    Returns:
        Cached USD wrapper path.
    """
    asset_stage = Usd.Stage.Open(usd_path)
    assert asset_stage and asset_stage.GetDefaultPrim(), "Appliance requires a default prim"
    assert UsdGeom.GetStageUpAxis(asset_stage) == UsdGeom.Tokens.z, "Appliance must use Z-up coordinates"
    assert UsdGeom.GetStageMetersPerUnit(asset_stage) == 1.0, "Appliance must use meters"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(usd_path)
    stage.SetDefaultPrim(root)
    apply_usd_variant_selections(stage, {"Physics": "physics", "PhysX": "Enabled"})
    assert not stage.GetCompositionErrors(), "Appliance has unresolved USD references"
    joints = [p for p in stage.Traverse() if p.GetName() == joint_name and p.IsA(joint_type)]
    assert len(joints) == 1, f"Expected one {joint_type.__name__} named {joint_name}"
    assert joint_type(joints[0]).GetJointEnabledAttr().Get(), "Appliance control joint is disabled"
    if knob_damping is not None:
        assert joint_type == UsdPhysics.RevoluteJoint, "Rotary damping requires a revolute joint"
        assert knob_damping >= 0, "Knob damping must be non-negative"
        drive = UsdPhysics.DriveAPI.Apply(joints[0], "angular")
        drive.CreateTypeAttr("force")
        # USD angular drives use degrees; the runtime coefficient is expressed per radian.
        drive.CreateDampingAttr(knob_damping * math.pi / 180.0)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots) <= 1, "Appliance must contain one articulation"
    if not roots:
        # Some SimReady packages author connected rigid bodies and joints but no articulation root.
        body_paths = joint_type(joints[0]).GetBody0Rel().GetTargets()
        assert len(body_paths) == 1, "Appliance control requires a connected base body"
        base = stage.GetPrimAtPath(body_paths[0])
        # Joint targets may name a mesh below the rigid body that owns it.
        while base and not base.HasAPI(UsdPhysics.RigidBodyAPI):
            base = base.GetParent()
        assert base, "Appliance control base must belong to a rigid body"
        UsdPhysics.ArticulationRootAPI.Apply(base)
    cache = Path.home() / ".cache" / "isaaclab_arena" / "turnable-appliances"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{usd_path}\n{joint_name}\n{knob_damping}\n{joint_type.__name__}".encode()).hexdigest()
    wrapper = cache / f"{key}.usda"
    stage.GetRootLayer().Export(str(wrapper))
    return str(wrapper)
