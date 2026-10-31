# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic SimReady USD object registered for agent-generated environments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.assets.simready_constants import (
    SIMREADY_PHYSICS_VARIANTS,
    SIMREADY_SEARCH_REGISTRY_PREFIX,
    SIMREADY_USD_ARTICULATION_REGISTRY_NAME,
    SIMREADY_USD_BACKGROUND_REGISTRY_NAME,
    SIMREADY_USD_OBJECT_REGISTRY_NAME,
    SIMREADY_USD_OPENABLE_ARTICULATION_REGISTRY_NAME,
    SIMREADY_USD_PRESSABLE_ARTICULATION_REGISTRY_NAME,
)
from isaaclab_arena.utils.pose import Pose

_NON_IDENTIFIER_CHARACTERS = re.compile(r"[^a-z0-9]+")


def _typed_spawn_cfg_addon(spawn_cfg_addon: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML mappings for common physics overrides to Isaac Lab configs."""
    from dataclasses import fields

    from isaaclab.sim import CollisionPropertiesCfg, MeshCollisionPropertiesCfg, RigidBodyPropertiesCfg

    spawn_cfg_addon = dict(spawn_cfg_addon)
    rigid_props = spawn_cfg_addon.get("rigid_props")
    rigid_fields = {field.name for field in fields(RigidBodyPropertiesCfg)}
    if isinstance(rigid_props, dict) and set(rigid_props) <= rigid_fields:
        spawn_cfg_addon["rigid_props"] = RigidBodyPropertiesCfg(**rigid_props)
    collision_props = spawn_cfg_addon.get("collision_props")
    collision_fields = {field.name for field in fields(CollisionPropertiesCfg)}
    if isinstance(collision_props, dict) and set(collision_props) <= collision_fields:
        collision_props = dict(collision_props)
        mesh_collision_property = collision_props.get("mesh_collision_property")
        if isinstance(mesh_collision_property, dict):
            collision_props["mesh_collision_property"] = MeshCollisionPropertiesCfg(**mesh_collision_property)
        spawn_cfg_addon["collision_props"] = CollisionPropertiesCfg(**collision_props)
    return spawn_cfg_addon


def _background_wrapper(
    usd_path: str,
    variants: dict[str, str],
    static_collision_only: bool,
    reference_colliders: dict[str, dict[str, Any]],
) -> str:
    """Write a SimReady background wrapper with optional static colliders."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    source_stage = Usd.Stage.Open(usd_path)
    assert source_stage and source_stage.GetDefaultPrim(), "SimReady background requires a default prim."
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(usd_path)
    stage.SetDefaultPrim(root)
    apply_usd_variant_selections(stage, variants)

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
        if prim.IsInstance():
            prim.SetInstanceable(False)
    if static_collision_only:
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

    material = None
    for relative_path, collider_cfg in reference_colliders.items():
        path = PurePosixPath(relative_path)
        assert (
            relative_path and not path.is_absolute() and ".." not in path.parts
        ), f"Reference collider path must be asset-relative, got {relative_path!r}."
        assert set(collider_cfg) == {
            "position_xyz",
            "size_xyz",
        }, f"Reference collider {relative_path!r} requires position_xyz and size_xyz."
        position_xyz = tuple(collider_cfg["position_xyz"])
        size_xyz = tuple(collider_cfg["size_xyz"])
        assert len(position_xyz) == 3, f"Reference collider {relative_path!r} position_xyz must have three values."
        assert len(size_xyz) == 3 and all(
            size > 0.0 for size in size_xyz
        ), f"Reference collider {relative_path!r} size_xyz must contain three positive values."
        prim_path = f"/Asset/{path.as_posix()}"
        assert not stage.GetPrimAtPath(prim_path), f"Reference collider prim already exists at {prim_path!r}."
        body = UsdGeom.Xform.Define(stage, prim_path)
        body.AddTranslateOp().Set(Gf.Vec3d(*position_xyz))
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Surface")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if material is None:
            material = UsdShade.Material.Define(stage, "/Asset/InvisibleReferenceCollider")
            shader = UsdShade.Shader.Define(stage, "/Asset/InvisibleReferenceCollider/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    cache = Path.home() / ".cache" / "isaaclab_arena" / "simready-backgrounds"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        (
            f"{usd_path}\0{static_collision_only}\0"
            f"{json.dumps(reference_colliders, sort_keys=True)}\0{sorted(variants.items())}"
        ).encode()
    ).hexdigest()
    wrapper = cache / f"{key}.usda"
    stage.GetRootLayer().Export(str(wrapper))
    return str(wrapper)


@register_asset
class SimReadyUsdBackground(Background):
    """Spawn a SimReady background with optional static reference colliders."""

    registry_name = SIMREADY_USD_BACKGROUND_REGISTRY_NAME
    tags = ["background", "sim-ready"]

    def __init__(
        self,
        usd_path: str,
        object_min_z: float,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        static_collision_only: bool = False,
        reference_colliders: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """Create a SimReady background from an explicit USD path."""
        assert usd_path, "simready_usd_background requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            _download_central_asset_package_to_cache,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = _download_central_asset_package_to_cache(usd_path)
        spawn_cfg_addon = _typed_spawn_cfg_addon(kwargs.pop("spawn_cfg_addon", None) or {})
        default_variants = {**SIMREADY_PHYSICS_VARIANTS, "PhysX": "Enabled"}
        spawn_cfg_addon.setdefault("variants", default_variants)
        reference_colliders = reference_colliders or {}
        if static_collision_only or reference_colliders:
            usd_path = _background_wrapper(
                usd_path,
                spawn_cfg_addon["variants"],
                static_collision_only,
                reference_colliders,
            )
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            usd_path=usd_path,
            object_min_z=object_min_z,
            initial_pose=initial_pose,
            reset_nested_physics=not static_collision_only,
            tags=self.tags,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


@register_asset
class SimReadyUsdObject(Object):
    """Spawn a SimReady asset from an explicit USD path supplied in graph params."""

    registry_name = SIMREADY_USD_OBJECT_REGISTRY_NAME

    tags = ["sim-ready"]

    object_type = ObjectType.RIGID
    """The search only accepts an asset with exactly one rigid body."""

    def __init__(
        self,
        usd_path: str,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ):
        """Create an object from a SimReady USD file.

        Args:
            usd_path: USD path or URL of the SimReady asset.
            name: Name of the object in the scene. Defaults to the registry name.
            prim_path: Prim path of the object. Defaults to ``{ENV_REGEX_NS}/<name>``.
            initial_pose: Initial pose of the object.
            scale: Scale of the object. Defaults to ``(1.0, 1.0, 1.0)``.
            **kwargs: Extra arguments passed to ``Object``.
        """
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        assert usd_path, "simready_usd_object requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            download_and_wrap_central_asset,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = download_and_wrap_central_asset(usd_path)
        spawn_cfg_addon = _typed_spawn_cfg_addon(kwargs.pop("spawn_cfg_addon", None) or {})
        spawn_cfg_addon.setdefault("variants", dict(SIMREADY_PHYSICS_VARIANTS))
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )


def _articulation_wrapper(
    usd_path: str,
    variants: dict[str, str],
    fixed_base_body_path: str | None,
    articulation_root_prim_path: str | None,
) -> str:
    """Write a wrapper that optionally fixes and re-roots an articulation."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    def wrapper_prim(path: str, parameter_name: str):
        relative_path = PurePosixPath(path)
        assert (
            path and not relative_path.is_absolute() and ".." not in relative_path.parts
        ), f"{parameter_name} must be an asset-relative prim path, got {path!r}."
        suffix = "" if relative_path.as_posix() == "." else f"/{relative_path.as_posix()}"
        prim = stage.GetPrimAtPath(f"/Asset{suffix}")
        assert prim, f"{parameter_name} target '/Asset{suffix}' does not exist."
        return prim

    source_stage = Usd.Stage.Open(usd_path)
    assert source_stage and source_stage.GetDefaultPrim(), "Articulated SimReady asset requires a default prim."
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    root = stage.DefinePrim("/Asset")
    root.GetReferences().AddReference(usd_path)
    stage.SetDefaultPrim(root)
    apply_usd_variant_selections(stage, variants)
    if articulation_root_prim_path is not None:
        articulation_root = wrapper_prim(articulation_root_prim_path, "articulation_root_prim_path")
        for prim in Usd.PrimRange(root):
            if prim != articulation_root and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if not articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(articulation_root)
    if fixed_base_body_path is not None:
        body = wrapper_prim(fixed_base_body_path, "fixed_base_body_path")
        assert body.HasAPI(UsdPhysics.RigidBodyAPI), f"Fixed-base target '{body.GetPath()}' must be a rigid body."
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/ArenaFixedBase")
        fixed_joint.CreateBody1Rel().SetTargets([body.GetPath()])
    cache = Path.home() / ".cache" / "isaaclab_arena" / "simready-articulations"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{usd_path}\0{fixed_base_body_path}\0{articulation_root_prim_path}\0{sorted(variants.items())}".encode()
    ).hexdigest()
    wrapper = cache / f"{key}.usda"
    stage.GetRootLayer().Export(str(wrapper))
    return str(wrapper)


@register_asset
class SimReadyUsdArticulation(Object):
    """Spawn an articulated SimReady USD with optional fixed-base and initial-joint configuration."""

    registry_name = SIMREADY_USD_ARTICULATION_REGISTRY_NAME
    tags = ["object", "sim-ready"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        usd_path: str,
        name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        fixed_base_body_path: str | None = None,
        articulation_root_prim_path: str | None = None,
        initial_joint_positions: dict[str, float] | None = None,
        **kwargs: Any,
    ):
        """Create an articulation from a SimReady USD path."""
        assert "instance_name" not in kwargs, "instance_name was renamed to name; pass name= instead."
        assert usd_path, "simready_usd_articulation requires params.usd_path"
        from isaaclab_arena.assets.simready_central_asset_loader import (
            _download_central_asset_package_to_cache,
            is_central_asset_url,
        )

        if is_central_asset_url(usd_path):
            usd_path = _download_central_asset_package_to_cache(usd_path)
        spawn_cfg_addon = _typed_spawn_cfg_addon(kwargs.pop("spawn_cfg_addon", None) or {})
        default_variants = {**SIMREADY_PHYSICS_VARIANTS, "PhysX": "Enabled"}
        spawn_cfg_addon.setdefault("variants", default_variants)
        if fixed_base_body_path is not None or articulation_root_prim_path is not None:
            usd_path = _articulation_wrapper(
                usd_path,
                spawn_cfg_addon["variants"],
                fixed_base_body_path,
                articulation_root_prim_path,
            )
        super().__init__(
            name=name or self.registry_name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=usd_path,
            object_type=self.object_type,
            scale=scale if scale is not None else (1.0, 1.0, 1.0),
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)


@register_asset
class SimReadyUsdOpenableArticulation(SimReadyUsdArticulation, Openable):
    """Spawn an articulated SimReady USD that exposes one openable joint."""

    registry_name = SIMREADY_USD_OPENABLE_ARTICULATION_REGISTRY_NAME

    def __init__(self, openable_joint_name: str, **kwargs: Any):
        """Create an openable articulation from a SimReady USD path."""
        super().__init__(openable_joint_name=openable_joint_name, **kwargs)


@register_asset
class SimReadyUsdPressableArticulation(SimReadyUsdArticulation, Pressable):
    """Spawn an articulated SimReady USD that exposes one pressable joint."""

    registry_name = SIMREADY_USD_PRESSABLE_ARTICULATION_REGISTRY_NAME

    def __init__(self, pressable_joint_name: str, **kwargs: Any):
        """Create a pressable articulation from a SimReady USD path."""
        super().__init__(pressable_joint_name=pressable_joint_name, **kwargs)


def simready_search_registry_name(search_phrase: str, usd_path: str) -> str:
    """Return a stable catalogue name for a phrase and asset path."""
    slug = _NON_IDENTIFIER_CHARACTERS.sub("_", search_phrase.strip().lower()).strip("_")
    assert slug, f"search phrase {search_phrase!r} has no characters usable in a registry name"
    digest = hashlib.sha256(usd_path.encode()).hexdigest()[:16]
    return f"{SIMREADY_SEARCH_REGISTRY_PREFIX}{slug}_{digest}"


def register_searched_simready_object(
    search_phrase: str,
    usd_path: str,
    tags: Sequence[str] = (),
) -> type[SimReadyUsdObject]:
    """Register a searched SimReady asset so it can be picked like any other catalogue object.

    Args:
        search_phrase: Phrase the asset was found for; used with its path to name the catalogue entry.
        usd_path: USD path or URL of the found asset.
        tags: Extra catalogue tags to expose alongside the SimReady ones.

    Returns:
        The registered asset class.
    """
    searched_registry_name = simready_search_registry_name(search_phrase, usd_path)
    registry = AssetRegistry()
    if registry.is_registered(searched_registry_name, ensure_loaded=False):
        return registry.get_asset_by_name(searched_registry_name)
    found_usd_path = usd_path
    # Tagged 'object' unlike the generic asset, because this one carries its own usd_path.
    merged_tags = list(dict.fromkeys(["object", *SimReadyUsdObject.tags, *tags]))

    class SearchedSimReadyObject(SimReadyUsdObject):
        """A SimReady asset the search found for one object in a prompt."""

        registry_name = searched_registry_name
        tags = merged_tags

        def __init__(self, **kwargs: Any):
            kwargs.setdefault("usd_path", found_usd_path)
            super().__init__(**kwargs)

    registry.register(SearchedSimReadyObject, searched_registry_name)
    return SearchedSimReadyObject
