# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from functools import partial

from isaaclab.sim import UsdFileCfg, UsdFileWithCompliantContactCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.spawners.utils import fragment_mapping
from isaaclab.sim.utils import make_uninstanceable, standardize_xform_ops
from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.physics_config import UsdFileCfgPrimPhysicsWrapper
from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry
from isaaclab_arena.utils.usd.rigid_bodies import get_all_rigid_body_prim_paths_from_stage

_COPY_ROOT = Sdf.Path("/AlignedAsset")
"""Default prim of a written copy."""

_COPY_REST_OF_ASSET = _COPY_ROOT.AppendChild("source")
"""Everything from the source asset except its rigid body."""

_COPY_BODY = _COPY_ROOT.AppendChild("rigid_body")
"""The moved rigid body, at /rigid_body relative to the spawn root."""


def align_rigid_body_paths(spawn_cfgs: Sequence[SpawnerCfg]) -> list[SpawnerCfg]:
    """Return spawn configurations whose rigid bodies sit at the same path inside their assets.

    Isaac Lab finds a rigid object's body at one path in all environments, so assets spawned as one object must
    agree on it. If they already do, unchanged copies of the configurations are returned. Otherwise each USD asset
    gets a cached copy with its rigid body moved to ``/rigid_body``, for example::

        cracker box                         copy
        /_03_cracker_box  (rigid body)      /AlignedAsset
            /Looks                              /rigid_body  (rigid body)
            /_03_cracker_box                        /Looks
                                                    /_03_cracker_box

        sweet potato                        copy
        /root                               /AlignedAsset
            /Looks                              /source
            /Physics                                /Looks
            /SweetPotato005  (rigid body)           /Physics
                /Collisions                     /rigid_body  (rigid body)
                /Visuals                            /Collisions
                /Sites                              /Visuals
                                                    /Sites

    Args:
        spawn_cfgs: Spawn configurations of the assets spawned as one object.

    Returns:
        Spawn configurations in the same order. When the paths differ, each points at its copy and has variants set
        to None.
    """
    assert spawn_cfgs, "At least one spawn configuration is required."
    geometries = [SpawnCfgGeometry(spawn_cfg) for spawn_cfg in spawn_cfgs]
    body_paths = [geometry.get_contact_body_path() for geometry in geometries]
    if len(set(body_paths)) == 1:
        return [deepcopy(spawn_cfg) for spawn_cfg in spawn_cfgs]
    for spawn_cfg in spawn_cfgs:
        assert isinstance(spawn_cfg, UsdFileCfg), (
            "Spawn configurations with different rigid-body paths must be USD assets, so their bodies can be moved "
            f"to one path; got {type(spawn_cfg).__name__} and rigid-body paths {body_paths}."
        )
        _assert_settings_do_not_name_asset_prims(spawn_cfg)
    # Build all copies before saving any, so a rejected asset leaves no cache files behind.
    copied_layers = [_build_copy_with_body_moved(geometry) for geometry in geometries]
    aligned_spawn_cfgs = []
    for spawn_cfg, copied_layer in zip(spawn_cfgs, copied_layers):
        aligned_spawn_cfg = deepcopy(spawn_cfg)
        aligned_spawn_cfg.usd_path = str(_save_to_cache(copied_layer))
        # The copy has the variant selections baked in.
        aligned_spawn_cfg.variants = None
        aligned_spawn_cfgs.append(aligned_spawn_cfg)
    return aligned_spawn_cfgs


def _assert_settings_do_not_name_asset_prims(spawn_cfg: UsdFileCfg) -> None:
    """Assert that no spawn setting names a prim inside the USD file, since the copy moves those prims."""
    cannot_move = f"Cannot move the rigid body of {spawn_cfg.usd_path} to a common path"
    if isinstance(spawn_cfg, UsdFileCfgPrimPhysicsWrapper):
        assert not spawn_cfg.prim_physics, f"{cannot_move}: prim_physics names source prims."
    # Fields that also accept schema fragments keyed by prim path pattern; deformable_props takes only a config.
    property_settings = {
        "mass_props": spawn_cfg.mass_props,
        "volume_deformable_props": spawn_cfg.volume_deformable_props,
        "surface_deformable_props": spawn_cfg.surface_deformable_props,
        "rigid_props": spawn_cfg.rigid_props,
        "collision_props": spawn_cfg.collision_props,
        "articulation_props": spawn_cfg.articulation_props,
        "fixed_tendons_props": spawn_cfg.fixed_tendons_props,
        "spatial_tendons_props": spawn_cfg.spatial_tendons_props,
        "joint_drive_props": spawn_cfg.joint_drive_props,
    }
    # Isaac Lab reads these fields with its private fragment_mapping: a mapping for fragments, None for a config.
    # TODO(cvolk, 2026.10.13): Allow bare fragments. Isaac Lab applies them to the whole asset, so they still apply
    # after the move.
    for field_name, property_setting in property_settings.items():
        assert not fragment_mapping(property_setting), (
            f"{cannot_move}: {field_name} uses schema fragments or target patterns. "
            "Use a properties config such as MassPropertiesCfg, or spawn configurations with the same rigid-body path."
        )
    assert not spawn_cfg.visual_material_bindings, f"{cannot_move}: visual_material_bindings name source prims."
    # Isaac Lab reads visual_material_path only when visual_material is set.
    assert spawn_cfg.visual_material is None or spawn_cfg.visual_material_path.startswith(
        "/"
    ), f"{cannot_move}: visual_material_path is relative to the asset root."
    if isinstance(spawn_cfg, UsdFileWithCompliantContactCfg):
        assert (
            spawn_cfg.physics_material_prim_path is None
        ), f"{cannot_move}: physics_material_prim_path names source prims."


def _build_copy_with_body_moved(geometry: SpawnCfgGeometry) -> Sdf.Layer:
    """Return a flattened copy of one USD asset with its rigid body moved to /rigid_body.

    Args:
        geometry: Geometry of a UsdFileCfg spawn configuration.

    Returns:
        The copy as a layer, not yet saved to the cache.
    """
    stage = geometry.open_usd_stage()
    source_root = stage.GetDefaultPrim().GetPath()
    _make_asset_editable(stage, source_root, geometry.spawn_cfg)
    source_body_path = Sdf.Path(str(source_root) + geometry.get_contact_body_path())
    body_path_under_source = source_body_path.ReplacePrefix(source_root, _COPY_REST_OF_ASSET)
    _assert_single_rigid_body(stage, source_body_path, geometry.spawn_cfg.usd_path)

    copy_stage = _move_body_to_common_path(stage, source_root, body_path_under_source)
    _keep_inherited_placement_and_materials(
        stage.GetPrimAtPath(source_body_path), copy_stage, source_root, body_path_under_source
    )
    return copy_stage.GetRootLayer()


def _make_asset_editable(stage: Usd.Stage, source_root: Sdf.Path, spawn_cfg: UsdFileCfg) -> None:
    """Make the asset at ``source_root`` editable in its in-memory stage and clear the spawn root's placement."""
    # Traverse() skips instance proxies, and Flatten would move instanced prims out of the asset into prototypes.
    make_uninstanceable(source_root, stage=stage)
    # Isaac Lab sets the spawn root's pose, and a tuple spawn scale replaces its scale, so neither may end up in the
    # moved body's transform.
    standardize_xform_ops(
        stage.GetPrimAtPath(source_root),
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0) if spawn_cfg.scale is not None else None,
    )


def _assert_single_rigid_body(stage: Usd.Stage, body_path: Sdf.Path, usd_path: str) -> None:
    """Assert that the stage's only rigid body is at ``body_path``."""
    rigid_body_paths = get_all_rigid_body_prim_paths_from_stage(stage)
    assert rigid_body_paths == [
        str(body_path)
    ], f"Moving the rigid body of {usd_path} requires exactly one rigid body; found {rigid_body_paths}."


def _move_body_to_common_path(stage: Usd.Stage, source_root: Sdf.Path, body_path_under_source: Sdf.Path) -> Usd.Stage:
    """Flatten the asset into a new stage with its rigid body moved to /AlignedAsset/rigid_body.

    The rest of the asset goes to /AlignedAsset/source, and paths that pointed into the body follow it.

    Args:
        stage: Source stage with the asset at ``source_root``.
        source_root: Path of the source asset in ``stage``.
        body_path_under_source: Path of the rigid body once the asset is copied to /AlignedAsset/source.

    Returns:
        A stage on the new layer.
    """
    # Flatten bakes in references and variant selections and makes asset paths absolute, so the copy can live anywhere.
    layer = stage.Flatten(addSourceFileComment=False)
    copy_stage = Usd.Stage.Open(layer)
    copy_root = UsdGeom.Xform.Define(copy_stage, _COPY_ROOT).GetPrim()
    # Keep the rest of the asset: the body may use materials and other prims outside it.
    copied_rest_of_asset = Sdf.CopySpec(layer, source_root, layer, _COPY_REST_OF_ASSET)
    assert copied_rest_of_asset, f"Could not copy {source_root} to {_COPY_REST_OF_ASSET}."
    copied_body = Sdf.CopySpec(layer, body_path_under_source, layer, _COPY_BODY)
    assert copied_body, f"Could not copy {body_path_under_source} to {_COPY_BODY}."
    copy_stage.RemovePrim(body_path_under_source)
    copy_stage.RemovePrim(source_root)
    copy_stage.SetDefaultPrim(copy_root)
    _point_links_at_moved_body(layer, body_path_under_source)
    return copy_stage


def _point_links_at_moved_body(layer: Sdf.Layer, body_path_under_source: Sdf.Path) -> None:
    """Point relationship targets and attribute connections into body_path_under_source at the moved body."""
    # Sdf.CopySpec remaps paths only inside the copied subtree, so joints, filtered pairs or collections outside
    # the body still name its old path.
    path_after_body_move = partial(_path_after_body_move, body_path_under_source=body_path_under_source)
    property_paths = []

    def collect_property_path(spec_path: Sdf.Path) -> None:
        if spec_path.IsPropertyPath():
            property_paths.append(spec_path)

    # Collect the paths first, so the list edits below never run during the traversal.
    layer.Traverse(Sdf.Path.absoluteRootPath, collect_property_path)
    for property_path in property_paths:
        property_spec = layer.GetPropertyAtPath(property_path)
        if isinstance(property_spec, Sdf.RelationshipSpec):
            property_spec.targetPathList.ModifyItemEdits(path_after_body_move)
        elif isinstance(property_spec, Sdf.AttributeSpec):
            property_spec.connectionPathList.ModifyItemEdits(path_after_body_move)


def _path_after_body_move(path: Sdf.Path, body_path_under_source: Sdf.Path) -> Sdf.Path:
    """Map a path at or below ``body_path_under_source`` to the moved body; return other paths unchanged."""
    if path.HasPrefix(body_path_under_source):
        return path.ReplacePrefix(body_path_under_source, _COPY_BODY)
    return path


def _keep_inherited_placement_and_materials(
    source_body: Usd.Prim, copy_stage: Usd.Stage, source_root: Sdf.Path, body_path_under_source: Sdf.Path
) -> None:
    """Give the moved body the transform and material bindings it had in the source asset, including inherited ones.

    Args:
        source_body: The rigid body in the source stage.
        copy_stage: Stage with the moved body at /AlignedAsset/rigid_body.
        source_root: Path of the source asset in the source stage.
        body_path_under_source: Path of the rigid body once the asset is copied to /AlignedAsset/source.
    """
    # The body's parents are not in the copy, so read what it inherited from them in the source stage.
    moved_body = copy_stage.GetPrimAtPath(_COPY_BODY)
    # The source root has no parent, so the body's world transform is its transform within the asset.
    body_transform = UsdGeom.Xformable(source_body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    UsdGeom.Xformable(moved_body).MakeMatrixXform().Set(body_transform)
    for purpose in (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview, UsdShade.Tokens.full):
        source_material, binding_relationship = UsdShade.MaterialBindingAPI(source_body).ComputeBoundMaterial(purpose)
        if not source_material:
            continue
        binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(binding_relationship)
        material_path_in_copy = _path_after_body_move(
            source_material.GetPath().ReplacePrefix(source_root, _COPY_REST_OF_ASSET), body_path_under_source
        )
        material = UsdShade.Material(copy_stage.GetPrimAtPath(material_path_in_copy))
        UsdShade.MaterialBindingAPI.Apply(moved_body).Bind(
            material, bindingStrength=binding_strength, materialPurpose=purpose
        )


def _save_to_cache(layer: Sdf.Layer) -> pathlib.Path:
    """Save the layer to a cache file named by its content hash, reusing an existing file, and return the path."""
    content_hash = hashlib.sha256(layer.ExportToString().encode()).hexdigest()
    cache_path = get_arena_asset_cache_dir() / f"rigid_{content_hash}.usd"
    if cache_path.exists():
        return cache_path
    # Export to a temporary file and rename it, so cache_path never names a half-written file.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".usd", delete=False) as temporary_file:
            temporary_path = temporary_file.name
        layer.Export(temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return cache_path
