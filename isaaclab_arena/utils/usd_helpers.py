# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import colorsys
import random
from contextlib import contextmanager

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


def get_all_prims(
    stage: Usd.Stage, prim: Usd.Prim | None = None, prims_list: list[Usd.Prim] | None = None
) -> list[Usd.Prim]:
    """Get all prims in the stage.

    Performs a Depth First Search (DFS) through the prims in a stage
    and returns all the prims.

    Args:
        stage: The stage to get the prims from.
        prim: The prim to start the search from. Defaults to the pseudo-root.
        prims_list: The list to store the prims in. Defaults to an empty list.

    Returns:
        A list of prims in the stage.
    """
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetAllChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def has_light(stage: Usd.Stage) -> bool:
    """Check if the stage has a light"""
    LIGHT_TYPES = (
        UsdLux.SphereLight,
        UsdLux.RectLight,
        UsdLux.DomeLight,
        UsdLux.DistantLight,
        UsdLux.DiskLight,
    )
    has_light = False
    all_prims = get_all_prims(stage)
    for prim in all_prims:
        if any(prim.IsA(t) for t in LIGHT_TYPES):
            has_light = True
            break
    return has_light


def is_articulation_root(prim: Usd.Prim) -> bool:
    """Check if prim is articulation root"""
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigid_body(prim: Usd.Prim) -> bool:
    """Check if prim is rigidbody"""
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_prim_depth(prim: Usd.Prim) -> int:
    """Get the depth of a prim"""
    return len(str(prim.GetPath()).split("/")) - 2


@contextmanager
def open_stage(path):
    """Open a stage and ensure it is closed after use."""
    stage = Usd.Stage.Open(path)
    try:
        yield stage
    finally:
        # Drop the local reference; Garbage Collection will reclaim once no prim/attr handles remain
        del stage


def get_asset_usd_path_from_prim_path(prim_path: str, stage: Usd.Stage) -> str | None:
    """Get the USD path from a prim path, that is referring to an asset."""
    # Note (xinjieyao, 2025.12.12): preferred way to get the composed asset path is to ask the Usd.Prim object itself,
    # which handles the entire composition stack. Here it achieved this goal thru root layer due to the USD API limitations.
    # It only finds references authored on the root layer.
    # If the asset was referenced in an intermediate sublayer, this method would fail to find the asset path.
    root_layer = stage.GetRootLayer()
    prim_spec = root_layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return None

    try:
        reference_list = prim_spec.referenceList.GetAddedOrExplicitItems()
    except Exception as e:
        print(f"Failed to get reference list for prim {prim_path}: {e}")
        return None
    if len(reference_list) > 0:
        for reference_spec in reference_list:
            if reference_spec.assetPath:
                return reference_spec.assetPath

    return None


def apply_material_variants_to_objects(
    prim_paths: list[str],
    stage: Usd.Stage,
    randomize: bool = True,
):
    """
    Apply UsdPreviewSurface materials to objects with optional randomization.
    Uses standard USD shaders for maximum compatibility.

    Args:
        prim_paths: List of USD prim paths to apply material to.
        stage: The USD stage
        randomize: If True, randomizes color, roughness, and metallic for each prim. Otherwise, uses default values.
    """

    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"Warning: Prim at path '{path}' does not exist. Skipping.")
            continue

        # Generate material properties
        if randomize:
            hue = random.random()
            saturation = random.random()
            value = random.random()
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            mat_color = Gf.Vec3f(rgb[0], rgb[1], rgb[2])
            # roughness is a float between 0 and 1, 0 is smooth, 1 is rough
            mat_roughness = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 1.0)])
            # metallic is a float between 0 and 1, 0 is dielectric, 1 is metal
            mat_metallic = random.choice([0.0, random.uniform(0.8, 1.0)])
        else:
            mat_color = Gf.Vec3f(0.0, 1.0, 1.0)
            mat_roughness = 0.5
            mat_metallic = 0.0

        # Create and bind material for this prim
        material_path = create_usdpreviewsurface_material(stage, prim.GetPath(), mat_color, mat_roughness, mat_metallic)
        bind_material_to_object(prim, material_path, stage)


def create_usdpreviewsurface_material(
    stage: Usd.Stage, prim_path: Sdf.Path, color: Gf.Vec3f, roughness: float, metallic: float
) -> str:
    """
    Create a UsdPreviewSurface material with specified properties under the object's prim path.

    Args:
        stage: The USD stage
        prim_path: Path of the prim this material will be bound to
        color: Diffuse color (RGB, 0-1 range)
        roughness: Reflection roughness (0-1)
        metallic: Metallic value (0-1)

    Returns:
        The material path as string
    """
    # Create material under the object's prim path
    material_path = f"{str(prim_path)}/MaterialVariants"

    # Always create a new material (or update if exists)
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = f"{material_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)

    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    # Set opacity to fully opaque
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)

    # Connect shader output to material surface
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)

    print(
        f"Created UsdPreviewSurface material at {material_path} (color: {color}, roughness: {roughness:.2f}, metallic:"
        f" {metallic:.2f})"
    )

    return material_path


def bind_material_to_object(prim: Usd.Prim, material_path: str, stage: Usd.Stage):
    """
    Recursively bind a material to an object and all its children.

    Args:
        prim: The object to bind the material to
        material_path: USD path to the material to bind
        stage: The USD stage
    """
    if prim.IsA(UsdGeom.Mesh):
        # Bind the material to this object with strong binding
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))

        # Unbind any existing material first
        binding_api.UnbindAllBindings()

        # Note (xinjieyao, 2025.12.17): Bind with "strongerThanDescendants" strength to override child materials
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        print(f"Bound material (strong) to mesh: {prim.GetPath()}")

    # Recursively apply to children
    for child in prim.GetChildren():
        bind_material_to_object(child, material_path, stage)


def randomize_objects_texture(object_names: list[str], num_envs: int, env_ns: str, stage: Usd.Stage):
    assert object_names is not None and len(object_names) > 0
    for object_name in object_names:
        expanded_paths = [f"{env_ns}/env_{i}/{object_name}" for i in range(num_envs)]
        apply_material_variants_to_objects(prim_paths=expanded_paths, stage=stage, randomize=True)
