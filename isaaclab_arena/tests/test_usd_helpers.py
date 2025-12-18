# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_apply_material_variants_to_objects(simulation_app) -> bool:
    """Test applying UsdPreviewSurface materials to objects with randomization."""
    from pxr import Usd, UsdShade

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.utils.usd_helpers import apply_material_variants_to_objects

    stage = Usd.Stage.CreateInMemory()

    root = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root)

    # Get asset registry and reference two cracker boxes
    asset_registry = AssetRegistry()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    # Create two cracker box prims by referencing the USD
    box1_prim = stage.DefinePrim("/World/cracker_box_1", "Xform")
    box1_prim.GetReferences().AddReference(cracker_box.usd_path)

    box2_prim = stage.DefinePrim("/World/cracker_box_2", "Xform")
    box2_prim.GetReferences().AddReference(cracker_box.usd_path)

    # Apply randomized materials
    prim_paths = ["/World/cracker_box_1", "/World/cracker_box_2"]
    apply_material_variants_to_objects(
        prim_paths=prim_paths,
        stage=stage,
        randomize=True,
    )

    # Verify materials were created under each object's prim path
    material_paths = [
        "/World/cracker_box_1/MaterialVariants",
        "/World/cracker_box_2/MaterialVariants",
    ]
    for material_path in material_paths:
        material_prim = stage.GetPrimAtPath(material_path)
        assert material_prim.IsValid(), f"Material prim not created at {material_path}"

        # Verify shader has UsdPreviewSurface ID
        shader = UsdShade.Shader.Get(stage, f"{material_path}/Shader")
        shader_id = shader.GetIdAttr().Get()
        assert shader_id == "UsdPreviewSurface", f"Shader ID is {shader_id}, expected 'UsdPreviewSurface'"

        # Verify shader inputs exist
        assert shader.GetInput("diffuseColor"), "diffuseColor not found"
        assert shader.GetInput("roughness"), "roughness not found"
        assert shader.GetInput("metallic"), "metallic not found"

    return True


def test_apply_material_variants_to_objects():
    result = run_simulation_app_function(
        _test_apply_material_variants_to_objects,
        headless=HEADLESS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_apply_material_variants_to_objects()

