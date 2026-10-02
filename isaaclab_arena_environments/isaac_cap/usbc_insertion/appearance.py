# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Use Isaac Sim's complete OmniPBR implementation for the USB-C asset bundle."""

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from pxr import Sdf, Usd, UsdShade


@clone
def spawn_usbc_with_materials(prim_path, cfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """Spawn a USB-C asset with built-in OmniPBR before cloning it.

    Args:
        prim_path: Asset path, optionally with an environment expression in its parent path.
        cfg: USD spawn configuration.
        translation: Root translation forwarded to the USD spawner.
        orientation: Root quaternion forwarded to the USD spawner.
        **kwargs: Additional USD spawner arguments.

    Returns:
        The first spawned asset root, retaining its authored textures and material bindings.
    """
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        source = shader.GetSourceAsset("mdl")
        if source and source.path.endswith("/materials/OmniPBR/OmniPBR.mdl"):
            # TODO(xinjieyao, 10/1/2026): Check if necessary when migrating to CAP repo.
            # The bundled copy imports OmniPBR_ClearCoat, which is absent from the bundle.
            # Use Kit's complete module while retaining the orange albedo and metal ORM maps.
            shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    return prim
