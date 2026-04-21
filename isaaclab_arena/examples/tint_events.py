# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env visual tint event that preserves the asset's original texture.

Unlike :class:`isaaclab.envs.mdp.randomize_visual_color`, which *replaces* the
bound material with a fresh ``OmniPBR`` instance (dropping any diffuse
texture), this term writes a random tint color onto the material that the USD
asset already has bound. For textured assets the effect is a hue shift applied
on top of the original texture.

The term is scoped to the prototype in ``compile_env_notebook.py`` and kept
alongside it so it's easy to iterate without touching the main arena package.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import get_current_stage
from pxr import Gf, Sdf, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers.manager_term_cfg import EventTermCfg


class randomize_visual_diffuse_tint(ManagerTermBase):
    """Randomize the diffuse tint of each env's copy of an asset in place.

    Finds the material bound to each Mesh prim under the asset (per env) and
    writes a random color to the bound shader's tint input. The material is
    **not** replaced, so any diffuse texture is preserved — the tint simply
    multiplies with it.

    Supported shaders:
    * ``UsdPreviewSurface``: writes ``inputs:diffuseColor``.
    * MDL shaders (e.g. ``OmniPBR``): writes ``inputs:diffuse_tint``.

    Requirements:
    * ``env.cfg.scene.replicate_physics`` must be False. With replication on,
      every env shares a single source material and per-env tinting is
      impossible.
    * Run with ``mode="prestartup"`` for a stable tint per env, or
      ``mode="reset"`` to resample on every episode reset (note: the current
      implementation resamples for *all* envs, matching the ignored-``env_ids``
      behavior of :class:`~isaaclab.envs.mdp.randomize_visual_color`).

    Params (on the :class:`EventTermCfg`):
    * ``asset_cfg`` (:class:`SceneEntityCfg`): which scene asset to tint.
    * ``colors`` (``dict[str, tuple[float, float]]``): per-channel uniform
      ranges, e.g. ``{"r": (0.4, 1.0), "g": (0.4, 1.0), "b": (0.4, 1.0)}``.
      Narrow ranges near 1.0 produce subtle tints; wide ranges look
      aggressive.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "randomize_visual_diffuse_tint requires "
                "scene.replicate_physics=False (all env clones otherwise share "
                "a single source material)."
            )

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]

        # ``asset.cfg.prim_path`` is already a regex-ready string such as
        # ``/World/envs/env_.*/cracker_box``. We match descendants by anchoring
        # the pattern and requiring a trailing ``/`` so that sibling prims
        # with a common prefix (e.g. ``cracker_box_2``) don't match.
        prim_path_re = re.compile("^" + asset.cfg.prim_path + "(/|$)")

        stage = get_current_stage()
        # One shader-tint target per env. If an asset has multiple textured
        # meshes we tint all of them — they'll share the same random color
        # per env, which is what a human would expect for a uniform "tint".
        self._shader_targets: list[tuple[str, str]] = []  # (shader_path, attr_name)
        seen_materials: set[str] = set()

        for prim in stage.Traverse():
            prim_path_str = str(prim.GetPath())
            if not prim_path_re.match(prim_path_str):
                continue
            if prim.GetTypeName() != "Mesh":
                continue

            # Make sure the cloned mesh isn't an instance (otherwise all envs
            # share a single material, defeating per-env tinting).
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

            mbapi = UsdShade.MaterialBindingAPI(prim)
            material = mbapi.ComputeBoundMaterial()[0]
            if not material:
                continue

            material_path = str(material.GetPath())
            # Avoid writing to the same material twice if multiple meshes
            # under the same env share one material.
            if material_path in seen_materials:
                continue
            seen_materials.add(material_path)

            # The surface output of a Material prim drives a Shader prim.
            surface_output = material.GetSurfaceOutput()
            source = surface_output.GetConnectedSource() if surface_output else None
            if not source:
                continue
            shader_api = source[0]
            shader_prim = shader_api.GetPrim()

            attr_name = _tint_attribute_for_shader(shader_prim)
            if attr_name is None:
                continue
            self._shader_targets.append((str(shader_prim.GetPath()), attr_name))

        self._num_envs = env.scene.num_envs

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        colors: dict[str, tuple[float, float]],
    ) -> None:
        if not self._shader_targets:
            return

        low = torch.tensor([colors["r"][0], colors["g"][0], colors["b"][0]])
        high = torch.tensor([colors["r"][1], colors["g"][1], colors["b"][1]])
        samples = low + (high - low) * torch.rand((len(self._shader_targets), 3))

        stage = get_current_stage()
        for (shader_path, attr_name), rgb in zip(self._shader_targets, samples.tolist()):
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim.IsValid():
                continue
            attr = shader_prim.GetAttribute(attr_name)
            if not attr:
                # Create the attribute lazily; OmniPBR shader inputs aren't
                # authored until they're overridden.
                attr = shader_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Color3f, custom=False)
            attr.Set(Gf.Vec3f(*rgb))


def _tint_attribute_for_shader(shader_prim) -> str | None:
    """Return the shader input used to tint a surface, or ``None`` if unknown.

    * ``UsdPreviewSurface`` uses ``inputs:diffuseColor`` (which, when a diffuse
      texture is connected, is multiplied with the texture sample).
    * MDL shaders (OmniPBR and friends) expose ``inputs:diffuse_tint``.
    """
    info_id_attr = shader_prim.GetAttribute("info:id")
    info_id = info_id_attr.Get() if info_id_attr else None
    if info_id == "UsdPreviewSurface":
        return "inputs:diffuseColor"

    # MDL shaders advertise their source through ``info:mdl:sourceAsset``.
    mdl_source_attr = shader_prim.GetAttribute("info:mdl:sourceAsset")
    if mdl_source_attr and mdl_source_attr.Get():
        return "inputs:diffuse_tint"

    # Unknown shader type — skip rather than scribble attributes that won't
    # be consumed by any rendering path.
    return None
