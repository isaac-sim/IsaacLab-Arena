# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rendering-only material contract from Berkeley's cable task."""


def refresh_reset_cameras(env) -> None:
    """Drive headless camera reads during warm-up without stepping physics."""
    cameras = [
        camera
        for camera in env.scene.sensors.values()
        if getattr(getattr(camera.cfg, "renderer_cfg", None), "renderer_type", None) == "isaac_rtx"
    ]
    if cameras:
        for _ in range(env.cfg.num_rerenders_on_reset):
            env.sim.render()
            for camera in cameras:
                camera.update(0.0, force_recompute=True)


def camera_warmup_recorder_cfg():
    """Use Arena's post-reset hook, after poses are forwarded and before GaP."""
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass

    class CameraWarmup(RecorderTerm):
        def record_post_reset(self, env_ids):
            refresh_reset_cameras(self._env)
            return None, None

    @configclass
    class CameraWarmupCfg(RecorderManagerBaseCfg):
        cable_camera_warmup = RecorderTermCfg(class_type=CameraWarmup)

    return CameraWarmupCfg()


def bind_cable_material(stage, prim, color) -> None:
    """Match EntityRod.apply_display / finish_for(..., 'deformables')."""
    from pxr import Gf, Sdf, UsdShade

    path = prim.GetPath().AppendChild("JacketMaterial")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
