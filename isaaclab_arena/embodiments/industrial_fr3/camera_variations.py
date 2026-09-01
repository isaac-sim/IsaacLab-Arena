# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera variations shared by the industrial FR3 control modes."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from pxr import Sdf

from isaaclab_arena.variations.camera_extrinsics_variation import (
    CameraExtrinsicsVariation,
    apply_camera_extrinsics_from_sampler,
)
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera, TiledCamera


class ApplyCameraExtrinsicsAndRefresh(apply_camera_extrinsics_from_sampler):
    """Keep Newton's camera frame and the RTX-rendered USD camera in sync."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        sampler: ContinuousSampler,
    ) -> None:
        super().__call__(env, env_ids, asset_cfg, sampler)

        view = self._camera._view
        assert view is not None, "Camera view was not initialized."
        if view.count != 1:
            raise RuntimeError("Rendered FR3 camera variations require num_envs=1 under Newton.")
        translations, _ = view.get_local_poses(wp.from_torch(env_ids))
        _write_usd_camera_translations(self._camera, env_ids, translations.torch)
        self._camera.reset(env_ids)


class RenderedCameraExtrinsicsVariation(CameraExtrinsicsVariation):
    """Camera translation variation whose rendered pixels follow its sampled pose."""

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_name, event_cfg = super().build_event_cfg()
        event_cfg.func = ApplyCameraExtrinsicsAndRefresh
        return event_name, event_cfg


def _write_usd_camera_translations(
    camera: Camera | TiledCamera,
    env_ids: torch.Tensor,
    translations: torch.Tensor,
) -> None:
    """Write sampled local translations to the camera prims consumed by RTX."""

    indices = [int(value) for value in env_ids.detach().cpu().tolist()]
    values = translations.detach().cpu().tolist()
    with Sdf.ChangeBlock():
        for index, xyz in zip(indices, values, strict=True):
            prim = camera._sensor_prims[index].GetPrim()
            attr = prim.GetAttribute("xformOp:translate")
            current = attr.Get()
            if current is None:
                raise RuntimeError(f"Camera prim '{prim.GetPath()}' has no authored xformOp:translate")
            updated = type(current)(*(float(value) for value in xyz))
            if not attr.Set(updated):
                raise RuntimeError(f"Failed to update camera prim '{prim.GetPath()}' xformOp:translate")
