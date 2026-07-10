# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera intrinsics variations.

Perturbs a pinhole camera's focal lengths (fx, fy) and principal point (cx, cy)
by sampler-drawn fractional amounts, modelling an intrinsic-calibration error.
Image resolution is left untouched.

Two flavours are provided:

* :class:`CameraIntrinsicsBuildTimeVariation` — sampled once and written into the
  camera's spawn cfg before the scene is composed.
* :class:`CameraIntrinsicsRunTimeVariation` — sampled on every reset and applied via
  USD pinhole parameters on the live camera prim.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors.camera.camera_cfg import CameraCfg
    from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg


@configclass
class CameraIntrinsicsVariationCfg(VariationBaseCfg):
    """Shared configuration for camera intrinsics variations.

    ``sampler_cfg`` draws four signed fractional perturbations ``(d_fx, d_fy, d_cx, d_cy)``.
    ``d_fx`` / ``d_fy`` scale the focal lengths (``fx -> fx * (1 + d_fx)``), while ``d_cx`` /
    ``d_cy`` scale the principal point relative to its nominal image-centre value
    (``cx -> (width / 2) * (1 + d_cx)``).
    """

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.1, -0.1, -0.1, -0.1],
            high=[0.1, 0.1, 0.1, 0.1],
        )
    )
    """Uniform distribution over fractional (fx, fy, cx, cy) perturbations."""


CameraIntrinsicsBuildTimeVariationCfg = CameraIntrinsicsVariationCfg
CameraIntrinsicsRunTimeVariationCfg = CameraIntrinsicsVariationCfg


def _perturbed_aperture_params(
    nominal_horizontal_aperture: float,
    nominal_vertical_aperture: float,
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
) -> dict[str, float]:
    """Map fractional intrinsic perturbations to USD pinhole aperture parameters.

    ``focal_length`` is held fixed, so focal-length scaling is realised through
    aperture size and principal-point shifts through aperture offsets.
    """
    horizontal_aperture = nominal_horizontal_aperture / (1.0 + d_fx)
    vertical_aperture = nominal_vertical_aperture / (1.0 + d_fy)
    return {
        "horizontal_aperture": horizontal_aperture,
        "vertical_aperture": vertical_aperture,
        "horizontal_aperture_offset": horizontal_aperture * d_cx / 2.0,
        "vertical_aperture_offset": vertical_aperture * d_cy / 2.0,
    }


class CameraIntrinsicsBuildTimeVariation(BuildTimeVariationBase):
    """Perturb a pinhole camera's focal lengths and principal point at build time.

    A single ``(d_fx, d_fy, d_cx, d_cy)`` fraction is sampled per env build and written
    into the camera's ``PinholeCameraCfg`` spawn parameters before the scene is composed.
    Nominal apertures are snapshotted on construction so perturbations do not compound.

    Args:
        camera_cfg: The camera cfg to mutate; its ``spawn`` must be a ``PinholeCameraCfg``.
        camera_name: Camera identifier, used only to name the variation.
        cfg: Tunable parameters. Override the perturbation distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_intrinsics_build_time_{camera_name}"``.
    """

    cfg: CameraIntrinsicsBuildTimeVariationCfg

    def __init__(
        self,
        camera_cfg: CameraCfg | TiledCameraCfg,
        camera_name: str,
        cfg: CameraIntrinsicsBuildTimeVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraIntrinsicsBuildTimeVariationCfg()
        name = name if name is not None else f"camera_intrinsics_build_time_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self._camera_cfg = camera_cfg
        self._nominal_horizontal_aperture = camera_cfg.spawn.horizontal_aperture
        self._nominal_vertical_aperture = camera_cfg.spawn.vertical_aperture

    def apply(self) -> None:
        assert self.sampler is not None, "CameraIntrinsicsBuildTimeVariation: sampler not set."
        d_fx, d_fy, d_cx, d_cy = self.sampler.sample(num_samples=1)[0].tolist()
        params = _perturbed_aperture_params(
            self._nominal_horizontal_aperture,
            self._nominal_vertical_aperture,
            d_fx,
            d_fy,
            d_cx,
            d_cy,
        )
        spawn = self._camera_cfg.spawn
        spawn.horizontal_aperture = params["horizontal_aperture"]
        spawn.vertical_aperture = params["vertical_aperture"]
        spawn.horizontal_aperture_offset = params["horizontal_aperture_offset"]
        spawn.vertical_aperture_offset = params["vertical_aperture_offset"]


class CameraIntrinsicsRunTimeVariation(RunTimeVariationBase):
    """Perturb a pinhole camera's focal lengths and principal point on every reset.

    Each reset samples a ``(d_fx, d_fy, d_cx, d_cy)`` fraction per env and writes the
    perturbed USD pinhole parameters on the live camera prim. Nominal apertures are
    snapshotted on the first event call so perturbations do not compound.

    Args:
        camera_name: Scene-entity name of the target camera.
        cfg: Tunable parameters. Override the perturbation distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_intrinsics_{camera_name}"``.
    """

    cfg: CameraIntrinsicsRunTimeVariationCfg

    def __init__(
        self,
        camera_name: str,
        cfg: CameraIntrinsicsRunTimeVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraIntrinsicsRunTimeVariationCfg()
        name = name if name is not None else f"camera_intrinsics_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self.camera_name = camera_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"CameraIntrinsicsRunTimeVariation on '{self.camera_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_name = f"{self.camera_name}_intrinsics_variation"
        event_cfg = EventTermCfg(
            func=apply_camera_intrinsics_from_sampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.camera_name),
                "sampler": self._sampler,
            },
        )
        return event_name, event_cfg


class apply_camera_intrinsics_from_sampler(ManagerTermBase):
    """Event term: perturb a camera's USD pinhole parameters by a sampler-drawn delta.

    The nominal aperture sizes are snapshotted on the first call; each later call
    rewrites USD parameters from that nominal plus the sampled perturbation so offsets
    do not compound across resets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]

        camera = env.scene[asset_cfg.name]
        assert isinstance(camera, (Camera, TiledCamera)), (
            "apply_camera_intrinsics_from_sampler expects a Camera or TiledCamera at "
            f"scene['{asset_cfg.name}']; got {type(camera).__name__}."
        )
        assert tuple(sampler.shape_per_sample) == (4,), (
            "apply_camera_intrinsics_from_sampler expects a sampler with shape_per_sample (4,) over "
            f"(d_fx, d_fy, d_cx, d_cy); got {tuple(sampler.shape_per_sample)}."
        )

        self._camera = camera
        # Snapshotted on first ``__call__``.
        self._nominal_horizontal_aperture: float | None = None
        self._nominal_vertical_aperture: float | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
    ):
        if self._nominal_horizontal_aperture is None or self._nominal_vertical_aperture is None:
            sensor_prim = self._camera._sensor_prims[0]
            self._nominal_horizontal_aperture = sensor_prim.GetHorizontalApertureAttr().Get()
            self._nominal_vertical_aperture = sensor_prim.GetVerticalApertureAttr().Get()

        assert self._nominal_horizontal_aperture is not None
        assert self._nominal_vertical_aperture is not None

        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        env_id_list = env_ids.tolist()
        for env_id, deltas in zip(env_id_list, sample.tolist()):
            params = _perturbed_aperture_params(
                self._nominal_horizontal_aperture,
                self._nominal_vertical_aperture,
                *deltas,
            )
            sensor_prim = self._camera._sensor_prims[env_id]
            sensor_prim.GetHorizontalApertureAttr().Set(params["horizontal_aperture"])
            sensor_prim.GetVerticalApertureAttr().Set(params["vertical_aperture"])
            sensor_prim.GetHorizontalApertureOffsetAttr().Set(params["horizontal_aperture_offset"])
            sensor_prim.GetVerticalApertureOffsetAttr().Set(params["vertical_aperture_offset"])

        self._camera._update_intrinsic_matrices(env_id_list)
