# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time camera intrinsics variation.

Perturbs a pinhole camera's focal lengths (fx, fy) and principal point (cx, cy)
by sampler-drawn fractional amounts, modelling an intrinsic-calibration error.
Image resolution is left untouched.

The perturbations are realised through the USD pinhole parameters on the camera's
spawn cfg (``focal_length`` is held fixed):

* fx and fy scale inversely with ``horizontal_aperture`` / ``vertical_aperture``.
* cx and cy shift with ``horizontal_aperture_offset`` / ``vertical_aperture_offset``,
  whose nominal value places the principal point at the image centre.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_cfg import CameraCfg
    from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg


@configclass
class CameraIntrinsicsVariationCfg(VariationBaseCfg):
    """Configuration for CameraIntrinsicsVariation.

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


class CameraIntrinsicsVariation(BuildTimeVariationBase):
    """Perturb a pinhole camera's focal lengths and principal point at build time.

    A single ``(d_fx, d_fy, d_cx, d_cy)`` fraction is sampled per env build and written
    into the camera's ``PinholeCameraCfg`` spawn parameters before the scene is composed.
    Nominal apertures are snapshotted on construction so perturbations do not compound.

    Args:
        camera_cfg: The camera cfg to mutate; its ``spawn`` must be a ``PinholeCameraCfg``.
        camera_name: Camera identifier, used only to name the variation.
        cfg: Tunable parameters. Override the perturbation distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_intrinsics_{camera_name}"``.
    """

    cfg: CameraIntrinsicsVariationCfg

    def __init__(
        self,
        camera_cfg: CameraCfg | TiledCameraCfg,
        camera_name: str,
        cfg: CameraIntrinsicsVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraIntrinsicsVariationCfg()
        name = name if name is not None else f"camera_intrinsics_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self._camera_cfg = camera_cfg
        self._nominal_horizontal_aperture = camera_cfg.spawn.horizontal_aperture
        self._nominal_vertical_aperture = camera_cfg.spawn.vertical_aperture

    def apply(self) -> None:
        assert self.sampler is not None, "CameraIntrinsicsVariation: sampler not set."
        d_fx, d_fy, d_cx, d_cy = self.sampler.sample(num_samples=1)[0].tolist()

        spawn = self._camera_cfg.spawn
        # fx, fy scale inversely with aperture (focal_length fixed).
        spawn.horizontal_aperture = self._nominal_horizontal_aperture / (1.0 + d_fx)
        spawn.vertical_aperture = self._nominal_vertical_aperture / (1.0 + d_fy)
        # A principal-point shift of a fraction of the image maps to the same fraction of the
        # aperture; the nominal centre is half the image, hence the factor of one half.
        spawn.horizontal_aperture_offset = spawn.horizontal_aperture * d_cx / 2.0
        spawn.vertical_aperture_offset = spawn.vertical_aperture * d_cy / 2.0
