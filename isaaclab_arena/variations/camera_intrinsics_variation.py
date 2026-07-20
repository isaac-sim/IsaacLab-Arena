# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera intrinsics variation.

Perturbs a pinhole camera's focal lengths (fx, fy) and principal point (cx, cy)
by sampler-drawn fractional amounts, modelling an intrinsic-calibration error.
Image resolution is left untouched. The perturbation is sampled on every reset and
applied via USD pinhole parameters on the live camera prim.
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
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.utils.cameras import ArenaCameraCfg


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


class CameraIntrinsicsVariation(RunTimeVariationBase):
    """Perturb a pinhole camera's focal lengths and principal point on every reset.

    Each reset samples a ``(d_fx, d_fy, d_cx, d_cy)`` fraction per env and writes the
    perturbed USD pinhole parameters on the live camera prim. Nominal apertures are
    snapshotted on the first event call so perturbations do not compound.

    Tiled cameras share one USD sensor across envs, so a per-env intrinsic edit would leak
    across all tiles. When ``camera_rig`` is provided, this variation forces that rig untiled
    at build time (via :meth:`apply_build_time_effects`) so the per-env perturbation takes effect.

    Args:
        camera_name: Scene-entity name of the target camera.
        camera_rig: The camera-rig cfg to force untiled when this variation is enabled. Pass the
            embodiment's ``camera_config``. When ``None`` the camera is assumed already untiled.
        cfg: Tunable parameters. Override the perturbation distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_intrinsics_{camera_name}"``.
    """

    cfg: CameraIntrinsicsVariationCfg

    def __init__(
        self,
        camera_name: str,
        camera_rig: ArenaCameraCfg | None = None,
        cfg: CameraIntrinsicsVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraIntrinsicsVariationCfg()
        name = name if name is not None else f"camera_intrinsics_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self.camera_name = camera_name
        self._camera_rig = camera_rig

    def apply_build_time_effects(self) -> None:
        """Force the target camera's rig untiled so the per-env perturbation takes effect."""
        if self._camera_rig is not None:
            self._camera_rig.set_use_tiled_camera(False)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"CameraIntrinsicsVariation on '{self.camera_name}' is enabled but no sampler is set; "
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
