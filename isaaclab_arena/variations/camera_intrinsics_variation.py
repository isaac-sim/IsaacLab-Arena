# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera intrinsics variation.

Perturbs a pinhole camera's focal lengths (fx, fy) by sampler-drawn fractional amounts.
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
    """Configuration for CameraIntrinsicsVariation.

    ``sampler_cfg`` draws two signed fractional perturbations ``(d_fx, d_fy)`` that scale the
    focal lengths (``fx -> fx * (1 + d_fx)``).
    """

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.1, -0.1],
            high=[0.1, 0.1],
        )
    )
    """Uniform distribution over fractional (fx, fy) perturbations."""


class CameraIntrinsicsVariation(RunTimeVariationBase):
    """Perturb a pinhole camera's focal lengths on every reset.

    Each reset samples a ``(d_fx, d_fy)`` fraction per env and writes the perturbed USD
    aperture parameters on the live camera prim. Nominal apertures are snapshotted on the
    first event call so perturbations do not compound.

    Tiled cameras share one USD sensor across envs, so a per-env intrinsic edit would leak
    across all tiles. This variation forces ``camera_rig`` untiled at build time (via
    _prepare_at_build_time) so the per-env perturbation takes effect.

    Args:
        camera_name: Scene-entity name of the target camera.
        camera_rig: The camera-rig cfg to force untiled. Pass the embodiment's ``camera_config``.
        cfg: Tunable parameters. Override the perturbation distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_intrinsics_{camera_name}"``.
    """

    cfg: CameraIntrinsicsVariationCfg

    def __init__(
        self,
        camera_name: str,
        camera_rig: ArenaCameraCfg,
        cfg: CameraIntrinsicsVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraIntrinsicsVariationCfg()
        name = name if name is not None else f"camera_intrinsics_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self.camera_name = camera_name
        self._camera_rig = camera_rig

    def _prepare_at_build_time(self) -> None:
        """Force the target camera's rig untiled so the per-env perturbation takes effect."""
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
    """Event term: perturb a camera's USD aperture parameters by a sampler-drawn delta.

    The nominal aperture sizes are snapshotted on the first call; each later call
    rewrites the apertures from that nominal plus the sampled perturbation so they
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
        assert tuple(sampler.shape_per_sample) == (2,), (
            "apply_camera_intrinsics_from_sampler expects a sampler with shape_per_sample (2,) over "
            f"(d_fx, d_fy); got {tuple(sampler.shape_per_sample)}."
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
        for env_id, (d_fx, d_fy) in zip(env_id_list, sample.tolist()):
            # ``focal_length`` is held fixed, so focal-length scaling is realised through aperture size.
            sensor_prim = self._camera._sensor_prims[env_id]
            sensor_prim.GetHorizontalApertureAttr().Set(self._nominal_horizontal_aperture / (1.0 + d_fx))
            sensor_prim.GetVerticalApertureAttr().Set(self._nominal_vertical_aperture / (1.0 + d_fy))

        self._camera._update_intrinsic_matrices(env_id_list)
