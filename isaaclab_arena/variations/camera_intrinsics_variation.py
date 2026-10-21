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
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.utils.cameras import ArenaCameraCfg


@configclass
class CameraIntrinsicsVariationCfg(VariationBaseCfg):
    """Configuration for CameraIntrinsicsVariation.

    ``sampler_cfg`` draws one signed fractional perturbation ``d_f`` that scales both focal
    lengths (``fx, fy -> fx, fy * (1 + d_f)``).
    """

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.1],
            high=[0.1],
        )
    )
    """Uniform distribution over a shared fractional focal-length perturbation."""


class CameraIntrinsicsVariation(RunTimeVariationBase):
    """Perturb a pinhole camera's focal lengths on every reset.

    Each reset samples one fractional focal-length perturbation per environment and applies it
    to both ``fx`` and ``fy`` relative to the nominal intrinsic matrices. This preserves the
    centered, square-pixel projection required by Isaac Lab pinhole cameras.
    Because tiled configurations do not support independent rendered projections for
    per-environment runtime intrinsics, this variation makes ``camera_rig`` untiled at build
    time.

    This uses the runtime calibration API introduced by
    https://github.com/isaac-sim/IsaacLab/pull/7916.

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
    """Apply sampled focal-length perturbations without compounding across resets."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]

        camera = env.scene[asset_cfg.name]
        assert isinstance(camera, (Camera, TiledCamera)), (
            "apply_camera_intrinsics_from_sampler expects a Camera or TiledCamera at "
            f"scene['{asset_cfg.name}']; got {type(camera).__name__}."
        )
        assert tuple(sampler.shape_per_sample) == (1,), (
            "apply_camera_intrinsics_from_sampler expects a sampler with shape_per_sample (1,) over "
            f"a shared focal-length delta; got {tuple(sampler.shape_per_sample)}."
        )

        self._camera = camera
        # Snapshotted on first ``__call__``.
        self._nominal_intrinsic_matrices: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
    ):
        if self._nominal_intrinsic_matrices is None:
            self._nominal_intrinsic_matrices = self._camera.data.intrinsic_matrices.torch.clone()

        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        # Focal lengths scale by (1 + d); deltas <= -1 would make them non-positive.
        assert bool((sample > -1.0).all()), (
            "apply_camera_intrinsics_from_sampler expects focal-length deltas > -1.0 so focal lengths stay "
            "positive; got a value <= -1.0. Check your sampler_cfg bounds."
        )
        env_id_list = env_ids.tolist()
        intrinsic_matrices = self._nominal_intrinsic_matrices[env_id_list].clone()
        sample = sample.to(device=intrinsic_matrices.device)
        intrinsic_matrices[:, 0, 0] *= 1.0 + sample[:, 0]
        intrinsic_matrices[:, 1, 1] *= 1.0 + sample[:, 0]
        self._camera.set_intrinsic_matrices(intrinsic_matrices, env_ids=env_id_list)
