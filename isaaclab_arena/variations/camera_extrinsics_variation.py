# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env camera extrinsics variation.

Adds a small sampler-drawn translation to a camera's nominal local position so
its observed pose drifts from the calibrated reference, modelling a mounting or
calibration error.

Sampled decalibration vectors are expressed in the camera's ROS optical frame:
+X right, +Y down, +Z forward. See :class:`CameraExtrinsicsVariationCfg` for the
sampler axis convention.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_arena.patches import CameraLocalOffsetWriter
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class CameraExtrinsicsVariationCfg(VariationBaseCfg):
    """Configuration for CameraExtrinsicsVariation.

    ``sampler_cfg`` draws 3D translation offsets in the camera ROS optical frame
    (+X right, +Y down, +Z forward).
    """

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.005, -0.005, -0.005],
            high=[0.005, 0.005, 0.005],
        )
    )
    """Uniform distribution over decalibration XYZ in the ROS camera frame [m]."""


class CameraExtrinsicsVariation(RunTimeVariationBase):
    """Vary a camera's extrinsics by adding a small offset to its nominal local position.

    Each reset samples a translation in the camera ROS optical frame (+X right,
    +Y down, +Z forward), converts it to the parent frame, and adds it to the
    nominal local translation.

    Only the camera's local transform is touched, so wrist-mounted cameras keep
    tracking their parent body.

    Args:
        camera_name: Scene-entity name of the target camera.
        cfg: Tunable parameters. Override the translation distribution via
            ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"camera_extrinsics_{camera_name}"``.
    """

    cfg: CameraExtrinsicsVariationCfg

    def __init__(
        self,
        camera_name: str,
        cfg: CameraExtrinsicsVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else CameraExtrinsicsVariationCfg()
        name = name if name is not None else f"camera_extrinsics_{camera_name}"
        super().__init__(cfg=cfg, name=name)
        self.camera_name = camera_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"CameraExtrinsicsVariation on '{self.camera_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_name = f"{self.camera_name}_extrinsics_variation"
        event_cfg = EventTermCfg(
            func=apply_camera_extrinsics_from_sampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.camera_name),
                "sampler": self._sampler,
            },
        )
        return event_name, event_cfg


class apply_camera_extrinsics_from_sampler(ManagerTermBase):
    """Event term: offset a camera's local position by a sampler-drawn delta.

    Sampler output is a translation in the ROS camera frame (+X right, +Y down,
    +Z forward). The nominal local pose is snapshotted on the first call; each
    later call rewrites the translation to nominal + delta so offsets don't
    compound across resets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]

        camera = env.scene[asset_cfg.name]
        assert isinstance(camera, (Camera, TiledCamera)), (
            "apply_camera_extrinsics_from_sampler expects a Camera or TiledCamera at "
            f"scene['{asset_cfg.name}']; got {type(camera).__name__}."
        )
        assert tuple(sampler.shape_per_sample) == (3,), (
            "apply_camera_extrinsics_from_sampler expects a sampler with shape_per_sample (3,) over XYZ; "
            f"got {tuple(sampler.shape_per_sample)}."
        )

        self._camera = camera
        # Writes the sampled pose so both camera.data and the RTX render follow it, on PhysX and Newton.
        self._pose_writer = CameraLocalOffsetWriter(camera)

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
    ):
        # Sample a decalibration vector in the camera's ROS-style optical frame. Pass env_ids so
        # sample listeners (e.g. the variation recorder) can attribute each row to its env.
        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        t_C_Cnew_in_Cros = sample.to(device=self._camera.device, dtype=torch.float32)

        # Isaac Lab tensors use xyzw. 180 deg about +X maps ROS optical axes to OpenGL camera axes.
        q_ros_to_opengl_xyzw = t_C_Cnew_in_Cros.new_tensor((1.0, 0.0, 0.0, 0.0)).expand(len(env_ids), 4)
        t_C_Cnew_in_C = quat_apply(q_ros_to_opengl_xyzw, t_C_Cnew_in_Cros)

        # Offset the camera by the decalibration vector (camera frame) so its rendered pose follows.
        self._pose_writer.apply_camera_frame_offset(t_C_Cnew_in_C, env_ids)
