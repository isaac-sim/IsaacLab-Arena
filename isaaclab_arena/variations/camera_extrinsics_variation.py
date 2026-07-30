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

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

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
        # Snapshotted on first ``__call__``.
        self._t_parent_C_in_parent: torch.Tensor | None = None
        self._q_parent_C_xyzw: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
    ):
        view = self._camera._view
        assert view is not None, "Camera view was not initialized."
        if self._t_parent_C_in_parent is None:
            # NOTE(alexmillane): in the version of Isaac Lab that we're using, the get_local_poses()
            # method claims to return w,x,y,z, but it actually returns x,y,z,w.
            # I am testing this in the test test_isaaclab_bug_get_local_poses.py.
            t_parent_C, q_parent_C_xyzw = view.get_local_poses()
            self._t_parent_C_in_parent = t_parent_C.torch.detach().clone()
            self._q_parent_C_xyzw = q_parent_C_xyzw.torch.detach().clone()

        assert self._t_parent_C_in_parent is not None
        assert self._q_parent_C_xyzw is not None

        # Sample a decalibration vector in the camera's ROS-style optical frame. Pass env_ids so
        # sample listeners (e.g. the variation recorder) can attribute each row to its env.
        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        t_C_Cnew_in_Cros = sample.to(device=self._t_parent_C_in_parent.device, dtype=self._t_parent_C_in_parent.dtype)

        # Isaac Lab tensors use xyzw. 180 deg about +X maps ROS optical axes to OpenGL camera axes.
        q_ros_to_opengl_xyzw = t_C_Cnew_in_Cros.new_tensor((1.0, 0.0, 0.0, 0.0)).expand(len(env_ids), 4)
        t_C_Cnew_in_C = quat_apply(q_ros_to_opengl_xyzw, t_C_Cnew_in_Cros)

        # Compose the decalibration vector in the camera's parent frame, by first rotating into the
        # parent's frame, and then adding the original translation.
        t_C_Cnew_in_parent = quat_apply(self._q_parent_C_xyzw[env_ids], t_C_Cnew_in_C)
        t_parent_Cnew_in_parent = self._t_parent_C_in_parent[env_ids] + t_C_Cnew_in_parent

        # Apply the the sim.
        view.set_local_poses(translations=t_parent_Cnew_in_parent, orientations=None, indices=wp.from_torch(env_ids))
