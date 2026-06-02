# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env camera extrinsics variation.

Adds a small sampler-drawn translation to a camera's nominal local position so
its observed pose drifts from the calibrated reference, modelling a mounting or
calibration error. See :class:`CameraExtrinsicsVariationCfg` for the sampler
axis convention.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_arena.variations.sampler_base import SamplerBase
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class CameraExtrinsicsVariationCfg(VariationBaseCfg):
    """Configuration for CameraExtrinsicsVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.005, -0.005, -0.005],
            high=[0.005, 0.005, 0.005],
        )
    )
    """3D translation distribution in the camera's ROS-style optical frame, axes (x_right, y_down, z_forward)."""


class CameraExtrinsicsVariation(RunTimeVariationBase):
    """Vary a camera's extrinsics by adding a small offset to its nominal local position.

    Only the camera's local transform is touched, so wrist-mounted cameras keep
    tracking their parent body.

    Args:
        camera_name: Scene-entity name of the target camera.
        cfg: Tunable parameters. Override the translation distribution via
            ``cfg.sampler_cfg``.
    """

    name = "camera_extrinsics"

    cfg: CameraExtrinsicsVariationCfg

    def __init__(
        self,
        camera_name: str,
        cfg: CameraExtrinsicsVariationCfg | None = None,
    ):
        super().__init__(cfg=cfg if cfg is not None else CameraExtrinsicsVariationCfg())
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

    The nominal local pose is snapshotted on the first call; each later call
    rewrites the translation to nominal + delta so offsets don't compound
    across resets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: SamplerBase = cfg.params["sampler"]

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
        sampler: SamplerBase,
    ):
        view = self._camera._view
        if self._t_parent_C_in_parent is None:
            t_parent_C, q_parent_C_wxyz = view.get_local_poses()
            print(f"t_parent_C: {t_parent_C}")
            print(f"q_parent_C_wxyz: {q_parent_C_wxyz}")
            self._t_parent_C_in_parent = t_parent_C.detach().clone()
            self._q_parent_C_xyzw = torch.roll(q_parent_C_wxyz.detach(), shifts=-1, dims=-1).clone()

        assert self._t_parent_C_in_parent is not None
        assert self._q_parent_C_xyzw is not None

        sample = sampler.sample(num_samples=len(env_ids))
        print(f"sample: {sample}")
        t_Cnew_C_in_C = sample.to(device=self._t_parent_C_in_parent.device, dtype=self._t_parent_C_in_parent.dtype)

        # print(f"self._q_parent_C_xyzw[env_ids]: {self._q_parent_C_xyzw[env_ids]}")
        # print(f"self._q_parent_C_xyzw: {self._q_parent_C_xyzw}")
        # print(f"self._t_parent_C_in_parent: {self._t_parent_C_in_parent}")

        # deltas_opengl = deltas_input * deltas_input.new_tensor((-1.0, 1.0, -1.0))
        t_Cnew_C_in_parent = quat_apply(self._q_parent_C_xyzw[env_ids], t_Cnew_C_in_C)
        t_Cnew_C_in_parent = self._t_parent_C_in_parent[env_ids] + t_Cnew_C_in_parent
        # view.set_local_poses(translations=t_Cnew_C_in_parent, orientations=None, indices=env_ids)

        # Empirical sign mapping from the sampler's ROS-style input axes to the
        # camera's OpenGL local frame for the droid wrist mount. The textbook
        # ROS -> OpenGL conversion is ``(+x, -y, -z)``; the mapping below was
        # verified visually. TODO: replace with the standard conversion once
        # the underlying discrepancy (suspected spawn-path quaternion order or
        # renderer-side image flip) is understood.
        # deltas_opengl = deltas_input * deltas_input.new_tensor((-1.0, 1.0, -1.0))
        # deltas_parent = quat_apply(self._q_parent_C_xyzw[env_ids], deltas_opengl)
        # new_local_pos = self._t_parent_C_in_parent[env_ids] + deltas_parent
        # view.set_local_poses(translations=new_local_pos, orientations=None, indices=env_ids)
