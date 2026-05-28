# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-env camera decalibration variation.

Adds a small sampler-drawn translation on top of a camera's nominal local
position so its observed pose drifts from the calibrated reference. Deltas are
sampled in the camera's own frame (see
:class:`CameraDecalibrationVariationCfg` for the axis convention) and rotated
into the camera's parent (USD local) frame before being applied as a local
translation. The nominal (un-decalibrated) local translation is snapshotted on
the first event tick, and each subsequent tick rewrites the local translation
to ``nominal + R · delta`` for the envs being touched, so deltas don't compound
across resets.
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
class CameraDecalibrationVariationCfg(VariationBaseCfg):
    """Configuration for :class:`CameraDecalibrationVariation`.

    Attributes:
        mode: Event mode forwarded to :class:`EventTermCfg`. ``"reset"`` resamples
            the decalibration on every episode reset; ``"prestartup"`` picks a
            stable per-env offset that persists for the run.
        sampler: 3D translation distribution in the camera's own frame, with
            axes interpreted as ``(x_right_in_image, y_down_in_image,
            z_forward_into_scene)`` (ROS optical-frame convention). Defaults
            to a uniform over ``[-5 mm, +5 mm]`` on every axis.
    """

    mode: str = "reset"
    sampler: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.005, -0.005, -0.005],
            high=[0.005, 0.005, 0.005],
        )
    )


class CameraDecalibrationVariation(RunTimeVariationBase):
    """Decalibrate a camera by adding a small offset to its nominal local position.

    The camera's nominal (un-decalibrated) local translation is captured once
    on the first event tick, and every subsequent tick rewrites the local
    translation to ``nominal + delta`` for the envs being touched. This models
    a small mounting / calibration error without compounding across resets,
    and keeps wrist-mounted cameras tracking their parent body because only
    the camera's local transform (not its world pose) is modified.

    Args:
        camera_name: The scene-entity name of the target camera (e.g.
            ``"wrist_cam"``). Must resolve to a :class:`Camera` or
            :class:`TiledCamera` in ``env.scene``.
        cfg: Tunable parameters. Defaults to a
            :class:`CameraDecalibrationVariationCfg` with a ``±5 mm`` per-axis
            uniform sampler and reset-mode resampling.
        sampler: Optional override for the translation distribution. If
            ``None``, the sampler in ``cfg`` is used.
    """

    name = "camera_decalibration"

    cfg: CameraDecalibrationVariationCfg

    def __init__(
        self,
        camera_name: str,
        cfg: CameraDecalibrationVariationCfg | None = None,
        sampler: SamplerBase | UniformSamplerCfg | None = None,
    ):
        super().__init__(cfg=cfg if cfg is not None else CameraDecalibrationVariationCfg())
        self.camera_name = camera_name
        self.set_sampler(sampler if sampler is not None else self.cfg.sampler)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"CameraDecalibrationVariation on '{self.camera_name}' is enabled but no sampler is set; "
            "call .set_sampler(...) before building the env."
        )
        event_name = f"{self.camera_name}_decalibration_variation"
        event_cfg = EventTermCfg(
            func=decalibrate_camera_from_sampler,
            mode=self.cfg.mode,
            params={
                "asset_cfg": SceneEntityCfg(self.camera_name),
                "sampler": self._sampler,
            },
        )
        return event_name, event_cfg


class decalibrate_camera_from_sampler(ManagerTermBase):
    """Add a sampler-drawn camera-frame translation to a camera's nominal local position.

    Snapshots the camera's factory local pose on first call and rewrites the
    local translation to ``nominal + R · delta`` per env on subsequent calls
    so decalibrations don't compound across resets. Operates on the camera's
    underlying :class:`~isaaclab.sim.views.XformPrimView` via
    ``set_local_poses`` so wrist-mounted cameras keep tracking their parent
    body.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: SamplerBase = cfg.params["sampler"]

        camera = env.scene[asset_cfg.name]
        assert isinstance(camera, (Camera, TiledCamera)), (
            f"decalibrate_camera_from_sampler expects a Camera or TiledCamera at "
            f"scene['{asset_cfg.name}']; got {type(camera).__name__}."
        )
        assert tuple(sampler.event_shape) == (3,), (
            "decalibrate_camera_from_sampler expects a sampler with event_shape (3,) over XYZ; "
            f"got {tuple(sampler.event_shape)}."
        )

        self._camera = camera
        # Snapshotted on first ``__call__`` (the camera's view is only
        # initialised by the sensor's lifecycle hooks at that point).
        self._t_parent_C: torch.Tensor | None = None
        self._q_parent_C_xyzw: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: SamplerBase,
    ):
        view = self._camera._view
        if self._t_parent_C is None:
            t_parent_C, q_parent_C_wxyz = view.get_local_poses()
            self._t_parent_C = t_parent_C.detach().clone()
            self._q_parent_C_xyzw = torch.roll(q_parent_C_wxyz.detach(), shifts=-1, dims=-1).clone()

        assert self._t_parent_C is not None
        assert self._q_parent_C_xyzw is not None

        sample = sampler.sample(num_samples=len(env_ids))
        deltas_input = sample.to(device=self._t_parent_C.device, dtype=self._t_parent_C.dtype)
        # Empirical sign mapping from the sampler's ROS-style input axes to the
        # camera's OpenGL local frame for the droid wrist mount. The textbook
        # ROS -> OpenGL conversion is ``(+x, -y, -z)``; the mapping below was
        # verified visually. TODO: replace with the standard conversion once
        # the underlying discrepancy (suspected spawn-path quaternion order or
        # renderer-side image flip) is understood.
        deltas_opengl = deltas_input * deltas_input.new_tensor((-1.0, 1.0, -1.0))
        deltas_parent = quat_apply(self._q_parent_C_xyzw[env_ids], deltas_opengl)
        new_local_pos = self._t_parent_C[env_ids] + deltas_parent
        view.set_local_poses(translations=new_local_pos, orientations=None, indices=env_ids)
