# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Make a camera's local-pose change take effect on the RTX render under Newton.

Writing ``camera._view.set_local_poses`` (or the public ``camera.set_world_poses``) updates the physics
FrameView. Under PhysX that view is a ``FabricFrameView``, which forwards the write to a ``UsdFrameView``
over the same prim, and the RTX renderer -- which reads the USD camera prim -- follows. Under Newton the
FrameView (``NewtonSiteFrameView``) updates only in-memory Warp state, so the render does not move.

This writer also mirrors the local pose onto the USD camera prim via ``UsdFrameView`` (which
``IsaacRtxRenderer`` reads on both backends) and calls ``camera.reset`` so ``camera.data.pos_w`` reflects
the new pose. On PhysX that second write is redundant -- it re-authors the same xformOps the FrameView
already authored -- but it is harmless, and keeping it unconditional keeps both backends on one path.
"""

from __future__ import annotations

import torch

import warp as wp


# TODO(alexmillane, 2026-09-08): [isaac-lab-camera-pose-write-bug] Remove this whole class once IsaacLab's
# NewtonSiteFrameView mirrors poses onto the USD camera prim like FabricFrameView; then a FrameView write
# plus camera.reset suffice on both backends.
class CameraPoseWriter:
    """Write a camera's local pose so both ``camera.data`` and the RTX render follow it, on any backend."""

    def __init__(self, camera) -> None:
        self._camera = camera
        self._physics_view = None
        self._usd_view = None
        self._usd_device: torch.device | None = None

    def _ensure_initialized(self) -> None:
        # Deferred rather than done in __init__ so the writer can be constructed before the simulation
        # starts, i.e. before the camera's FrameView and USD prim exist.
        if self._usd_view is not None:
            return
        from isaaclab.sim.views.usd_frame_view import UsdFrameView

        self._physics_view = self._camera._view
        assert self._physics_view is not None, "Camera FrameView was not initialized."
        self._usd_view = UsdFrameView(self._camera.cfg.prim_path)
        self._usd_device = self._usd_view.get_local_poses()[0].torch.device
        self._assert_views_share_local_frame()

    def _assert_views_share_local_frame(self) -> None:
        """Check the physics and USD views resolve the camera in the same local frame.

        The same pose is written to both, so their current (nominal, pre-offset) local poses must match.
        """
        physics_t, physics_q = self._physics_view.get_local_poses()
        physics_t, physics_q = physics_t.torch, physics_q.torch
        usd_t, usd_q = self._usd_view.get_local_poses()
        usd_t, usd_q = usd_t.torch.to(physics_t.device), usd_q.torch.to(physics_q.device)
        assert torch.allclose(physics_t, usd_t, atol=1.0e-4), (
            "Physics FrameView and USD camera-prim local translations disagree "
            f"(physics={physics_t.tolist()}, usd={usd_t.tolist()}); one pose cannot serve both."
        )
        # Quaternions are equal up to sign; compare the absolute dot product.
        alignment = (physics_q * usd_q).sum(dim=-1).abs()
        assert bool(
            torch.all(alignment > 1.0 - 1.0e-3)
        ), f"Physics FrameView and USD camera-prim local orientations disagree (alignment={alignment.tolist()})."

    def set_local_translations(self, translations: torch.Tensor, env_ids: torch.Tensor):
        """Set local-space translations for the camera."""
        self._ensure_initialized()
        self._physics_view.set_local_poses(
            translations=translations, orientations=None, indices=wp.from_torch(env_ids.to(torch.int32))
        )
        usd_env_ids = env_ids.to(self._usd_device)
        self._usd_view.set_local_poses(
            translations=translations.to(self._usd_device),
            orientations=None,
            indices=wp.from_torch(usd_env_ids.to(torch.int32)),
        )
        self._camera.reset(env_ids)
