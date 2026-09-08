# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Make a camera's local-pose change take effect on the RTX render under Newton.

Writing ``camera._view.set_local_poses`` (or the public ``camera.set_world_poses``) updates the physics
FrameView. Under PhysX that view is Fabric-backed and the RTX renderer -- which reads the USD/Fabric camera
prim -- follows. Under Newton the FrameView (``NewtonSiteFrameView``) updates only in-memory Warp state, so
the render does not move. This writer additionally mirrors the local pose onto the USD camera prim via
``UsdFrameView`` (which ``IsaacRtxRenderer`` reads on both backends) and calls ``camera.reset`` so
``camera.data.pos_w`` reflects the new pose.

Tracked as ``[isaac-lab-camera-pose-write-bug]``: remove the USD mirror once IsaacLab's
``NewtonSiteFrameView`` mirrors poses into Fabric like ``FabricFrameView`` does; then the physics-view
write plus ``camera.reset`` suffice on both backends.
"""

from __future__ import annotations

import torch


class CameraLocalOffsetWriter:
    """Offset a camera's local pose so ``camera.data`` and the RTX render both follow it, on any backend.

    Construct one per camera; the nominal local pose is snapshotted on first use and offsets are applied
    relative to it, so they do not compound across resets.
    """

    # The physics and USD views resolve the same camera prim, so their nominal local poses must agree.
    _TRANSLATION_ATOL = 1.0e-4
    _ORIENTATION_ATOL = 1.0e-3

    def __init__(self, camera) -> None:
        self._camera = camera
        self._physics_view = None
        self._usd_view = None
        # Snapshotted on first use as (translations [N, 3], orientations xyzw [N, 4]) in the parent frame.
        self._nominal_physics_poses: tuple[torch.Tensor, torch.Tensor] | None = None
        self._nominal_usd_poses: tuple[torch.Tensor, torch.Tensor] | None = None

    def _ensure_initialized(self) -> None:
        if self._usd_view is not None:
            return
        from isaaclab.sim.views.usd_frame_view import UsdFrameView

        self._physics_view = self._camera._view
        assert self._physics_view is not None, "Camera FrameView was not initialized."
        self._usd_view = UsdFrameView(self._camera.cfg.prim_path)
        # NOTE: get_local_poses returns the orientation as xyzw despite its wxyz docstring
        # (see test_isaaclab_bug_get_local_poses.py).
        physics_t, physics_q = self._physics_view.get_local_poses()
        usd_t, usd_q = self._usd_view.get_local_poses()
        self._nominal_physics_poses = (physics_t.torch.detach().clone(), physics_q.torch.detach().clone())
        self._nominal_usd_poses = (usd_t.torch.detach().clone(), usd_q.torch.detach().clone())
        self._assert_nominal_poses_agree()

    def _assert_nominal_poses_agree(self) -> None:
        physics_t, physics_q = self._nominal_physics_poses
        usd_t = self._nominal_usd_poses[0].to(physics_t.device)
        usd_q = self._nominal_usd_poses[1].to(physics_q.device)
        assert torch.allclose(physics_t, usd_t, atol=self._TRANSLATION_ATOL), (
            "Physics FrameView and USD camera-prim nominal translations disagree "
            f"(physics={physics_t.tolist()}, usd={usd_t.tolist()}); the views resolve the camera in "
            "different local frames, so one offset cannot serve both."
        )
        # Quaternions are equal up to sign; compare the absolute dot product.
        alignment = (physics_q * usd_q).sum(dim=-1).abs()
        assert bool(
            torch.all(alignment > 1.0 - self._ORIENTATION_ATOL)
        ), f"Physics FrameView and USD camera-prim nominal orientations disagree (alignment={alignment.tolist()})."

    def apply_camera_frame_offset(self, offset_in_camera: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Offset the camera by ``offset_in_camera`` (a translation in the camera's local frame, [N, 3]).

        The offset is rotated into the parent frame and added to the nominal local translation of both the
        physics FrameView (feeds ``camera.data.pos_w``) and the USD camera prim (feeds the RTX render);
        ``camera.reset`` then refreshes ``camera.data``.
        """
        import warp as wp
        from isaaclab.utils.math import quat_apply

        self._ensure_initialized()

        physics_nominal_t, physics_nominal_q = self._nominal_physics_poses
        offset_in_parent = quat_apply(physics_nominal_q[env_ids], offset_in_camera)

        physics_translations = physics_nominal_t[env_ids] + offset_in_parent
        self._physics_view.set_local_poses(
            translations=physics_translations, orientations=None, indices=wp.from_torch(env_ids.to(torch.int32))
        )

        # TODO(alexmillane, 2026-09-08): [isaac-lab-camera-pose-write-bug] Mirror the local pose onto the
        # USD camera prim so IsaacRtxRenderer follows it under Newton, where NewtonSiteFrameView writes only
        # Warp state. Remove once NewtonSiteFrameView mirrors poses into Fabric like FabricFrameView; then
        # the physics-view write above suffices on both backends.
        usd_nominal_t = self._nominal_usd_poses[0]
        usd_env_ids = env_ids.to(usd_nominal_t.device)
        usd_translations = usd_nominal_t[usd_env_ids] + offset_in_parent.to(usd_nominal_t.device)
        self._usd_view.set_local_poses(
            translations=usd_translations, orientations=None, indices=wp.from_torch(usd_env_ids.to(torch.int32))
        )

        self._camera.reset(env_ids)
